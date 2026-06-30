"""
BFCL에서 여러 모델을 평가하여 eval_results.json 을 생성하는 스크립트.

prepare_bfcl_data.py embed 이후, prepare_bfcl_data.py convert 이전에 실행.

사용법:
  python routellm/evals/eval_bfcl_models.py \\
    --prompts-path ./bfcl_data/prompts.json \\
    --output-path ./eval_results.json \\
    --models Qwen/Qwen3.5-0.6B Qwen/Qwen3.5-2B Qwen/Qwen3.5-9B

GPU 분산:
  사용 가능한 GPU 수만큼 모델을 동시에 평가한다.
  예) GPU 2개, 모델 4개 → [model1‖model2] → [model3‖model4]
  GPU가 없으면 CPU에서 순차 평가.

BFCL 카테고리별 평가 기준:
  simple/multiple/parallel : ground truth 함수 호출 일치 여부
  irrelevance              : 함수 호출 없음(거부)이 정답
  live_relevance           : 함수 호출이 있으면 pass (구체적 GT 없음)
"""

import argparse
import copy
import json
import re
import time
import urllib.error
import urllib.request

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# BFCL v4 질문 데이터 (function definitions, questions)
GORILLA_RAW_BASE = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main"
    "/berkeley-function-call-leaderboard/bfcl_eval/data"
)
# BFCL v4 정답 데이터 (ground truth function calls)
GORILLA_ANSWER_BASE = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main"
    "/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer"
)

# Multi-turn 제외: 가상 환경 시뮬레이터(GorillaFileSystem 등) 없이는 정확한 평가 불가.
# 제외: BFCL_v4_memory, BFCL_v4_web_search (외부 인프라), BFCL_v4_format_sensitivity (비채점)
BFCL_SPLITS = [
    # Non-live
    "BFCL_v4_simple_python",
    "BFCL_v4_simple_java",
    "BFCL_v4_simple_javascript",
    "BFCL_v4_multiple",
    "BFCL_v4_parallel",
    "BFCL_v4_parallel_multiple",
    "BFCL_v4_irrelevance",
    # Live
    "BFCL_v4_live_simple",
    "BFCL_v4_live_multiple",
    "BFCL_v4_live_parallel",
    "BFCL_v4_live_parallel_multiple",
    "BFCL_v4_live_relevance",
    "BFCL_v4_live_irrelevance",
]


# ─────────────────────────────────────────────
# 모델 로드 / 언로드
# ─────────────────────────────────────────────

def load_model(model_name: str, device: str, load_in_4bit: bool):
    """
    device 형식:
      "cuda:0", "cuda:1"  → 해당 GPU에만 올림 (device_map={"": device})
      "cuda"              → 가용 GPU 전체에 자동 분산 (device_map="auto")
      "cpu"               → CPU
    """
    print(f"\nLoading {model_name} on {device} (4-bit={load_in_4bit}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # 배치 생성 시 left padding 필요 (generation은 항상 시퀀스 끝에서 시작)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # H100은 bfloat16이 float16보다 빠르고 수치적으로 안정적
    # Qwen3.5는 linear attention(SSM hybrid) 아키텍처이므로 attn_implementation 설정 불필요
    kwargs = {"trust_remote_code": True, "dtype": torch.bfloat16}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = {"": device} if ":" in device else "auto"
    elif device == "cpu":
        kwargs["device_map"] = None
    elif ":" in device:
        kwargs["device_map"] = {"": device}
    else:
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────
# BFCL 데이터 로드
# ─────────────────────────────────────────────

def _fetch_json(url: str, retries: int = 4) -> list[dict]:
    """
    URL에서 JSON 또는 JSONL을 다운로드하여 반환.

    raw.githubusercontent.com은 일시적으로 400/429/5xx를 반환할 때가 있어
    지수 백오프로 재시도한다. 재시도 없이 한 번 실패하면 split 하나가 통째로
    누락되어(예: irrelevance) 평가 데이터가 조용히 오염되므로 중요하다.
    """
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "lm-routing-bfcl"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read().decode("utf-8")
            try:
                data = json.loads(content)
                return data if isinstance(data, list) else [data]
            except json.JSONDecodeError:
                return [json.loads(line) for line in content.splitlines() if line.strip()]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            if attempt < retries - 1:
                wait = 2 ** attempt  # 1, 2, 4, 8s
                print(f"  [retry {attempt + 1}/{retries - 1}] {url} failed ({e}); waiting {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts: {last_err}")


def load_bfcl_by_id() -> dict:
    """모든 BFCL split을 로드하여 id → sample 딕셔너리로 반환."""
    id_to_sample = {}
    failed_splits = []
    for split in BFCL_SPLITS:
        try:
            samples = _fetch_json(f"{GORILLA_RAW_BASE}/{split}.json")
        except Exception as e:
            print(f"  [FAILED] {split}: {e}")
            failed_splits.append(split)
            continue
        for sample in samples:
            id_to_sample[sample["id"]] = sample
    print(f"Loaded {len(id_to_sample)} BFCL samples total.")
    if failed_splits:
        # 누락된 question split은 해당 샘플 전체가 fail로 처리되어 결과를 왜곡한다.
        # 긴 로그에 묻히지 않도록 크게 경고한다.
        print("\n" + "!" * 60)
        print(f"WARNING: {len(failed_splits)} split(s) failed to download after retries:")
        for s in failed_splits:
            print(f"    - {s}")
        print("Their samples will be scored as FAIL. Re-run when the network recovers.")
        print("!" * 60 + "\n")
    return id_to_sample


def load_bfcl_answers_by_id() -> dict:
    """
    possible_answer/ 디렉토리에서 정답을 로드하여 {id: ground_truth} 반환.

    answer 파일의 ID 형식: "simple_python_0"  (prefix 없음 — question 파일과 동일)

    정답 형식:
      [{"func_name": {"param": [acceptable_values], ...}}]

    다음 split은 answer 파일이 없는 것이 정상 (pass/fail이 tool call 유무로만 판단):
      irrelevance     → tool call 없으면 pass (함수가 요청에 무관한 상황)
      live_irrelevance → tool call 없으면 pass
      live_relevance  → tool call 있으면 pass (구체적 정답 없이 호출 여부만 평가)
    """
    NO_ANSWER_FILE_SPLITS = {
        "BFCL_v4_irrelevance",
        "BFCL_v4_live_irrelevance",
        "BFCL_v4_live_relevance",
    }

    id_to_answer = {}
    for split in BFCL_SPLITS:
        if split in NO_ANSWER_FILE_SPLITS:
            continue
        try:
            samples = _fetch_json(f"{GORILLA_ANSWER_BASE}/{split}.json")
        except Exception as e:
            print(f"  [skip answers] {split}: {e}")
            continue
        for sample in samples:
            # Answer file IDs have no BFCL_v4_ prefix — store as-is to match question IDs
            sample_id = sample.get("id", "")
            id_to_answer[sample_id] = sample.get("ground_truth", [])
    print(f"Loaded answers for {len(id_to_answer)} samples.")
    return id_to_answer


# ─────────────────────────────────────────────
# 프롬프트 포맷
# ─────────────────────────────────────────────

# BFCL uses Gorilla-style type names; map to OpenAPI/JSON Schema types for the model.
# Source: gorilla/berkeley-function-call-leaderboard/bfcl_eval/constants/type_mappings.py
_GORILLA_TO_OPENAPI = {
    "integer": "integer", "number": "number", "float": "number",
    "string": "string", "boolean": "boolean", "bool": "boolean",
    "array": "array", "list": "array", "tuple": "array",
    "dict": "object", "object": "object",
    "any": "string", "byte": "integer", "short": "integer",
    "long": "integer", "double": "number", "char": "string",
    "ArrayList": "array", "Array": "array",
    "HashMap": "object", "Hashtable": "object",
    "Queue": "array", "Stack": "array",
    "Any": "string", "String": "string", "Bigint": "integer",
}


def _cast_props(props: dict) -> dict:
    """Recursively map Gorilla type names to OpenAPI types in a properties dict."""
    result = copy.deepcopy(props)
    for key, val in result.items():
        if "type" not in val:
            val["type"] = "string"
        else:
            val["type"] = _GORILLA_TO_OPENAPI.get(val["type"], "string")
        if val["type"] in ("array", "object"):
            if "properties" in val:
                val["properties"] = _cast_props(val["properties"])
            elif "items" in val:
                items = val["items"]
                items["type"] = _GORILLA_TO_OPENAPI.get(items.get("type", "string"), "string")
                if items["type"] == "object" and "properties" in items:
                    items["properties"] = _cast_props(items["properties"])
    return result


def build_tools(function_list: list) -> list:
    """
    BFCL function 리스트를 OpenAI 호환 tool 형식으로 변환.

    공식 BFCL convert_to_tool() 로직을 재현:
      - parameters.type "dict" → "object"  (Gorilla → OpenAPI 타입 변환)
      - 모든 property type을 OpenAPI 타입으로 변환 (list→array, float→number 등)
      - 함수 이름의 "." → "_"  (OpenAI 함수명 규칙: ^[a-zA-Z0-9_-]{1,64}$)
    """
    tools = []
    for func in function_list:
        func = copy.deepcopy(func)
        name = re.sub(r"\.", "_", func.get("name", ""))
        params = copy.deepcopy(func.get("parameters", {"type": "object", "properties": {}}))
        params["type"] = "object"  # BFCL uses "dict"; OpenAI requires "object"
        if "properties" in params:
            params["properties"] = _cast_props(params["properties"])
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": func.get("description", ""),
                "parameters": params,
            },
        })
    return tools


def build_messages(question: list) -> list:
    """BFCL question 필드에서 첫 번째 턴 메시지를 추출한다."""
    if not question:
        return []
    first_turn = question[0] if isinstance(question[0], list) else question
    return [{"role": m["role"], "content": m["content"]} for m in first_turn]


# ─────────────────────────────────────────────
# 추론
# ─────────────────────────────────────────────

def _apply_template(tokenizer, messages: list, tools: list) -> str:
    """단일 샘플에 chat template 적용 (텍스트 반환)."""
    apply_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        apply_kwargs["tools"] = tools
    try:
        apply_kwargs["enable_thinking"] = False
    except Exception:
        pass
    return tokenizer.apply_chat_template(messages, **apply_kwargs)


def run_batch_inference(
    model,
    tokenizer,
    batch_inputs: list[tuple[list, list]],
    max_new_tokens: int,
) -> list[str]:
    """
    (messages, tools) 쌍의 배치를 한 번의 model.generate()로 추론한다.
    left padding 사용: 서로 길이가 다른 시퀀스를 왼쪽에 패딩하여 배치 생성.
    """
    texts = [_apply_template(tokenizer, msgs, tools) for msgs, tools in batch_inputs]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
    input_len = inputs["input_ids"].shape[1]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # skip_special_tokens=False: Qwen adds <tool_call>/</tool_call> as special tokens.
    # Skipping them strips the tags and breaks multi-call regex parsing (parallel/multiple).
    # <|im_end|> in the output is harmless — parse_tool_calls ignores it.
    return [
        tokenizer.decode(out[input_len:], skip_special_tokens=False)
        for out in output_ids
    ]


# ─────────────────────────────────────────────
# 출력 파싱
# ─────────────────────────────────────────────

def _parse_xml_function(inner: str) -> dict | None:
    """
    Qwen3.5의 XML 스타일 tool call 본문을 파싱한다.

        <function=NAME>
        <parameter=PNAME>
        VALUE
        </parameter>
        ...
        </function>

    → {"name": NAME, "arguments": {PNAME: VALUE, ...}}

    VALUE는 JSON으로 파싱을 시도하고(list/dict/number/bool), 실패하면 문자열로 둔다.
    """
    fmatch = re.search(r"<function=([^>]+)>(.*?)</function>", inner, re.DOTALL)
    if not fmatch:
        return None
    name = fmatch.group(1).strip()
    body = fmatch.group(2)

    args = {}
    for pmatch in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", body, re.DOTALL):
        pname = pmatch.group(1).strip()
        raw = pmatch.group(2).strip()
        try:
            args[pname] = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            args[pname] = raw
    return {"name": name, "arguments": args}


def parse_tool_calls(response: str) -> list[dict]:
    """
    모델 응답에서 tool call을 추출한다. 두 가지 출력 포맷을 모두 지원:

      포맷 A (JSON):  <tool_call>{"name": "func", "arguments": {...}}</tool_call>
      포맷 B (XML) :  <tool_call><function=func><parameter=p>v</parameter></function></tool_call>

    Qwen3.5 small 계열은 포맷 B(XML)를 사용한다.
    """
    tool_calls = []

    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    for match in pattern.finditer(response):
        inner = match.group(1).strip()

        # 포맷 A: <tool_call> 안에 JSON object/array
        try:
            obj = json.loads(inner)
            if isinstance(obj, dict):
                tool_calls.append(obj)
                continue
            if isinstance(obj, list):
                tool_calls.extend(o for o in obj if isinstance(o, dict))
                continue
        except json.JSONDecodeError:
            pass

        # 포맷 B: XML 스타일 <function=...><parameter=...>
        xml_call = _parse_xml_function(inner)
        if xml_call:
            tool_calls.append(xml_call)

    if tool_calls:
        return tool_calls

    # Fallback 1: 태그 없이 본문 전체가 XML function 블록인 경우
    if "<function=" in response:
        for fmatch in re.finditer(
            r"<function=[^>]+>.*?</function>", response, re.DOTALL
        ):
            xml_call = _parse_xml_function(fmatch.group(0))
            if xml_call:
                tool_calls.append(xml_call)
        if tool_calls:
            return tool_calls

    # Fallback 2: 태그 없이 본문 전체가 raw JSON
    stripped = response.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                tool_calls = [parsed]
            elif isinstance(parsed, list):
                tool_calls = [o for o in parsed if isinstance(o, dict)]
        except json.JSONDecodeError:
            pass

    return tool_calls


# ─────────────────────────────────────────────
# 정답 비교
# ─────────────────────────────────────────────

def _standardize(v) -> str:
    """공식 BFCL string_checker와 동일: 공백·,./-_*^ 제거, 소문자, 단따옴표→쌍따옴표."""
    s = re.sub(r"[ ,./\-_*^]", "", str(v))
    return s.lower().replace("'", '"')


def _val_matches(pred_val, acceptable_values: list) -> bool:
    """예측값이 acceptable_values 중 하나와 일치하는지 확인."""
    norm_pred = _standardize(pred_val)
    for av in acceptable_values:
        if _standardize(av) == norm_pred:
            return True
        try:
            if float(pred_val) == float(av):
                return True
        except (ValueError, TypeError):
            pass
    return False


def calls_match_bfcl(predicted: dict, gt_entry: dict) -> bool:
    """
    공식 BFCL simple_function_checker 로직에 맞춘 단일 call 비교.
    gt_entry 형식: {"func_name": {"param": [acceptable_values], ...}}

    - extra params (GT에 없는 파라미터) → fail
    - optional params ("" in acceptable_values) → 생략 허용
    - string 비교는 _standardize 적용
    """
    if not gt_entry:
        return False
    func_name = next(iter(gt_entry))
    if _standardize(predicted.get("name", "")) != _standardize(func_name):
        return False

    pred_args = predicted.get("arguments", {})
    # Some models serialise arguments as a JSON string rather than a dict
    if isinstance(pred_args, str):
        try:
            pred_args = json.loads(pred_args)
        except json.JSONDecodeError:
            return False
    gt_params = gt_entry[func_name]

    # Extra params: GT에 정의되지 않은 파라미터 → fail
    for key in pred_args:
        if key not in gt_params:
            return False

    # GT 파라미터 검사
    for key, acceptable_values in gt_params.items():
        if key not in pred_args:
            # "" in acceptable_values → optional (생략 가능)
            if "" not in acceptable_values:
                return False
        else:
            if not _val_matches(pred_args[key], acceptable_values):
                return False

    return True


def is_pass(predicted_calls: list[dict], ground_truth: list, is_irrelevance: bool) -> bool:
    """
    공식 BFCL 평가 로직에 맞춘 pass/fail 판정.

    ground_truth 형식: [{"func_name": {"param": [acceptable_values]}}]

    - simple   : len == 1 정확히 일치
    - parallel : len 정확히 일치, 순서 무관 1:1 매칭
    - multiple : len 정확히 일치, 순서 무관 1:1 매칭
    """
    if is_irrelevance:
        return len(predicted_calls) == 0

    if not ground_truth:
        # live_relevance: 구체적 GT 없음, 호출이 있으면 pass
        return len(predicted_calls) > 0

    # 공식 BFCL: 예측 call 수가 GT와 정확히 일치해야 함
    if len(predicted_calls) != len(ground_truth):
        return False

    # 순서 무관 1:1 매칭 (parallel_function_checker_no_order와 동일)
    matched = [False] * len(predicted_calls)
    for gt_entry in ground_truth:
        found = False
        for i, pred in enumerate(predicted_calls):
            if not matched[i] and calls_match_bfcl(pred, gt_entry):
                matched[i] = True
                found = True
                break
        if not found:
            return False
    return True


# ─────────────────────────────────────────────
# 자동 배치 크기 탐지
# ─────────────────────────────────────────────

def auto_batch_size(
    model,
    tokenizer,
    calib_texts: list[str],
    max_new_tokens: int,
    device: str,
    target_fraction: float = 0.8,
) -> int:
    """
    2-point GPU memory calibration to find the largest safe batch size.

    For device_map="auto" (device="cuda"), sums memory across all GPUs.
    For a specific device ("cuda:0"), measures only that GPU.
    Returns 1 for CPU or if calibration fails.
    """
    if not torch.cuda.is_available() or device == "cpu":
        return 1

    if ":" in device:
        gpu_indices = [int(device.split(":")[1])]
    else:
        gpu_indices = list(range(torch.cuda.device_count()))

    total_mem = sum(
        torch.cuda.get_device_properties(i).total_memory for i in gpu_indices
    )

    def _reset():
        for i in gpu_indices:
            torch.cuda.reset_peak_memory_stats(i)

    def _peak():
        return sum(torch.cuda.max_memory_allocated(i) for i in gpu_indices)

    def _run(texts: list[str]) -> int:
        _reset()
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        del enc, out
        torch.cuda.empty_cache()
        return _peak()

    if not calib_texts:
        return 1

    text_a = calib_texts[0]
    text_b = calib_texts[1] if len(calib_texts) > 1 else calib_texts[0]

    try:
        peak_1 = _run([text_a])
        peak_2 = _run([text_a, text_b])
    except Exception as e:
        print(f"  [auto_batch_size] calibration failed: {e} — defaulting to 1")
        return 1

    per_sample = peak_2 - peak_1
    available = total_mem * target_fraction - peak_1

    if per_sample <= 0:
        batch = 1
    else:
        batch = max(1, min(1 + int(available / per_sample), 128))

    gb, mb = 1024 ** 3, 1024 ** 2
    gpu_label = f"{len(gpu_indices)}×GPU" if len(gpu_indices) > 1 else f"GPU:{gpu_indices[0]}"
    print(
        f"  {gpu_label} memory: total={total_mem/gb:.1f}GB, "
        f"model+overhead={peak_1/gb:.2f}GB, "
        f"per_sample={per_sample/mb:.1f}MB "
        f"→ auto batch_size={batch}"
    )
    return batch


# ─────────────────────────────────────────────
# 단일 모델 평가
# ─────────────────────────────────────────────

def evaluate_model(
    model_name: str,
    prompts: list[dict],
    id_to_sample: dict,
    id_to_answer: dict,
    max_new_tokens: int,
    device: str,
    load_in_4bit: bool,
    batch_size: int = 0,
    debug_n: int = 0,
    trace_sink: list | None = None,
) -> tuple[dict[str, bool], str]:
    """한 모델을 전체 BFCL 샘플에 대해 배치 추론으로 평가하고 {id: pass} 딕셔너리 반환.

    trace_sink가 주어지면 샘플별 추적 레코드(raw 출력, 파싱 결과, 정답, pass)를
    append한다 (--save-responses 용).
    """
    model, tokenizer = load_model(model_name, device, load_in_4bit)

    results = {}
    failed_samples = 0
    debug_printed = 0

    valid = []
    for pm in prompts:
        sid = pm["id"]
        if id_to_sample.get(sid) is None:
            results[sid] = False
            failed_samples += 1
        else:
            id_to_sample[sid]["_split"] = pm.get("bfcl_split", "")
            valid.append(pm)

    # batch_size=0 → GPU 메모리 기반 자동 탐지
    if batch_size == 0:
        # 전체 데이터를 렌더링해 가장 긴 2개를 calibration에 사용.
        # _apply_template은 template 포맷팅만 하므로 전체 순회해도 빠름.
        # 첫 2개만 쓰면 짧은 irrelevance 샘플이 걸려 per_sample을 과소추정할 수 있음.
        all_texts = []
        for pm in valid:
            sid = pm["id"]
            sample = id_to_sample[sid]
            msgs = build_messages(sample.get("question", []))
            if msgs:
                tools = build_tools(sample.get("function", []))
                all_texts.append(_apply_template(tokenizer, msgs, tools))
        all_texts.sort(key=len, reverse=True)
        calib_texts = all_texts[:2]
        batch_size = auto_batch_size(model, tokenizer, calib_texts, max_new_tokens, device)

    pbar = tqdm(
        range(0, len(valid), batch_size),
        desc=model_name,
        leave=True,
    )
    for batch_start in pbar:
        batch_metas = valid[batch_start : batch_start + batch_size]

        batch_inputs, batch_ids, batch_samples = [], [], []
        for pm in batch_metas:
            sid = pm["id"]
            sample = id_to_sample[sid]
            msgs = build_messages(sample.get("question", []))
            if not msgs:
                results[sid] = False
                failed_samples += 1
                continue
            tools = build_tools(sample.get("function", []))
            batch_inputs.append((msgs, tools))
            batch_ids.append(sid)
            batch_samples.append(sample)

        if not batch_inputs:
            continue

        try:
            responses = run_batch_inference(model, tokenizer, batch_inputs, max_new_tokens)
        except Exception as e:
            # Batch too large (OOM or 32-bit index overflow) — retry one sample at a time
            print(f"\n  [warn] batch@{batch_start} failed ({type(e).__name__}), retrying sample-by-sample ...")
            torch.cuda.empty_cache()
            responses = []
            for single_input in batch_inputs:
                try:
                    r = run_batch_inference(model, tokenizer, [single_input], max_new_tokens)
                    responses.extend(r)
                except Exception as e2:
                    print(f"\n  [error] single sample failed: {e2}")
                    responses.append("")
                    failed_samples += 1

        for sid, response, sample, (msgs, tools) in zip(
            batch_ids, responses, batch_samples, batch_inputs
        ):
            split_name = sample.get("_split", "")
            is_irrelevance = "irrelevance" in split_name
            predicted_calls = parse_tool_calls(response)
            ground_truth = id_to_answer.get(sid, [])
            results[sid] = is_pass(predicted_calls, ground_truth, is_irrelevance)

            # --save-responses: 샘플별 전체 추적 레코드.
            # "왜 이 샘플이 fail이지?"를 raw 출력까지 거슬러 확인할 수 있게 한다.
            if trace_sink is not None:
                user_msg = next(
                    (m["content"] for m in reversed(msgs) if m.get("role") == "user"), ""
                )
                trace_sink.append({
                    "model": model_name,
                    "id": sid,
                    "split": split_name,
                    "prompt": user_msg,
                    "offered_tools": [t["function"]["name"] for t in tools],
                    "raw_output": response,
                    "parsed_tool_calls": predicted_calls,
                    "ground_truth": ground_truth,
                    "is_irrelevance": is_irrelevance,
                    "pass": bool(results[sid]),
                })

            # --debug: 모델 raw 출력을 직접 보여줘 tool call 생성 여부를 확인.
            # 여러 모델이 정확히 같은 정답률을 보이는 경우(=irrelevance 바닥값) 원인
            # 진단에 사용한다. batch_size=1 vs N 비교로 배치 생성 깨짐도 잡을 수 있다.
            if debug_printed < debug_n:
                debug_printed += 1
                print("\n" + "─" * 70)
                print(f"[debug] id={sid}  split={split_name}  irrelevance={is_irrelevance}")
                print(f"[debug] raw output ({len(response)} chars):")
                print(repr(response[:800]))
                print(f"[debug] parsed tool calls: {predicted_calls}")
                print(f"[debug] ground truth      : {ground_truth}")
                print(f"[debug] → pass = {results[sid]}")

    # Explicitly release GPU memory before returning so the next model can load cleanly.
    # del must happen in this scope — a helper function's del only removes its local ref.
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n_pass = sum(results.values())
    print(f"\n{model_name}: {n_pass}/{len(results)} pass ({n_pass/max(len(results),1)*100:.1f}%)")
    if failed_samples:
        print(f"  Errors/missing: {failed_samples}")

    return results, model_name


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BFCL에서 여러 모델을 평가하여 eval_results.json 생성"
    )
    parser.add_argument(
        "--prompts-path",
        type=str,
        required=True,
        help="prepare_bfcl_data.py embed 이 생성한 prompts.json 경로",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./eval_results.json",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="평가할 모델 HuggingFace ID 목록 (예: Qwen/Qwen3.5-0.6B Qwen/Qwen3.5-2B Qwen/Qwen3.5-9B)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="4-bit 양자화 (VRAM 부족 시, bitsandbytes 필요)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="배치 추론 크기. 기본값 0 = GPU 메모리(80%% 목표)에서 자동 탐지.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="처음 N개 샘플만 평가 (0=전체). 빠른 진단용. "
        "예: --limit 10 --debug 10 --batch-size 1 로 raw 출력 확인.",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="각 모델의 처음 N개 샘플 raw 출력/파싱 결과를 출력 (0=끄기). "
        "여러 모델이 동일한 정답률을 보일 때 tool call 생성 여부 진단용.",
    )
    parser.add_argument(
        "--save-responses",
        type=str,
        default=None,
        help="설정 시 모든 샘플의 추적 레코드(prompt, offered_tools, raw_output, "
        "parsed_tool_calls, ground_truth, pass)를 이 JSON 경로에 저장. "
        "'왜 fail인지' 사후 추적용.",
    )
    args = parser.parse_args()

    with open(args.prompts_path) as f:
        prompts = json.load(f)
    if args.limit and args.limit > 0:
        prompts = prompts[: args.limit]
        print(f"[--limit] Evaluating only the first {len(prompts)} samples.")
    print(f"Prompts to evaluate: {len(prompts)}")
    print(f"Models to evaluate : {args.models}")

    print("\nLoading BFCL question data from GitHub ...")
    id_to_sample = load_bfcl_by_id()

    print("\nLoading BFCL ground truth from GitHub (possible_answer/) ...")
    id_to_answer = load_bfcl_answers_by_id()

    # GPU 수 자동 탐지: 모델 하나당 사용 가능한 GPU 전체를 device_map="auto"로 사용
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus == 0:
        device = "cpu"
        print("\nNo GPU detected — evaluating on CPU.")
    else:
        device = "cuda"  # device_map="auto" → accelerate가 모든 GPU에 분산
        print(f"\nDetected {n_gpus} GPU(s) — each model uses all {n_gpus} GPU(s) via device_map=auto.")

    eval_kwargs = dict(
        prompts=prompts,
        id_to_sample=id_to_sample,
        id_to_answer=id_to_answer,
        max_new_tokens=args.max_new_tokens,
        device=device,
        load_in_4bit=args.load_in_4bit,
        batch_size=args.batch_size,
        debug_n=args.debug,
    )

    all_model_results = {}  # model_name → {id: bool}
    trace_sink = [] if args.save_responses else None

    # 모델을 순차적으로 평가 (각 모델이 전체 GPU를 사용)
    for i, model_name in enumerate(args.models):
        print(f"\n[{i+1}/{len(args.models)}] Evaluating {model_name}")
        results, name = evaluate_model(model_name, trace_sink=trace_sink, **eval_kwargs)
        all_model_results[name] = results

    # eval_results.json 생성
    output = []
    for prompt_meta in prompts:
        sid = prompt_meta["id"]
        record = {"id": sid}
        for short_name, results in all_model_results.items():
            record[f"{short_name}_pass"] = results.get(sid, False)
        output.append(record)

    with open(args.output_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # --save-responses: 샘플별 추적 레코드 저장
    if trace_sink is not None:
        with open(args.save_responses, "w") as f:
            json.dump(trace_sink, f, ensure_ascii=False, indent=2)
        n_fail = sum(1 for r in trace_sink if not r["pass"])
        print(
            f"\nSaved {len(trace_sink)} response traces → {args.save_responses}  "
            f"({n_fail} fail). Filter fails e.g.:\n"
            f"  python -c \"import json; "
            f"[print(r['id'], r['raw_output'][:120]) "
            f"for r in json.load(open('{args.save_responses}')) if not r['pass']]\""
        )

    # 최종 요약 출력
    n_total = len(output)
    print("\n" + "=" * 60)
    print("BFCL Evaluation Summary")
    print("=" * 60)
    print(f"  Total samples : {n_total}")
    for short_name, results in all_model_results.items():
        n_pass = sum(results.values())
        print(f"  {short_name:>24} : {n_pass:4d}/{n_total}  ({n_pass/max(n_total,1)*100:.1f}%)")
    print("=" * 60)

    # Per-split breakdown — split별 정답률.
    # 여러 모델이 '정확히 같은' 전체 정답률을 보이면 보통 tool call을 전혀
    # 생성하지 못해 irrelevance/relevance 바닥값에만 깔린 경우다. split별로 쪼개면
    # (예: irrelevance 100%, 나머지 0%) 그 증상이 즉시 드러난다.
    from collections import defaultdict

    split_of = {p["id"]: p.get("bfcl_split", "?") for p in prompts}
    all_splits = sorted(set(split_of.values()))
    print("\nPer-split pass rate (%):")
    header = "  {:<32}".format("split") + "".join(
        f"{name.split('/')[-1]:>14}" for name in all_model_results
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for split in all_splits:
        ids_in_split = [sid for sid, s in split_of.items() if s == split]
        n_split = len(ids_in_split)
        row = "  {:<32}".format(f"{split} (n={n_split})")
        for results in all_model_results.values():
            n_pass = sum(1 for sid in ids_in_split if results.get(sid, False))
            row += f"{n_pass / max(n_split, 1) * 100:>13.1f}%"
        print(row)
    print("=" * 60)
    print(f"\nSaved eval_results → {args.output_path}  ({n_total} samples)")

    model_shorts = list(all_model_results.keys())
    print(
        "\n[다음 단계] prepare_bfcl_data.py convert 실행 시 "
        f"--weak-model <모델명> --strong-model <모델명> 으로 지정하세요."
    )
    print(f"  평가된 모델: {model_shorts}")
