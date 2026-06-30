"""
BFCL v4 데이터로 MF 라우터 학습/평가 데이터를 준비하는 스크립트.

두 단계로 동작:
  1. embed  : BFCL 프롬프트를 multilingual-e5-small로 임베딩 → .npy 저장
  2. convert: 모델 평가 결과(pass/fail JSON)를 train_data.json / test_data.json으로 변환

사용 예시:
  # Step 1 - 임베딩 생성
  python prepare_bfcl_data.py embed --output-dir ./bfcl_data

  # Step 2 - 학습/평가 데이터 분리 변환 (기본 80/20 stratified split)
  python prepare_bfcl_data.py convert \\
    --results-path ./eval_results.json \\
    --prompts-path ./bfcl_data/prompts.json \\
    --output-dir ./bfcl_data \\
    --strong-model Qwen/Qwen3.5-9B \\
    --weak-model Qwen/Qwen3.5-2B \\
    --train-ratio 0.8

출력 파일:
  bfcl_data/train_data.json  → train_matrix_factorization.py 입력
  bfcl_data/test_data.json   → BFCLBenchmark 입력

eval_results.json 형식 (모델 실행 후 직접 생성):
  [
    {
      "id": "BFCL_v4_live_simple_0",
      "Qwen/Qwen3.5-2B_pass": true,
      "Qwen/Qwen3.5-9B_pass": false
    },
    ...
  ]
"""

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request

import numpy as np
from sentence_transformers import SentenceTransformer

# BFCL v4 데이터는 Gorilla GitHub 레포에서 직접 다운로드.
# 파일 목록 출처: gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/
GORILLA_RAW_BASE = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main"
    "/berkeley-function-call-leaderboard/bfcl_eval/data"
)

# 존재하지 않는 split은 load_bfcl_prompts()에서 자동으로 skip됨.
#
# 제외한 카테고리:
#   BFCL_v4_memory          : key-value / vector / rec_sum 메모리 백엔드 필요
#   BFCL_v4_web_search      : 실시간 웹 검색 API 필요
#   BFCL_v4_format_sensitivity : 비채점(non-scoring) 카테고리
#   BFCL_v4_multi_turn_*    : 가상 환경 시뮬레이터(GorillaFileSystem 등) 없이는 정확한 평가 불가
BFCL_SPLITS = [
    # Non-live: 전문가 큐레이션, single-turn
    ("non_live", "BFCL_v4_simple_python"),
    ("non_live", "BFCL_v4_simple_java"),
    ("non_live", "BFCL_v4_simple_javascript"),
    ("non_live", "BFCL_v4_multiple"),
    ("non_live", "BFCL_v4_parallel"),
    ("non_live", "BFCL_v4_parallel_multiple"),
    ("non_live", "BFCL_v4_irrelevance"),
    # Live: 커뮤니티 기여, single-turn
    ("live", "BFCL_v4_live_simple"),
    ("live", "BFCL_v4_live_multiple"),
    ("live", "BFCL_v4_live_parallel"),
    ("live", "BFCL_v4_live_parallel_multiple"),
    ("live", "BFCL_v4_live_relevance"),
    ("live", "BFCL_v4_live_irrelevance"),
]


def extract_prompt(sample, split_name: str) -> str:
    """샘플에서 라우팅에 사용할 프롬프트(첫 번째 유저 메시지)를 추출한다."""
    question = sample.get("question", [])
    if not question:
        return ""
    first_turn = question[0] if isinstance(question[0], list) else question
    for msg in first_turn:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "").strip()
    return ""


def _fetch_split(split: str, retries: int = 4) -> list[dict]:
    """
    GitHub raw URL에서 BFCL split JSON을 다운로드하여 반환.

    raw.githubusercontent.com의 일시적 400/429/5xx에 대비해 지수 백오프로 재시도.
    재시도 없이 실패하면 split 하나가 통째로 누락된다.
    """
    url = f"{GORILLA_RAW_BASE}/{split}.json"
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
                # JSONL: one JSON object per line
                return [json.loads(line) for line in content.splitlines() if line.strip()]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            if attempt < retries - 1:
                wait = 2 ** attempt  # 1, 2, 4, 8s
                print(f"  [retry {attempt + 1}/{retries - 1}] {split} failed ({e}); waiting {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {split} after {retries} attempts: {last_err}")


def load_bfcl_prompts() -> list[dict]:
    """BFCL v4 에서 유니크한 프롬프트를 수집한다."""
    prompts = []
    seen_ids = set()

    for category, split in BFCL_SPLITS:
        try:
            samples = _fetch_split(split)
        except Exception as e:
            print(f"  [skip] {split}: {e}")
            continue

        added = 0
        for sample in samples:
            sample_id = sample.get("id", "")
            if sample_id in seen_ids:
                continue
            seen_ids.add(sample_id)

            prompt = extract_prompt(sample, split)
            if prompt:
                prompts.append(
                    {
                        "idx": len(prompts),
                        "id": sample_id,
                        "bfcl_category": category,
                        "bfcl_split": split,
                        "prompt": prompt,
                    }
                )
                added += 1

        print(f"  [ok] {split}: +{added} (total unique: {len(prompts)})")

    return prompts


def generate_embeddings(prompts: list[dict], output_dir: str,
                        embedding_model: str = "intfloat/multilingual-e5-small"):
    """지정한 e5 모델로 임베딩 생성 후 .npy와 prompts.json 저장."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading {embedding_model} ...")
    model = SentenceTransformer(embedding_model)

    texts = [f"query: {p['prompt']}" for p in prompts]
    print(f"Encoding {len(texts)} prompts ...")
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True)

    npy_path = os.path.join(output_dir, "embeddings.npy")
    np.save(npy_path, embeddings)
    print(f"Saved embeddings → {npy_path}  shape={embeddings.shape}")

    prompts_path = os.path.join(output_dir, "prompts.json")
    with open(prompts_path, "w") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    print(f"Saved prompts    → {prompts_path}")


def stratified_split(items: list, key_fn, train_ratio: float, seed: int = 42):
    """key_fn(item) 값을 기준으로 계층적 분리를 수행한다."""
    from collections import defaultdict

    buckets = defaultdict(list)
    for item in items:
        buckets[key_fn(item)].append(item)

    rng = random.Random(seed)
    train, test = [], []
    for bucket in buckets.values():
        rng.shuffle(bucket)
        n_train = max(1, round(len(bucket) * train_ratio))
        train.extend(bucket[:n_train])
        test.extend(bucket[n_train:])

    return train, test


def convert_results_to_split_data(
    results_path: str,
    prompts_path: str,
    output_dir: str,
    strong_model: str,
    weak_model: str,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    """
    모델 평가 결과를 train_data.json(학습용)과 test_data.json(평가용)으로 변환한다.

    분리 전략: BFCL split 카테고리 기준 stratified split
      - 각 카테고리(live_simple 등)에서 train_ratio 비율만큼 학습에 사용
      - 나머지는 평가 전용 → BFCLBenchmark에서 사용

    train_data.json 형식 (train_matrix_factorization.py 호환):
      [{"model_a": "Qwen/Qwen3.5-2B", "model_b": "Qwen/Qwen3.5-9B",
        "winner": "model_b", "idx": 0}, ...]

    test_data.json 형식 (BFCLBenchmark 호환):
      [{"idx": 0, "id": "...", "prompt": "...",
        "bfcl_split": "live_simple",
        "Qwen/Qwen3.5-2B": false, "Qwen/Qwen3.5-9B": true}, ...]
    """
    with open(results_path) as f:
        results = json.load(f)
    with open(prompts_path) as f:
        prompts = json.load(f)

    id_to_prompt = {p["id"]: p for p in prompts}
    weak_key = f"{weak_model}_pass"
    strong_key = f"{strong_model}_pass"

    # 각 샘플을 레이블링
    labeled = []
    skipped = 0
    for r in results:
        weak_pass = r.get(weak_key, False)
        strong_pass = r.get(strong_key, False)

        if weak_pass and strong_pass:
            winner = "model_a"  # 둘 다 성공 → weak으로 충분
        elif not weak_pass and strong_pass:
            winner = "model_b"  # weak 실패, strong 성공 → strong 필요
        elif weak_pass and not strong_pass:
            winner = "model_a"  # weak만 성공 → weak로 보냄
        else:
            winner = "model_b"  # 둘 다 실패 → strong(frontier)으로 보냄

        prompt_meta = id_to_prompt.get(r.get("id", ""))
        if prompt_meta is None:
            skipped += 1
            continue

        labeled.append(
            {
                "idx": prompt_meta["idx"],
                "id": prompt_meta["id"],
                "prompt": prompt_meta["prompt"],
                "bfcl_split": prompt_meta["bfcl_split"],
                "bfcl_category": prompt_meta["bfcl_category"],
                weak_model: weak_pass,
                strong_model: strong_pass,
                "_winner": winner,
            }
        )

    # BFCL split 카테고리 기준 stratified split
    train_items, test_items = stratified_split(
        labeled, key_fn=lambda x: x["bfcl_split"], train_ratio=train_ratio, seed=seed
    )

    # train_data.json: MF 호환 필드(model_a/b, winner, idx) +
    # UniRoute 필요 필드(bfcl_split, per-model pass/fail) 함께 저장
    train_data = [
        {
            "model_a": weak_model,
            "model_b": strong_model,
            "winner": item["_winner"],
            "idx": item["idx"],
            "bfcl_split": item["bfcl_split"],
            weak_model: item[weak_model],
            strong_model: item[strong_model],
        }
        for item in train_items
    ]

    # test_data.json: 평가에 필요한 전체 정보 저장
    test_data = [
        {k: v for k, v in item.items() if k != "_winner"}
        for item in test_items
    ]

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train_data.json")
    test_path = os.path.join(output_dir, "test_data.json")

    with open(train_path, "w") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    # 카테고리별 분포 출력
    from collections import Counter
    train_cats = Counter(item["bfcl_category"] for item in train_items)
    test_cats = Counter(item["bfcl_category"] for item in test_items)

    # winner 분포 출력
    from collections import Counter as _Counter
    winner_dist = _Counter(item["_winner"] for item in labeled)
    weak_cnt = winner_dist.get("model_a", 0)
    strong_cnt = winner_dist.get("model_b", 0)

    print(f"\nTotal labeled  : {len(labeled)}  (skipped/missing: {skipped})")
    print(f"  → weak  (model_a): {weak_cnt} ({weak_cnt/max(len(labeled),1)*100:.1f}%)")
    print(f"  → strong(model_b): {strong_cnt} ({strong_cnt/max(len(labeled),1)*100:.1f}%)")
    print(f"Train split    : {len(train_data)} ({train_ratio*100:.0f}%)")
    print(f"  by category  : {dict(train_cats)}")
    print(f"Test split     : {len(test_data)} ({(1-train_ratio)*100:.0f}%)")
    print(f"  by category  : {dict(test_cats)}")
    print(f"\nSaved train    → {train_path}")
    print(f"Saved test     → {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Step 1: embed
    embed_parser = subparsers.add_parser("embed", help="BFCL 프롬프트 임베딩 생성")
    embed_parser.add_argument("--output-dir", type=str, default="./bfcl_data")
    embed_parser.add_argument(
        "--embedding-model", type=str, default="intfloat/multilingual-e5-small",
        help="e5-small(384d) 또는 intfloat/multilingual-e5-large(1024d) 등. "
        "여기서 쓴 모델을 train_matrix_factorization/train_uniroute의 --embedding-model에도 동일하게 지정.",
    )

    # Step 2: convert (train/test split 포함)
    convert_parser = subparsers.add_parser(
        "convert", help="모델 평가 결과 → train/test 학습 데이터 변환"
    )
    convert_parser.add_argument("--results-path", type=str, required=True)
    convert_parser.add_argument("--prompts-path", type=str, required=True)
    convert_parser.add_argument("--output-dir", type=str, default="./bfcl_data")
    convert_parser.add_argument("--strong-model", type=str, default="Qwen/Qwen3.5-9B")
    convert_parser.add_argument("--weak-model", type=str, default="Qwen/Qwen3.5-2B")
    convert_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="학습 데이터 비율 (기본값: 0.8 = 80/20 split)",
    )
    convert_parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "embed":
        print("Loading BFCL splits ...")
        prompts = load_bfcl_prompts()
        print(f"\nTotal unique prompts: {len(prompts)}")
        generate_embeddings(prompts, args.output_dir, args.embedding_model)

    elif args.command == "convert":
        convert_results_to_split_data(
            args.results_path,
            args.prompts_path,
            args.output_dir,
            args.strong_model,
            args.weak_model,
            args.train_ratio,
            args.seed,
        )

    else:
        parser.print_help()
