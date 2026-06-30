"""
Local transformers-based inference pipeline for RouteLLM.

Loads both weak and strong models at startup (each on its own GPU) so the
router can dispatch without any model-loading latency per request.

Usage:
    from lm_routing.local_pipeline import LocalController

    controller = LocalController(
        routers=["mf"],
        strong_model="Qwen/Qwen3.5-9B",
        weak_model="Qwen/Qwen3.5-2B",
        strong_device="cuda:1",
        weak_device="cuda:0",
        config={"mf": {"checkpoint_path": "./bfcl_mf_model.pt", "text_dim": 384}},
    )

    response = controller.completion(
        router="mf",
        threshold=0.3,
        messages=[{"role": "user", "content": "What's the weather in Seoul?"}],
        tools=[...],   # optional, OpenAI tool format
        max_new_tokens=512,
    )
    print(response["choices"][0]["message"])
"""

import re
import json
import time
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_routing.controller import Controller, ModelPair


# ─────────────────────────────────────────────
# Single-model local pipeline
# ─────────────────────────────────────────────

class _LocalModel:
    def __init__(self, model_name: str, device: str, torch_dtype=torch.bfloat16):
        print(f"Loading {model_name} on {device} ...")
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        device_map = {"": device} if ":" in device else ("auto" if device != "cpu" else None)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.model.eval()
        print(f"  -> {model_name} ready on {device}")

    @torch.inference_mode()
    def generate(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> dict:
        """Run inference and return an OpenAI-compatible message dict."""
        apply_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools:
            apply_kwargs["tools"] = tools
        # Disable thinking mode for Qwen3 family (clean tool-call output)
        try:
            apply_kwargs["enable_thinking"] = False
        except Exception:
            pass

        text = self.tokenizer.apply_chat_template(messages, **apply_kwargs)
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return _build_response(raw, self.model_name)


# ─────────────────────────────────────────────
# Response formatting
# ─────────────────────────────────────────────

def _parse_tool_calls(text: str) -> Optional[list[dict]]:
    """Extract <tool_call>...</tool_call> blocks, then fallback to raw JSON."""
    calls = []
    for m in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        try:
            calls.append(json.loads(m.group(1).strip()))
        except json.JSONDecodeError:
            pass
    if calls:
        return calls

    stripped = text.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            return [parsed] if isinstance(parsed, dict) else parsed
        except json.JSONDecodeError:
            pass
    return None


def _build_response(raw: str, model_name: str) -> dict:
    """Wrap raw model output in an OpenAI-compatible response dict."""
    tool_calls = _parse_tool_calls(raw)

    if tool_calls:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(tc.get("arguments", {})),
                    },
                }
                for i, tc in enumerate(tool_calls)
            ],
        }
    else:
        message = {"role": "assistant", "content": raw}

    return {
        "id": f"local-{int(time.time())}",
        "object": "chat.completion",
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
    }


# ─────────────────────────────────────────────
# LocalController
# ─────────────────────────────────────────────

class LocalController(Controller):
    """
    Runs both models locally via transformers (each on its own GPU).
    Both models are loaded at __init__ time so there is no per-request
    model-loading overhead.
    """

    def __init__(
        self,
        routers: list[str],
        strong_model: str,
        weak_model: str,
        strong_device: str = "cuda:1",
        weak_device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.bfloat16,
        config: Optional[dict[str, dict[str, Any]]] = None,
        progress_bar: bool = False,
    ):
        # Initialize routers (parent __init__) — skip API-level args
        super().__init__(
            routers=routers,
            strong_model=strong_model,
            weak_model=weak_model,
            config=config,
            progress_bar=progress_bar,
        )

        # Load both models simultaneously, each on its own GPU
        self._models: dict[str, _LocalModel] = {
            weak_model: _LocalModel(weak_model, weak_device, torch_dtype),
            strong_model: _LocalModel(strong_model, strong_device, torch_dtype),
        }

    # ------------------------------------------------------------------
    # Override completion to use local inference
    # ------------------------------------------------------------------

    def completion(
        self,
        *,
        router: Optional[str] = None,
        threshold: Optional[float] = None,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        do_sample: bool = False,
        **kwargs,
    ) -> dict:
        if "model" in kwargs:
            router, threshold = self._parse_model_name(kwargs.pop("model"))

        self._validate_router_threshold(router, threshold)

        prompt = messages[-1]["content"]
        routed_model = self.routers[router].route(prompt, threshold, self.model_pair)
        self.model_counts[router][routed_model] += 1

        local_model = self._models[routed_model]
        return local_model.generate(
            messages=messages,
            tools=tools,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

    async def acompletion(self, **kwargs):
        # Run sync completion in async context (simple wrapper)
        # For true async, replace with asyncio.to_thread in Python 3.9+
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.completion(**kwargs)
        )
