import abc
import os
from collections import Counter

import numpy as np
import pandas as pd

from lm_routing.controller import Controller

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

pd.options.mode.copy_on_write = True


class Benchmark(abc.ABC):
    """
    Benchmark class for evaluating models.

    Internally, class should handle init and manage own cache (if needed).
    """

    @abc.abstractmethod
    def evaluate(
        self,
        controller: Controller,
        router: str,
        num_results: int,
        overwrite_router_cache: bool,
    ) -> tuple[str, dict[str, int], str]:
        """Takes in a router and threshold and returns a tuple of weighted accuracy, model counts, and number of requests."""
        pass

    @abc.abstractmethod
    def get_optimal_accuracy(self, strong_percent: float) -> float:
        """Takes in % strong model calls and returns the optimal score for the benchmark given these % of calls."""
        pass

    @abc.abstractmethod
    def get_model_accuracy(self, model: str) -> float:
        """Takes in a model name and returns the accuracy of that model on the benchmark."""
        pass


class BFCLBenchmark(Benchmark):
    """
    BFCL tool-call pass/fail 데이터 기반 벤치마크.

    test_data_path: prepare_bfcl_data.py convert 가 생성하는 test_data.json 경로.
    데이터 형식:
      [{"idx": 0, "prompt": "...", "bfcl_split": "...",
        "<weak_model>": bool, "<strong_model>": bool}, ...]
    """

    def __init__(self, routed_pair, test_data_path: str, overwrite_cache):
        self.routed_pair = routed_pair
        self.overwrite_cache = overwrite_cache
        self.cache_path = os.path.join(
            os.path.dirname(os.path.abspath(test_data_path)), "bfcl_cache.npy"
        )

        try:
            self.cache = np.load(self.cache_path, allow_pickle=True).item()
        except Exception:
            self.cache = {}

        self.all_data = pd.read_json(test_data_path)
        print(f"Loaded {len(self.all_data)} BFCL test samples from {test_data_path}")

        for col in [routed_pair.strong, routed_pair.weak]:
            if col not in self.all_data.columns:
                raise ValueError(
                    f"Column '{col}' not found in test_data. "
                    f"Available: {list(self.all_data.columns)}"
                )

    def evaluate(self, controller, router, num_results, overwrite_router_cache):
        if (
            router not in self.cache
            or router in self.overwrite_cache
            or overwrite_router_cache
        ):
            strong_win_rates = controller.batch_calculate_win_rate(
                prompts=self.all_data["prompt"], router=router
            )
            self.cache[router] = strong_win_rates
            np.save(self.cache_path, self.cache)
        else:
            strong_win_rates = self.cache[router]

        _, thresholds = pd.qcut(strong_win_rates, num_results, retbins=True, duplicates="drop")
        self.all_data["strong_win_rates"] = strong_win_rates

        for i, threshold in enumerate(thresholds):
            selection = (
                self.all_data["strong_win_rates"] >= threshold
                if i != len(thresholds) - 1
                else self.all_data["strong_win_rates"] > threshold
            )
            results = np.where(
                selection,
                self.all_data[self.routed_pair.strong],
                self.all_data[self.routed_pair.weak],
            )
            models = np.where(
                selection, self.routed_pair.strong, self.routed_pair.weak
            )
            model_counts = Counter(models)
            yield threshold, sum(results) / len(results) * 100, model_counts, len(
                results
            )

    def get_model_accuracy(self, model):
        df = self.all_data
        return len(df[df[model] == True]) / len(df) * 100

    def get_optimal_accuracy(self, strong_percent):
        df = self.all_data
        total = len(df)

        strong_calls = total * strong_percent
        weak_correct = len(df[df[self.routed_pair.weak] == True])

        df_sub = df[df[self.routed_pair.weak] == False]
        df_sub = df_sub[df_sub[self.routed_pair.strong] == True]

        strong_bonus = min(strong_calls, len(df_sub))
        opt_correct = weak_correct + strong_bonus
        return opt_correct / total * 100
