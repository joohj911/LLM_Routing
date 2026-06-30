import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import yaml
from pandarallel import pandarallel

from lm_routing.controller import Controller
from lm_routing.evals.benchmarks import BFCLBenchmark
from lm_routing.routers.routers import ROUTER_CLS

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def generate_results(
    df_router_result, benchmark, benchmark_name, routed_pair, output, plot_optimal=False
):
    plt.figure(figsize=(6, 5))
    for method in df_router_result["method"].unique():
        df_per_method = df_router_result[
            df_router_result["method"] == method
        ].sort_values(by=["strong_percentage"])

        plt.plot(
            df_per_method["strong_percentage"],
            df_per_method["accuracy"],
            label=f"{method}",
            marker=".",
            linestyle="-",
        )

    weak_accuracy = benchmark.get_model_accuracy(routed_pair.weak)
    print(f"{routed_pair.weak} score: {weak_accuracy}")

    strong_accuracy = benchmark.get_model_accuracy(routed_pair.strong)
    print(f"{routed_pair.strong} score: {strong_accuracy}")

    plt.axhline(
        y=weak_accuracy,
        color="grey",
        linestyle="--",
        label=routed_pair.weak,
    )
    plt.axhline(
        y=strong_accuracy,
        color="red",
        linestyle="--",
        label=routed_pair.strong,
    )

    if plot_optimal:
        optimal_accs = []
        optimal_range = range(0, 101, 10)
        for strong_percent in optimal_range:
            optimal_accs.append(benchmark.get_optimal_accuracy(strong_percent / 100))
        plt.plot(
            optimal_range,
            optimal_accs,
            label="Optimal",
            marker="x",
            linestyle="-",
        )

    plt.xlabel("Strong Model Calls (%)")
    plt.ylabel("Performance")
    plt.title(f"Router Performance ({benchmark_name})")
    plt.legend()

    file_name = f"{output}/{benchmark_name}.png"
    print("Saving plot to", file_name)
    plt.savefig(file_name, bbox_inches="tight")

    def pct_call_metric(row):
        df_per_method = df_router_result[
            df_router_result["method"] == row["method"]
        ].sort_values(by=["strong_percentage"])
        pct_calls = []

        for pct in [0.2, 0.5, 0.8]:
            pct_call = np.interp(
                pct * (strong_accuracy - weak_accuracy) + weak_accuracy,
                df_per_method["accuracy"],
                df_per_method["strong_percentage"],
            )
            pct_calls.append(f"{pct_call:.2f}%")

        return pd.Series(pct_calls)

    def auc_metric(row):
        df_per_method = df_router_result[
            df_router_result["method"] == row["method"]
        ].sort_values(by=["strong_percentage"])
        return np.trapz(
            df_per_method["accuracy"], df_per_method["strong_percentage"] / 100
        )

    def apgr_metric(row):
        df_per_method = df_router_result[
            df_router_result["method"] == row["method"]
        ].sort_values(by=["strong_percentage"])

        weak_auc = np.zeros([len(df_per_method)], dtype=float)
        weak_auc.fill(weak_accuracy)
        weak_auc = np.trapz(weak_auc, df_per_method["strong_percentage"] / 100)

        strong_auc = np.zeros([len(df_per_method)], dtype=float)
        strong_auc.fill(strong_accuracy)
        strong_auc = np.trapz(strong_auc, df_per_method["strong_percentage"] / 100)

        return (row["AUC"] - weak_auc) / (strong_auc - weak_auc)

    metrics = pd.DataFrame({"method": df_router_result["method"].unique()})
    metrics[["20% qual", "50% qual", "80% qual"]] = metrics.apply(
        pct_call_metric, axis=1
    )
    metrics["AUC"] = metrics.apply(auc_metric, axis=1)
    metrics["APGR"] = metrics.apply(apgr_metric, axis=1)
    metrics = metrics.sort_values(by=["APGR"], ascending=False)

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print("Metrics:\n", metrics)


def pretty_print_results(threshold, accuracy, model_counts, total):
    header = (
        "=" * 15
        + f" {router} with threshold {threshold} on {args.benchmark} "
        + "=" * 15
    )
    print("\n" + header)
    print("Average accuracy: {:.3f}".format(accuracy))
    print(f"Model counts: {', '.join([f'{k}: {v}' for k, v in model_counts.items()])}")
    print(
        f"Model %: {', '.join([f'{k}: {v / total * 100:.3f}%' for k, v in model_counts.items()])}"
    )
    weak_count = model_counts.get(controller.model_pair.weak, 0)
    print(f"  -> Routed to weak model: {weak_count}/{total} ({weak_count / total * 100:.1f}%)")
    print("=" * len(header) + "\n")


def print_bfcl_summary(df_router_result, benchmark, routed_pair):
    weak_acc = benchmark.get_model_accuracy(routed_pair.weak)
    strong_acc = benchmark.get_model_accuracy(routed_pair.strong)

    sep = "=" * 65
    print(f"\n{sep}")
    print("BFCL Routing Summary")
    print(sep)
    print(f"  Weak model   ({routed_pair.weak:>20}): {weak_acc:>6.1f}%")
    print(f"  Strong model ({routed_pair.strong:>20}): {strong_acc:>6.1f}%")
    print()
    print(f"  {'Router':<18} {'Threshold':>10} {'Pass Rate':>10} {'Weak%':>8} {'Strong%':>9}")
    print("  " + "-" * 57)
    for method in df_router_result["method"].unique():
        df_m = df_router_result[df_router_result["method"] == method].sort_values(
            by=["strong_percentage"]
        )
        for _, row in df_m.iterrows():
            weak_pct = 100.0 - row["strong_percentage"]
            print(
                f"  {method:<18} {row['threshold']:>10.4f} "
                f"{row['accuracy']:>9.1f}% {weak_pct:>7.1f}% {row['strong_percentage']:>8.1f}%"
            )
    print(sep + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate routers on various benchmarks."
    )
    parser.add_argument(
        "--routers",
        nargs="+",
        type=str,
        default=["random"],
        choices=list(ROUTER_CLS.keys()),
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["bfcl"],
        default="bfcl",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test_data.json generated by prepare_bfcl_data.py convert (required for bfcl benchmark)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
    )
    parser.add_argument(
        "--overwrite-cache",
        nargs="*",
        type=str,
        default=[],
        choices=list(ROUTER_CLS.keys()),
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=psutil.cpu_count(logical=False),
        help="Number of cores to use, all by default.",
    )
    parser.add_argument("--strong-model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--weak-model", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--mf-checkpoint",
        type=str,
        default=None,
        help="Shortcut: path to local .pt MF checkpoint (sets mf.checkpoint_path in config)",
    )
    parser.add_argument(
        "--uniroute-checkpoint",
        type=str,
        default=None,
        help="Shortcut: path to local .pt UniRoute checkpoint (sets uniroute.checkpoint_path in config)",
    )
    parser.add_argument(
        "--permodel-checkpoint",
        type=str,
        default=None,
        help="Shortcut: path to local .pt per-model regression router checkpoint (sets permodel.checkpoint_path in config)",
    )
    parser.add_argument("--num-results", type=int, default=10)
    parser.add_argument("--random-iters", type=int, default=10)
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="If set, saves all router results + weak/strong accuracy to this JSON file",
    )

    args = parser.parse_args()
    print(args)

    # Build config from shortcut args or YAML
    if args.config:
        config = yaml.safe_load(open(args.config, "r"))
    else:
        config = {}
        if args.mf_checkpoint:
            config["mf"] = {
                "checkpoint_path": args.mf_checkpoint,
                "strong_model": args.strong_model,
                "weak_model": args.weak_model,
            }
        if args.uniroute_checkpoint:
            config["uniroute"] = {"checkpoint_path": args.uniroute_checkpoint}
        if args.permodel_checkpoint:
            config["permodel"] = {"checkpoint_path": args.permodel_checkpoint}
        if not config:
            config = None

    pandarallel.initialize(progress_bar=True, nb_workers=args.parallel)
    controller = Controller(
        routers=args.routers,
        config=config,
        strong_model=args.strong_model,
        weak_model=args.weak_model,
        progress_bar=True,
    )

    if args.test_data is None:
        raise ValueError("--test-data path is required.")
    print("Running eval for BFCL.")
    benchmark = BFCLBenchmark(
        controller.model_pair, args.test_data, args.overwrite_cache
    )

    all_results = pd.DataFrame()
    for router in controller.routers:
        # Ensure reproducibility on a per-router basis
        random.seed(0)
        # For non-deterministic routers like random, we average over multiple runs
        if router in ["random"]:
            router_results = []
            for i in range(args.random_iters):
                for threshold, accuracy, model_counts, total in benchmark.evaluate(
                    controller, router, args.num_results, True
                ):
                    router_results.append(
                        {
                            "threshold": threshold,
                            "strong_percentage": model_counts[
                                controller.model_pair.strong
                            ]
                            / total
                            * 100,
                            "accuracy": accuracy,
                        }
                    )
            router_results_df = (
                pd.DataFrame(router_results)
                .groupby(["strong_percentage"], as_index=False)
                .mean()
            )
            router_results_df["method"] = str(router)
            all_results = pd.concat([all_results, router_results_df])
        else:
            router_results = []
            for threshold, accuracy, model_counts, total in benchmark.evaluate(
                controller, router, args.num_results, False
            ):
                print(f"Evaluating router: {router} with threshold {threshold}...")
                pretty_print_results(threshold, accuracy, model_counts, total)

                result = {
                    "method": str(router),
                    "threshold": threshold,
                    "strong_percentage": model_counts[controller.model_pair.strong]
                    / total
                    * 100,
                    "accuracy": accuracy,
                }
                router_results.append(result)
            all_results = pd.concat([all_results, pd.DataFrame(router_results)])

    generate_results(
        all_results,
        benchmark,
        args.benchmark,
        controller.model_pair,
        args.output,
    )

    if args.benchmark == "bfcl":
        print_bfcl_summary(all_results, benchmark, controller.model_pair)

    if args.output_json:
        import json as _json
        weak_acc = benchmark.get_model_accuracy(controller.model_pair.weak)
        strong_acc = benchmark.get_model_accuracy(controller.model_pair.strong)
        output_data = {
            "weak_model": controller.model_pair.weak,
            "strong_model": controller.model_pair.strong,
            "weak_only_accuracy": round(weak_acc, 4),
            "strong_only_accuracy": round(strong_acc, 4),
            "results": all_results[["method", "threshold", "strong_percentage", "accuracy"]]
            .round(4)
            .to_dict(orient="records"),
        }
        with open(args.output_json, "w") as f:
            _json.dump(output_data, f, indent=2)
        print(f"Saved router results → {args.output_json}")
