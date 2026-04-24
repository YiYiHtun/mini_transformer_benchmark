import argparse
import csv
import os
from types import SimpleNamespace

from train import train_one_run
from utils import ensure_dir


DEFAULT_VARIANTS = [
    {
        "run_name": "A_pos_heads1_layers1",
        "use_positional_encoding": True,
        "num_heads": 1,
        "num_layers": 1,
    },
    {
        "run_name": "B_pos_heads4_layers1",
        "use_positional_encoding": True,
        "num_heads": 4,
        "num_layers": 1,
    },
    {
        "run_name": "C_no_pos_heads4_layers1",
        "use_positional_encoding": False,
        "num_heads": 4,
        "num_layers": 1,
    },
    {
        "run_name": "D_pos_heads4_layers2",
        "use_positional_encoding": True,
        "num_heads": 4,
        "num_layers": 2,
    },
]


def save_benchmark_csv(results, path):
    fieldnames = [
        "run_name",
        "use_positional_encoding",
        "num_heads",
        "num_layers",
        "best_val_acc",
        "test_acc",
        "train_time_readable",
        "param_count",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fieldnames})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="train.csv")
    parser.add_argument("--val_csv", type=str, default="validation.csv")
    parser.add_argument("--test_csv", type=str, default="test.csv")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verify_labels", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    benchmark_results = []
    for variant in DEFAULT_VARIANTS:
        print("\n" + "=" * 80)
        print(f"Running benchmark variant: {variant['run_name']}")
        print("=" * 80)

        run_args = SimpleNamespace(
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            test_csv=args.test_csv,
            output_dir=args.output_dir,
            run_name=variant["run_name"],
            d_model=args.d_model,
            d_ff=args.d_ff,
            num_heads=variant["num_heads"],
            num_layers=variant["num_layers"],
            dropout=args.dropout,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            seed=args.seed,
            cpu=args.cpu,
            verify_labels=args.verify_labels,
            use_positional_encoding=variant["use_positional_encoding"],
            no_positional_encoding=not variant["use_positional_encoding"],
        )
        result = train_one_run(run_args)
        benchmark_results.append(result)

    csv_path = os.path.join(args.output_dir, "benchmark_results.csv")
    save_benchmark_csv(benchmark_results, csv_path)

    print("\nBenchmark summary")
    print(f"Saved to: {csv_path}")
    for row in benchmark_results:
        print(
            f"{row['run_name']}: "
            f"val={row['best_val_acc']:.4f}, "
            f"test={row['test_acc']:.4f}, "
            f"time={row['train_time_readable']}, "
            f"params={row['param_count']}"
        )
