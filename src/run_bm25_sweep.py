import itertools
import subprocess


RUNS = [
    {"k1": 0.8, "b": 0.4, "top_k": 100},
    {"k1": 0.8, "b": 0.75, "top_k": 100},
    {"k1": 1.2, "b": 0.75, "top_k": 100},
    {"k1": 1.6, "b": 0.75, "top_k": 100},
    {"k1": 1.6, "b": 0.9, "top_k": 100},
]


def main():
    for run in RUNS:
        print(f"\nRunning BM25: {run}")

        cmd = [
            "python",
            "run_bm25.py",
            "--k1",
            str(run["k1"]),
            "--b",
            str(run["b"]),
            "--top_k",
            str(run["top_k"]),
            "--use_wandb",
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()