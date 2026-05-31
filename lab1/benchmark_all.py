"""Запуск бенчмарков для нескольких значений BOARD_DIM без правки settings.py."""

import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

DIMENSIONS = [8, 10, 16, 20]
DEFAULT_REPEATS = 1000
OUTPUT_DIR = Path(__file__).resolve().parent


RECURSIVE_REPEATS = {
    8: 1000,
    10: 1000,
    16: 1000,
    20: 50,
}


def configure_board(dim: int) -> None:
    """Обновляет размер доски и перезагружает зависимые модули."""
    import settings

    settings.BOARD_DIM = dim
    settings.placement_grid = np.zeros((dim, dim), dtype=int)

    modules = [
        "grid_ops",
        "piece",
        "solver_stochastic",
        "solver_backtrack",
        "solver_cooperative",
        "profiler",
    ]
    for name in modules:
        if name in sys.modules:
            importlib.reload(sys.modules[name])
        else:
            importlib.import_module(name)


def prepare_numpy(dim: int) -> None:
    scratch = np.zeros((dim, dim), dtype=int)
    np.diagonal(scratch, offset=1)
    np.fliplr(scratch)
    indices = np.arange(dim)
    np.random.shuffle(indices)
    np.random.randint(0, dim, size=dim)


def run_single_dim(dim: int) -> dict[str, float | int]:
    configure_board(dim)
    prepare_numpy(dim)

    from profiler import average_runtime
    from solver_backtrack import backtrack_search
    from solver_cooperative import cooperative_search
    from solver_stochastic import stochastic_greedy

    recursive_repeats = RECURSIVE_REPEATS.get(dim, DEFAULT_REPEATS)
    results: dict[str, float | int] = {
        "brute_force": average_runtime(
            stochastic_greedy, repeats=DEFAULT_REPEATS, label="brute_force"
        ),
        "recursive": average_runtime(
            backtrack_search, repeats=recursive_repeats, label="recursive"
        ),
        "multi_agents": average_runtime(
            cooperative_search, repeats=DEFAULT_REPEATS, label="multi_agents"
        ),
        "repeats_brute_force": DEFAULT_REPEATS,
        "repeats_recursive": recursive_repeats,
        "repeats_multi_agents": DEFAULT_REPEATS,
    }

    for key in ("brute_force", "recursive", "multi_agents"):
        print(f"n={dim} | {key}: {results[key]:.6f} ({results[f'repeats_{key}']} прогонов)")

    return results


def main() -> None:
    all_results: dict[str, dict[str, float]] = {}
    started = time.perf_counter()

    for dim in DIMENSIONS:
        print(f"\n=== BOARD_DIM = {dim} ===")
        all_results[str(dim)] = run_single_dim(dim)

    payload = {
        "default_repeats": DEFAULT_REPEATS,
        "recursive_repeats": RECURSIVE_REPEATS,
        "dimensions": DIMENSIONS,
        "results": all_results,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }

    json_path = OUTPUT_DIR / "benchmark_results.json"
    txt_path = OUTPUT_DIR / "benchmark_results.txt"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"Бенчмарк N-ферзей (по умолчанию {DEFAULT_REPEATS} итераций на алгоритм)",
        f"Общее время прогона: {payload['elapsed_seconds']} с",
        "",
    ]
    header = f"{'n':>4} | {'brute_force':>14} | {'recursive':>14} | {'multi_agents':>14} | rec.runs"
    lines.append(header)
    lines.append("-" * len(header))
    for dim in DIMENSIONS:
        row = all_results[str(dim)]
        lines.append(
            f"{dim:>4} | {row['brute_force']:>14.6f} | "
            f"{row['recursive']:>14.6f} | {row['multi_agents']:>14.6f} | "
            f"{row['repeats_recursive']}"
        )

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
