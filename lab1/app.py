"""Точка входа: прогрев NumPy и сравнение трёх алгоритмов."""

import numpy as np

from settings import BOARD_DIM
from solver_stochastic import stochastic_greedy
from solver_backtrack import backtrack_search
from solver_cooperative import cooperative_search
from profiler import average_runtime


def prepare_numpy() -> None:
    """Прогрев операций NumPy перед замерами."""
    scratch = np.zeros((BOARD_DIM, BOARD_DIM), dtype=int)
    np.diagonal(scratch, offset=1)
    np.fliplr(scratch)
    indices = np.arange(BOARD_DIM)
    np.random.shuffle(indices)
    np.random.randint(0, BOARD_DIM, size=BOARD_DIM)


def run_benchmarks() -> None:
    prepare_numpy()

    print(
        f"Ср. время выполнения brute_force: "
        f"{average_runtime(stochastic_greedy, label='brute_force')}"
    )
    print(
        f"Ср. время выполнения recursive: "
        f"{average_runtime(backtrack_search, label='recursive')}"
    )
    print(
        f"Ср. время выполнения multi_agents: "
        f"{average_runtime(cooperative_search, label='multi_agents')}"
    )


if __name__ == "__main__":
    run_benchmarks()
