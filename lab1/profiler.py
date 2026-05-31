"""Замер времени выполнения алгоритмов."""

import time

from tqdm import tqdm

from grid_ops import reset_grid


def measure_once(target) -> float:
    reset_grid()
    started = time.perf_counter()
    target()
    finished = time.perf_counter()
    return finished - started


def average_runtime(target, repeats: int = 1000, label: str = "") -> float:
    elapsed_total = 0.0
    for _ in tqdm(range(repeats), desc=f"Тестирование {label}", unit="итерация"):
        elapsed_total += measure_once(target)
    return elapsed_total / repeats
