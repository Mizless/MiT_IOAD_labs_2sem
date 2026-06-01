

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from checkers.engine import GameEngine
from checkers.enums import SideType
from checkers.random_ai import RandomMoveAI
from checkers.rl_policy_ai import RLPolicyAI
from lab2.config import (
    BEST_CHECKPOINT,
    CHECKPOINT_DIR,
    COMPARISON_GAMES,
    COMPARISON_USE_GREEDY,
    DEFAULT_OPPONENT,
    MINIMAX_DEPTH,
    RL_ROOT_WIDTH,
    RL_SEARCH_DEPTH,
)
from lab2.game_runner import play_game


class LocalRLPlayer:
    def __init__(
        self,
        stochastic: bool = False,
        use_value_guided: bool = False,
        checkpoint: str | None = None,
        search_depth: int = RL_SEARCH_DEPTH,
        use_greedy: bool | None = None,
    ):
        ckpt = checkpoint or (str(BEST_CHECKPOINT) if BEST_CHECKPOINT.exists() else None)
        _ = search_depth
        self._policy = RLPolicyAI(checkpoint=ckpt)
        self.stochastic = stochastic

    def __call__(self, engine: GameEngine, side: SideType) -> dict:
        mv = self._policy.choose_move(engine, side)
        if mv is None:
            return {"error": "no_moves"}
        return {
            "fromRow": mv.from_y,
            "fromCol": mv.from_x,
            "toRow": mv.to_y,
            "toCol": mv.to_x,
            "captured": [],
        }


class RandomPlayer:
    def __init__(self):
        self.ai = RandomMoveAI()
        self.last_time = 0.0

    def __call__(self, engine: GameEngine, side: SideType) -> dict:
        move, stats = self.ai.choose_move_with_stats(engine, side)
        self.last_time = stats["time_sec"]
        if move is None:
            return {"error": "no_moves"}
        return {
            "fromRow": move.from_y,
            "fromCol": move.from_x,
            "toRow": move.to_y,
            "toCol": move.to_x,
            "captured": [],
        }


def create_opponent(opponent: str, minimax_depth: int = MINIMAX_DEPTH):
    _ = minimax_depth
    if opponent == "random":
        return RandomPlayer(), "Random"
    raise ValueError(f"Unknown opponent: {opponent!r}. Use 'random'.")


def run_matchup(
    rl_white: bool,
    games: int,
    opponent: str = DEFAULT_OPPONENT,
    minimax_depth: int = MINIMAX_DEPTH,
    checkpoint: str | None = None,
    use_greedy: bool | None = None,
) -> dict:
    rl = LocalRLPlayer(stochastic=False, checkpoint=checkpoint, use_greedy=use_greedy)
    baseline, opponent_label = create_opponent(opponent, minimax_depth)

    stats = {
        "rl_wins": 0,
        "opponent_wins": 0,
        "minimax_wins": 0,  # alias for backward-compatible reports
        "draws": 0,
        "rl_avg_time": 0.0,
        "opponent_avg_time": 0.0,
        "minimax_avg_time": 0.0,  # alias
        "games": games,
        "opponent": opponent,
        "opponent_label": opponent_label,
        "minimax_depth": minimax_depth if opponent == "minimax" else None,
        "rl_color": "White" if rl_white else "Black",
    }

    rl_times = []
    opp_times = []

    for i in range(games):
        if rl_white:
            white_fn, black_fn = rl, baseline
        else:
            white_fn, black_fn = baseline, rl

        t0 = time.perf_counter()
        result = play_game(white_fn, black_fn)
        _ = time.perf_counter() - t0

        rl_side = SideType.WHITE if rl_white else SideType.BLACK
        if result.winner == rl_side:
            stats["rl_wins"] += 1
        elif result.winner is None:
            stats["draws"] += 1
        else:
            stats["opponent_wins"] += 1
            stats["minimax_wins"] += 1

        rl_stat = result.white_stats if rl_white else result.black_stats
        opp_stat = result.black_stats if rl_white else result.white_stats
        rl_times.append(rl_stat.avg_move_time)
        opp_times.append(opp_stat.avg_move_time)
        print(f"  Игра {i + 1}/{games}: победитель={result.winner}, ходов={result.moves}")

    stats["rl_avg_time"] = sum(rl_times) / max(1, len(rl_times))
    stats["opponent_avg_time"] = sum(opp_times) / max(1, len(opp_times))
    stats["minimax_avg_time"] = stats["opponent_avg_time"]
    stats["rl_win_rate"] = stats["rl_wins"] / max(1, games)
    stats["speed_ratio"] = stats["opponent_avg_time"] / max(stats["rl_avg_time"], 1e-6)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Сравнение RL vs Random")
    parser.add_argument("--games", type=int, default=COMPARISON_GAMES)
    parser.add_argument("--output", default=str(CHECKPOINT_DIR / "comparison_report.json"))
    parser.add_argument("--checkpoint", default=None, help="Path to .pth checkpoint (default: best.pth)")
    args = parser.parse_args()

    ckpt = args.checkpoint
    opp_label = "Random"

    print(f"=== RL vs {opp_label} (RL белыми) ===")
    white_stats = run_matchup(rl_white=True, games=args.games, opponent="random", checkpoint=ckpt)

    print(f"\n=== RL vs {opp_label} (RL чёрными) ===")
    black_stats = run_matchup(rl_white=False, games=args.games, opponent="random", checkpoint=ckpt)

    total_opp_wins = white_stats["opponent_wins"] + black_stats["opponent_wins"]
    avg_opp_time = (white_stats["opponent_avg_time"] + black_stats["opponent_avg_time"]) / 2

    report = {
        "opponent": "random",
        "rl_as_white": white_stats,
        "rl_as_black": black_stats,
        "summary": {
            "total_rl_wins": white_stats["rl_wins"] + black_stats["rl_wins"],
            "total_opponent_wins": total_opp_wins,
            "total_minimax_wins": total_opp_wins,
            "total_draws": white_stats["draws"] + black_stats["draws"],
            "avg_rl_move_sec": (white_stats["rl_avg_time"] + black_stats["rl_avg_time"]) / 2,
            "avg_opponent_move_sec": avg_opp_time,
            "avg_minimax_move_sec": avg_opp_time,
        },
        "key_rl_feature": (
            "Self-play + PPO: политика улучшается из опыта партий; Random не обучается."
        ),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Итог ===")
    s = report["summary"]
    print(f"RL побед: {s['total_rl_wins']}")
    print(f"{opp_label} побед: {s['total_opponent_wins']}")
    print(f"Ничьи: {s['total_draws']}")
    print(f"Среднее время хода RL: {s['avg_rl_move_sec']:.4f} c")
    print(f"Среднее время хода {opp_label}: {s['avg_opponent_move_sec']:.4f} c")
    print(f"Отчёт: {out_path}")


if __name__ == "__main__":
    main()
