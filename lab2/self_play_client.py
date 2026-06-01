

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from checkers.enums import SideType
from lab2.config import LOG_EVERY_N_GAMES


def main():
    parser = argparse.ArgumentParser(description="Self-play для обучения RL")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Без TCP — в 10–50× быстрее (рекомендуется для 1M игр)",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log-every", type=int, default=LOG_EVERY_N_GAMES)
    args = parser.parse_args()

    if args.local:
        cmd = [
            sys.executable,
            str(ROOT / "lab2" / "local_self_play.py"),
            "--games",
            str(args.games),
            "--log-every",
            str(args.log_every),
        ]
        if args.quiet:
            cmd.append("--quiet")
        raise SystemExit(subprocess.call(cmd))

    from lab2.game_runner import play_self_play_episode
    from lab2.rl_client import RLClient

    client = RLClient(host=args.host, port=args.port)
    results = play_self_play_episode(
        client,
        episodes=args.games,
        quiet=args.quiet,
        log_every=max(1, args.log_every),
    )

    wins = {"White": 0, "Black": 0, "Draw": 0}
    for r in results:
        if r.winner == SideType.WHITE:
            wins["White"] += 1
        elif r.winner == SideType.BLACK:
            wins["Black"] += 1
        else:
            wins["Draw"] += 1

    print("\n=== Self-play статистика ===")
    print(f"Игр: {len(results)}")
    print(f"White/Black/Draw: {wins['White']}/{wins['Black']}/{wins['Draw']}")
    avg_moves = sum(r.moves for r in results) / max(1, len(results))
    print(f"Средняя длина партии: {avg_moves:.1f} ходов")


if __name__ == "__main__":
    main()
