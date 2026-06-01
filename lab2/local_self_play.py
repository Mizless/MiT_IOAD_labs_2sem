

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from checkers.engine import GameEngine
from checkers.enums import SideType
from lab2.config import (
    BEST_CHECKPOINT,
    CAPTURE_REWARD,
    CHECKPOINT_DIR,
    LOSE_PENALTY,
    LOG_EVERY_N_GAMES,
    QUEEN_REWARD,
    ROLLOUT_SIZE,
    SAVE_EVERY_N_GAMES,
    SELF_PLAY_MAX_PLIES,
    STEP_PENALTY,
    VOZHD_CAPTURE_REWARD,
    WIN_REWARD,
)
from lab2.encoding import encode_board
from lab2.game_runner import apply_rl_move, _winner_name
from lab2.ppo_agent import PPOAgent, RolloutBuffer


def step_reward(move_json: dict) -> float:
    reward = STEP_PENALTY
    reward += CAPTURE_REWARD * len(move_json.get("captured") or [])
    if move_json.get("kinged"):
        reward += QUEEN_REWARD
    if move_json.get("vozhdCaptured"):
        reward += VOZHD_CAPTURE_REWARD
    return reward


def play_local_episode(agent: PPOAgent, buffer: RolloutBuffer) -> tuple[SideType | None, int]:
    engine = GameEngine()
    ep_white = str(uuid.uuid4())
    ep_black = str(uuid.uuid4())
    ply = 0
    winner: SideType | None = None

    while ply < SELF_PLAY_MAX_PLIES:
        winner = engine.winner()
        if winner is not None:
            break

        side = engine.current_side
        ep_id = ep_white if side == SideType.WHITE else ep_black
        state = engine.to_state_json(side)
        legal = state.get("legal_moves", [])
        if not legal:
            winner = SideType.opposite(side)
            break

        move, action, logprob, value, mask = agent.select_action(state, legal)
        enriched = apply_rl_move(engine, dict(move), side)

        buffer.states.append(encode_board(state).copy())
        buffer.actions.append(action)
        buffer.old_logprobs.append(logprob)
        buffer.values.append(value)
        buffer.rewards.append(step_reward(enriched))
        buffer.masks.append(mask)
        buffer.dones.append(False)
        buffer.episode_ids.append(ep_id)

        ply += 1
        winner = engine.winner()

    if winner is not None and buffer.episode_ids:
        for ep_id, delta in (
            (ep_white, WIN_REWARD if winner == SideType.WHITE else -LOSE_PENALTY),
            (ep_black, WIN_REWARD if winner == SideType.BLACK else -LOSE_PENALTY),
        ):
            for i in range(len(buffer.episode_ids) - 1, -1, -1):
                if buffer.episode_ids[i] == ep_id:
                    buffer.rewards[i] += delta
                    break

    return winner, ply


def maybe_update(agent: PPOAgent, buffer: RolloutBuffer) -> bool:
    if len(buffer.states) < ROLLOUT_SIZE:
        return False
    agent.update_from_copy(
        buffer.states,
        buffer.actions,
        buffer.old_logprobs,
        buffer.values,
        buffer.rewards,
        buffer.masks,
        buffer.episode_ids,
    )
    buffer.clear()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Быстрый local self-play (без TCP)")
    parser.add_argument("--games", type=int, default=1000, help="Число партий (например 1000000)")
    parser.add_argument("--log-every", type=int, default=LOG_EVERY_N_GAMES)
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY_N_GAMES)
    parser.add_argument("--quiet", action="store_true", help="Минимум вывода")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(device=device)
    buffer = RolloutBuffer()

    wins = {"White": 0, "Black": 0, "Draw": 0}
    total_moves = 0
    updates = 0
    t0 = time.perf_counter()

    for game_idx in range(1, args.games + 1):
        winner, moves = play_local_episode(agent, buffer)
        total_moves += moves
        name = _winner_name(winner)
        if name in wins:
            wins[name] += 1

        if maybe_update(agent, buffer):
            updates += 1

        if game_idx % args.save_every == 0:
            agent.save(tag=f"local_{game_idx}")
            agent.save_best()

        if not args.quiet and game_idx % args.log_every == 0:
            elapsed = time.perf_counter() - t0
            gps = game_idx / max(elapsed, 1e-6)
            print(
                f"[{game_idx}/{args.games}] W/B/D={wins['White']}/{wins['Black']}/{wins['Draw']} "
                f"updates={updates} speed={gps:.1f} games/s",
                flush=True,
            )

    if maybe_update(agent, buffer):
        updates += 1

    agent.save(tag=f"local_{args.games}")
    agent.save_best()

    elapsed = time.perf_counter() - t0
    stats = {
        "games": args.games,
        "wins": wins,
        "avg_moves": total_moves / max(1, args.games),
        "updates": updates,
        "elapsed_sec": elapsed,
        "games_per_sec": args.games / max(elapsed, 1e-6),
        "checkpoint": str(BEST_CHECKPOINT),
    }
    report = CHECKPOINT_DIR / "local_self_play_report.json"
    report.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Local self-play завершён ===")
    print(f"Игр: {args.games}, White/Black/Draw: {wins['White']}/{wins['Black']}/{wins['Draw']}")
    print(f"Средняя длина: {stats['avg_moves']:.1f} ходов, обновлений PPO: {updates}")
    print(f"Скорость: {stats['games_per_sec']:.2f} games/s ({elapsed:.1f} c)")
    print(f"Best: {BEST_CHECKPOINT}")
    print(f"Отчёт: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
