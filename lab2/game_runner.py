

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from checkers.engine import GameEngine
from checkers.enums import SideType


MoveFn = Callable[[GameEngine, SideType], Optional[dict]]


@dataclass
class PlayerStats:
    total_move_time: float = 0.0
    move_count: int = 0

    @property
    def avg_move_time(self) -> float:
        if self.move_count == 0:
            return 0.0
        return self.total_move_time / self.move_count


@dataclass
class GameResult:
    winner: Optional[SideType]
    moves: int = 0
    white_stats: PlayerStats = field(default_factory=PlayerStats)
    black_stats: PlayerStats = field(default_factory=PlayerStats)
    reason: str = ""


def _winner_name(winner: Optional[SideType]) -> str:
    if winner == SideType.WHITE:
        return "White"
    if winner == SideType.BLACK:
        return "Black"
    return "Draw"


def apply_rl_move(engine: GameEngine, move_json: dict, side: SideType) -> dict:
    """Apply move with mandatory capture chain (same as Minimax simulation)."""
    engine.current_side = side
    move = GameEngine.move_from_json(move_json)
    snapshot = engine.copy()
    applied = engine.make_move(move)

    captured_cells: list[tuple[int, int]] = []
    kinged = False
    vozhd_captured = False
    for info in applied:
        captured_cells.extend(info.captured)
        kinged = kinged or info.kinged
        for row, col in info.captured:
            if snapshot.field.type_piece(col, row).name.endswith("_KING"):
                vozhd_captured = True

    move_json = dict(move_json)
    move_json["captured"] = [[r, c] for r, c in captured_cells]
    move_json["kinged"] = kinged
    move_json["vozhdCaptured"] = vozhd_captured
    return move_json


def play_game(
    white_move_fn: MoveFn,
    black_move_fn: MoveFn,
    max_plies: int = 500,
    on_move: Optional[Callable[[int, SideType, dict], None]] = None,
) -> GameResult:
    engine = GameEngine()
    result = GameResult(winner=None)
    ply = 0

    while ply < max_plies:
        winner = engine.winner()
        if winner is not None:
            result.winner = winner
            result.reason = "terminal"
            break

        side = engine.current_side
        fn = white_move_fn if side == SideType.WHITE else black_move_fn
        stats = result.white_stats if side == SideType.WHITE else result.black_stats

        t0 = time.perf_counter()
        move_json = fn(engine, side)
        stats.total_move_time += time.perf_counter() - t0
        stats.move_count += 1

        if not move_json or "error" in move_json:
            result.winner = SideType.opposite(side)
            result.reason = "no_legal_response"
            break

        apply_rl_move(engine, move_json, side)
        ply += 1
        result.moves = ply
        if on_move:
            on_move(ply, side, move_json)

    if result.winner is None:
        result.reason = "move_limit"
        white_score = engine.field.white_score
        black_score = engine.field.black_score
        if white_score > black_score:
            result.winner = SideType.WHITE
        elif black_score > white_score:
            result.winner = SideType.BLACK

    return result


def play_self_play_episode(client, episodes: int = 1, quiet: bool = False, log_every: int = 1) -> list[GameResult]:
    """Agent vs agent through RL training server (two logical clients)."""
    from lab2.rl_client import RLClient

    white_client = client if isinstance(client, RLClient) else RLClient()
    black_client = RLClient(host=white_client.host, port=white_client.port)
    white_client.connect()
    black_client.connect()

    results: list[GameResult] = []
    try:
        for ep_idx in range(episodes):
            engine = GameEngine()

            white_client.start_episode(engine, SideType.WHITE)
            black_client.start_episode(engine, SideType.BLACK)

            def white_fn(eng, side):
                return white_client.get_move(eng, side)

            def black_fn(eng, side):
                return black_client.get_move(eng, side)

            result = play_game(white_fn, black_fn)
            winner_name = _winner_name(result.winner)
            white_client.end_episode(engine, SideType.WHITE, result.winner)
            black_client.end_episode(engine, SideType.BLACK, result.winner)
            results.append(result)

            if not quiet and (ep_idx + 1) % log_every == 0:
                print(
                    f"Self-play [{ep_idx + 1}/{episodes}]: "
                    f"победитель={winner_name}, ходов={result.moves}, причина={result.reason}"
                )
    finally:
        white_client.close()
        black_client.close()

    return results
