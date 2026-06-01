
from __future__ import annotations

import numpy as np

from lab2.config import ACTION_SPACE, BOARD_SIZE, INPUT_DIM, NUM_CELLS

def encode_board(state_json: dict) -> np.ndarray:
    arr = np.zeros(INPUT_DIM, dtype=np.float32)
    c_white_reg = 0
    c_white_queen = NUM_CELLS
    c_black_reg = NUM_CELLS * 2
    c_black_queen = NUM_CELLS * 3
    c_white_vozhd = NUM_CELLS * 4
    c_black_vozhd = NUM_CELLS * 5
    c_side = NUM_CELLS * 6

    for piece in state_json.get("pieces", []) or []:
        row = int(piece.get("row", 0))
        col = int(piece.get("col", 0))
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            continue
        idx = row * BOARD_SIZE + col
        owner = piece.get("owner", "White")
        piece_type = piece.get("pieceType", "")
        is_vozhd = bool(piece.get("isVozhd", False)) or "KING" in piece_type
        is_king = bool(piece.get("isKing", False)) or "QUEEN" in piece_type or is_vozhd

        if owner == "White":
            if is_vozhd:
                arr[c_white_vozhd + idx] = 1.0
            elif is_king:
                arr[c_white_queen + idx] = 1.0
            else:
                arr[c_white_reg + idx] = 1.0
        else:
            if is_vozhd:
                arr[c_black_vozhd + idx] = 1.0
            elif is_king:
                arr[c_black_queen + idx] = 1.0
            else:
                arr[c_black_reg + idx] = 1.0

    player = state_json.get("player", "White")
    arr[c_side : c_side + NUM_CELLS] = 1.0 if player == "White" else -1.0
    return arr

def move_to_action_index(move: dict) -> int:
    fr = int(move["fromRow"])
    fc = int(move["fromCol"])
    tr = int(move["toRow"])
    tc = int(move["toCol"])
    return (fr * BOARD_SIZE + fc) * NUM_CELLS + (tr * BOARD_SIZE + tc)

def action_index_to_move(action_index: int) -> dict:
    from_idx = action_index // NUM_CELLS
    to_idx = action_index % NUM_CELLS
    fr = from_idx // BOARD_SIZE
    fc = from_idx % BOARD_SIZE
    tr = to_idx // BOARD_SIZE
    tc = to_idx % BOARD_SIZE
    return {"fromRow": fr, "fromCol": fc, "toRow": tr, "toCol": tc, "captured": []}

def mask_legal_actions(legal_moves: list[dict]) -> np.ndarray:
    mask = np.zeros(ACTION_SPACE, dtype=np.bool_)
    for move in legal_moves:
        try:
            idx = move_to_action_index(move)
            if 0 <= idx < ACTION_SPACE:
                mask[idx] = True
        except (KeyError, TypeError, ValueError):
            continue
    if not mask.any() and legal_moves:
        mask[move_to_action_index(legal_moves[0])] = True
    elif not mask.any():
        mask[0] = True
    return mask
