
from __future__ import annotations

import json
import socket
from typing import Any, Optional

from checkers.engine import GameEngine
from checkers.enums import SideType
from lab2.config import HOST, INFERENCE_PORT, TRAIN_PORT

class RLClient:
    def __init__(self, host: str = HOST, port: int = TRAIN_PORT, timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._file_in = None
        self._file_out = None
        self.episode_id: Optional[str] = None

    def connect(self) -> None:
        self.close()
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._sock.settimeout(self.timeout)
        self._file_in = self._sock.makefile("rb")
        self._file_out = self._sock.makefile("wb")

    def close(self) -> None:
        for handle in (self._file_in, self._file_out):
            try:
                if handle:
                    handle.close()
            except OSError:
                pass
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None
        self._file_in = None
        self._file_out = None
        self.episode_id = None

    def _ensure_connected(self) -> None:
        if self._sock is None:
            self.connect()

    def _send(self, payload: dict) -> dict:
        self._ensure_connected()
        assert self._file_out is not None and self._file_in is not None
        line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self._file_out.write(line)
        self._file_out.flush()
        raw = self._file_in.readline()
        if not raw:
            raise ConnectionError("Сервер закрыл соединение")
        return json.loads(raw.decode("utf-8-sig"))

    @staticmethod
    def _side_name(side: SideType) -> str:
        return "White" if side == SideType.WHITE else "Black"

    def start_episode(self, engine: GameEngine, side: SideType) -> str:
        state = engine.to_state_json(side)
        resp = self._send(
            {
                "type": "start_episode",
                "player": self._side_name(side),
                "state": json.dumps(state, ensure_ascii=False),
            }
        )
        self.episode_id = resp.get("episode_id")
        return self.episode_id or ""

    def get_move(self, engine: GameEngine, side: SideType) -> dict:
        state = engine.to_state_json(side)
        payload: dict[str, Any] = {
            "type": "get_move",
            "player": self._side_name(side),
            "state": json.dumps(state, ensure_ascii=False),
        }
        if self.episode_id:
            payload["episode_id"] = self.episode_id
        return self._send(payload)

    def end_episode(self, engine: GameEngine, side: SideType, winner: Optional[SideType]) -> None:
        winner_name = "Draw"
        if winner == SideType.WHITE:
            winner_name = "White"
        elif winner == SideType.BLACK:
            winner_name = "Black"

        payload: dict[str, Any] = {
            "type": "end_episode",
            "player": self._side_name(side),
            "state": json.dumps(engine.to_state_json(side), ensure_ascii=False),
            "winner": winner_name,
        }
        if self.episode_id:
            payload["episode_id"] = self.episode_id
        self._send(payload)
        self.episode_id = None

class InferenceClient(RLClient):
    def __init__(self, host: str = HOST, port: int = INFERENCE_PORT, timeout: float = 30.0):
        super().__init__(host=host, port=port, timeout=timeout)
