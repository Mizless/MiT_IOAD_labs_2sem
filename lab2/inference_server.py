
from __future__ import annotations

import json
import socketserver
import sys
import uuid
from json import JSONDecodeError
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from lab2.config import HOST, INFERENCE_PORT
from lab2.ppo_agent import PPOAgent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = PPOAgent(device=DEVICE)
agent.policy.eval()

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    episode_counter = 0
    win_counters = {"White": 0, "Black": 0, "Draw": 0}

    def handle(self):
        rfile = self.request.makefile("rb")
        wfile = self.request.makefile("wb")
        current_episode_id = None

        for raw_line in rfile:
            try:
                data = raw_line.decode("utf-8-sig", errors="ignore").strip()
                if not data:
                    continue
                try:
                    msg = json.loads(data)
                except JSONDecodeError:
                    idx = data.find("{")
                    if idx == -1:
                        continue
                    msg = json.loads(data[idx:])

                msg_type = msg.get("type")
                player = msg.get("player", "White")
                state_field = msg.get("state", "{}")
                state_json = json.loads(state_field) if isinstance(state_field, str) else state_field
                state_json["player"] = player
                current_episode_id = msg.get("episode_id") or current_episode_id
                resp: dict = {"error": "unknown_type"}

                if msg_type == "start_episode":
                    current_episode_id = str(uuid.uuid4())
                    resp = {"status": "ok", "episode_id": current_episode_id}

                elif msg_type == "get_move":
                    legal_moves = state_json.get("legal_moves", []) or []
                    if legal_moves:
                        resp = agent.greedy_action(state_json, legal_moves)
                    else:
                        resp = {"error": "no_moves"}

                elif msg_type == "end_episode":
                    winner = msg.get("winner")
                    ThreadedTCPRequestHandler.episode_counter += 1
                    if winner in ThreadedTCPRequestHandler.win_counters:
                        ThreadedTCPRequestHandler.win_counters[winner] += 1
                    total = ThreadedTCPRequestHandler.episode_counter
                    stats = ThreadedTCPRequestHandler.win_counters
                    print(
                        f"[Эпизод {total}] Победитель: {winner} | "
                        f"W/B/D: {stats['White']}/{stats['Black']}/{stats['Draw']}"
                    )
                    resp = {"status": "ok"}

                wfile.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
                wfile.flush()

            except Exception as exc:
                try:
                    wfile.write((json.dumps({"error": str(exc)}, ensure_ascii=False) + "\n").encode("utf-8"))
                    wfile.flush()
                except OSError:
                    pass

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True

if __name__ == "__main__":
    print(f"Сервер тестирования: {HOST}:{INFERENCE_PORT}, device={DEVICE}")
    with ThreadedTCPServer((HOST, INFERENCE_PORT), ThreadedTCPRequestHandler) as server:
        server.serve_forever()
