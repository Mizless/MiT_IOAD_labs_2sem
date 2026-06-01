

from __future__ import annotations

import json
import os
import socketserver
import sys
import threading
import time
import uuid
from json import JSONDecodeError
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from lab2.config import (
    CAPTURE_REWARD,
    DRAW_REWARD,
    HOST,
    LOG_EVERY_N_GAMES,
    LOSE_PENALTY,
    QUEEN_REWARD,
    ROLLOUT_SIZE,
    STEP_PENALTY,
    TRAIN_PORT,
    VOZHD_CAPTURE_REWARD,
    WIN_REWARD,
)
from lab2.encoding import encode_board, move_to_action_index
from lab2.ppo_agent import PPOAgent, RolloutBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

agent = PPOAgent(device=DEVICE)
buffer = RolloutBuffer()
buffer_lock = threading.Lock()
pending_episode_rewards: dict[str, float] = {}
pending_lock = threading.Lock()
update_in_progress = False


def step_reward(chosen_move: dict) -> float:
    reward = STEP_PENALTY
    captured = chosen_move.get("captured") or []
    reward += CAPTURE_REWARD * len(captured)
    if chosen_move.get("kinged"):
        reward += QUEEN_REWARD
    if chosen_move.get("vozhdCaptured"):
        reward += VOZHD_CAPTURE_REWARD
    return reward


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    episode_counter = 0
    win_counters = {"White": 0, "Black": 0, "Draw": 0}

    def handle(self):
        global update_in_progress
        self.current_episode_id = None
        rfile = self.request.makefile("rb")
        wfile = self.request.makefile("wb")

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

                ep_id = msg.get("episode_id") or self.current_episode_id
                resp: dict = {"error": "unknown_type"}

                if msg_type == "start_episode":
                    ep_id = str(uuid.uuid4())
                    self.current_episode_id = ep_id
                    with pending_lock:
                        pending_episode_rewards.setdefault(ep_id, 0.0)
                    resp = {"status": "ok", "episode_id": ep_id}
                    if ThreadedTCPRequestHandler.episode_counter % LOG_EVERY_N_GAMES == 0:
                        print(f"Начат эпизод ep_id={ep_id}")

                elif msg_type == "get_move":
                    legal_moves = state_json.get("legal_moves", []) or []
                    if not legal_moves:
                        resp = {"error": "no_moves"}
                    else:
                        if ep_id is None:
                            ep_id = str(uuid.uuid4())
                            self.current_episode_id = ep_id
                            with pending_lock:
                                pending_episode_rewards.setdefault(ep_id, 0.0)

                        chosen_move, action_idx, logprob, value, mask_np = agent.select_action(
                            state_json, legal_moves
                        )
                        reward = step_reward(chosen_move)

                        with pending_lock:
                            pending = pending_episode_rewards.get(ep_id, 0.0)
                            if pending != 0.0:
                                reward += pending
                                pending_episode_rewards[ep_id] = 0.0

                        do_start_update = False
                        with buffer_lock:
                            buffer.states.append(encode_board(state_json).copy())
                            buffer.actions.append(action_idx)
                            buffer.old_logprobs.append(logprob)
                            buffer.values.append(value)
                            buffer.rewards.append(reward)
                            buffer.masks.append(mask_np.copy())
                            buffer.dones.append(False)
                            buffer.episode_ids.append(ep_id)

                            if len(buffer.states) >= ROLLOUT_SIZE and not update_in_progress:
                                update_in_progress = True
                                states_copy = [s.copy() for s in buffer.states]
                                actions_copy = buffer.actions[:]
                                old_logprobs_copy = buffer.old_logprobs[:]
                                values_copy = buffer.values[:]
                                rewards_copy = buffer.rewards[:]
                                masks_copy = [m.copy() for m in buffer.masks]
                                episode_ids_copy = buffer.episode_ids[:]
                                buffer.clear()
                                do_start_update = True

                        if do_start_update:

                            def do_update(states_c, actions_c, old_lp_c, values_c, rewards_c, masks_c, episode_ids_c):
                                global update_in_progress
                                print(f"Обновление PPO, шагов={len(states_c)}")
                                try:
                                    agent.update_from_copy(
                                        states_c,
                                        actions_c,
                                        old_lp_c,
                                        values_c,
                                        rewards_c,
                                        masks_c,
                                        episode_ids_c,
                                    )
                                    path = agent.save(tag=ThreadedTCPRequestHandler.episode_counter)
                                    print("Модель сохранена:", path)
                                except Exception as exc:
                                    print("Ошибка обновления:", exc)
                                finally:
                                    with buffer_lock:
                                        update_in_progress = False

                            threading.Thread(
                                target=do_update,
                                args=(
                                    states_copy,
                                    actions_copy,
                                    old_logprobs_copy,
                                    values_copy,
                                    rewards_copy,
                                    masks_copy,
                                    episode_ids_copy,
                                ),
                                daemon=True,
                            ).start()

                        resp = chosen_move

                elif msg_type == "end_episode":
                    winner = msg.get("winner")
                    if ep_id is None:
                        ep_id = str(uuid.uuid4())

                    if winner == player:
                        reward_delta = WIN_REWARD
                    elif winner == "Draw":
                        reward_delta = DRAW_REWARD
                    else:
                        reward_delta = -LOSE_PENALTY

                    attached = False
                    with buffer_lock:
                        for i in range(len(buffer.episode_ids) - 1, -1, -1):
                            if buffer.episode_ids[i] == ep_id:
                                buffer.rewards[i] += reward_delta
                                attached = True
                                break

                    if not attached:
                        with pending_lock:
                            pending_episode_rewards[ep_id] = pending_episode_rewards.get(ep_id, 0.0) + reward_delta

                    ThreadedTCPRequestHandler.episode_counter += 1
                    if winner in ThreadedTCPRequestHandler.win_counters:
                        ThreadedTCPRequestHandler.win_counters[winner] += 1
                    total = ThreadedTCPRequestHandler.episode_counter
                    if total % LOG_EVERY_N_GAMES == 0 or total <= 3:
                        print(f"Эпизод {total} завершён. Победитель: {winner}")
                    agent.writer.add_scalar("game/episodes", total, total)
                    resp = {"status": "ok"}

                elif msg_type == "save_model":
                    path = agent.save(tag=f"manual_{int(time.time())}")
                    resp = {"status": "saved", "path": path}

                elif msg_type == "load_model":
                    agent.load_latest_checkpoint_if_any()
                    resp = {"status": "loaded"}

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


def periodic_save(interval_sec: int = 300) -> None:
    while True:
        time.sleep(interval_sec)
        try:
            path = agent.save(tag=f"auto_{int(time.time())}")
            print("Автосохранение:", path)
        except Exception as exc:
            print("Ошибка автосохранения:", exc)


if __name__ == "__main__":
    threading.Thread(target=periodic_save, args=(300,), daemon=True).start()
    print(f"Сервер обучения: {HOST}:{TRAIN_PORT}, device={DEVICE}, pid={os.getpid()}")
    with ThreadedTCPServer((HOST, TRAIN_PORT), ThreadedTCPRequestHandler) as server:
        server.serve_forever()
