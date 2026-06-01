
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from lab2.config import (
    BEST_CHECKPOINT,
    CHECKPOINT_DIR,
    CLIP_EPS,
    ENTROPY_COEF,
    GAMMA,
    LAMBDA,
    LR,
    MINIBATCH_SIZE,
    RUN_DIR,
    UPDATE_EPOCHS,
    VALUE_COEF,
)
from lab2.encoding import action_index_to_move, encode_board, mask_legal_actions, move_to_action_index
from lab2.policy import Policy

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.old_logprobs: list[float] = []
        self.values: list[float] = []
        self.rewards: list[float] = []
        self.masks: list[np.ndarray] = []
        self.dones: list[bool] = []
        self.episode_ids: list[str] = []

class PPOAgent:
    def __init__(
        self,
        device: Optional[torch.device] = None,
        writer: Optional[SummaryWriter] = None,
        load_checkpoint: bool = True,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = Policy().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: max(0.1, 1 - step / 1_000_000))
        self.step = 0
        self.writer = writer or SummaryWriter(RUN_DIR)
        if load_checkpoint:
            self.load_latest_checkpoint_if_any()

    def select_action(self, state_json: dict, legal_moves: list[dict]):
        state_np = encode_board(state_json)
        state = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.policy(state)
        logits = logits.squeeze(0)
        value_f = float(value.item())

        mask_np = mask_legal_actions(legal_moves)
        mask_t = torch.from_numpy(mask_np).to(self.device)
        masked_logits = logits.clone()
        masked_logits[~mask_t] = -1e9

        dist = torch.distributions.Categorical(logits=masked_logits)
        action = int(dist.sample().item())
        logprob = float(dist.log_prob(torch.tensor(action, device=self.device)).item())

        chosen = next((mv for mv in legal_moves if move_to_action_index(mv) == action), None)
        if chosen is None:
            chosen = action_index_to_move(action)
        return chosen, action, logprob, value_f, mask_np.copy()

    def greedy_action(self, state_json: dict, legal_moves: list[dict]):
        state_np = encode_board(state_json)
        state = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.policy(state)
        logits = logits.squeeze(0).cpu().numpy()

        best_move = None
        best_val = -1e9
        for move in legal_moves:
            idx = move_to_action_index(move)
            if logits[idx] > best_val:
                best_val = logits[idx]
                best_move = move
        if best_move is None:
            best_move = legal_moves[0] if legal_moves else action_index_to_move(0)
        return best_move

    def estimate_value(self, state_json: dict) -> float:
        state_np = encode_board(state_json)
        state = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, value = self.policy(state)
        return float(value.item())

    def value_guided_action(self, engine, side, legal_moves: list[dict] | None = None):
        from checkers.enums import SideType
        from lab2.game_runner import apply_rl_move

        if legal_moves is None:
            legal_moves = engine.to_state_json(side).get("legal_moves", [])
        if not legal_moves:
            return action_index_to_move(0)

        best_move = None
        best_score = -1e18

        for move in legal_moves:
            child = engine.copy()
            child.current_side = side
            apply_rl_move(child, dict(move), side)

            winner = child.winner()
            if winner == side:
                return move
            if winner is not None and winner != side:
                score = -1e6
            else:
                opp_state = child.to_state_json(child.current_side)
                score = -self.estimate_value(opp_state)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move or self.greedy_action(engine.to_state_json(side), legal_moves)

    def supervised_train(
        self,
        states: list[np.ndarray],
        actions: list[int],
        masks: list[np.ndarray],
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 1e-3,
    ) -> float:
        if not states:
            return 0.0

        opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy.train()
        n = len(states)
        total_loss = 0.0
        steps = 0

        for _ in range(epochs):
            perm = np.random.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                batch_states = torch.from_numpy(np.vstack([states[i] for i in idx])).float().to(self.device)
                batch_actions = torch.tensor([actions[i] for i in idx], dtype=torch.long, device=self.device)
                batch_masks = torch.from_numpy(np.vstack([masks[i] for i in idx]).astype(np.bool_)).to(self.device)

                logits, _ = self.policy(batch_states)
                masked_logits = logits.clone()
                masked_logits[~batch_masks] = -1e9
                loss = torch.nn.functional.cross_entropy(masked_logits, batch_actions)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                opt.step()
                total_loss += float(loss.item())
                steps += 1

        self.policy.train()
        return total_loss / max(1, steps)

    def update_from_copy(
        self,
        states_np,
        actions_np,
        old_logprobs_np,
        values_np,
        rewards_np,
        masks_np,
        episode_ids_np,
        update_epochs: int = UPDATE_EPOCHS,
        minibatch_size: int = MINIBATCH_SIZE,
    ) -> None:
        states = torch.from_numpy(np.vstack([s.copy() for s in states_np])).float().to(self.device)
        actions = torch.tensor(actions_np, dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor(old_logprobs_np, dtype=torch.float32, device=self.device)
        masks = torch.from_numpy(np.vstack([m.copy() for m in masks_np]).astype(np.bool_)).to(self.device)

        values = list(values_np)
        n = len(rewards_np)
        advantages: list[float] = []
        gae = 0.0
        for t in reversed(range(n)):
            if t + 1 < n and episode_ids_np[t] == episode_ids_np[t + 1]:
                next_value = values[t + 1]
                same_episode = True
            else:
                next_value = 0.0
                same_episode = False
            delta = rewards_np[t] + GAMMA * next_value - values[t]
            gae = delta + GAMMA * LAMBDA * gae if same_episode else delta
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values)]
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

        entropy_coef = max(0.0005, ENTROPY_COEF * (0.999**self.step))
        total = states.shape[0]

        for _ in range(update_epochs):
            perm = torch.randperm(total, device=self.device)
            for start in range(0, total, minibatch_size):
                idx = perm[start : start + minibatch_size]
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_logprobs = old_logprobs[idx]
                mb_returns = returns_t[idx]
                mb_adv = adv_t[idx]
                mb_masks = masks[idx]

                logits, values_pred = self.policy(mb_states)
                masked_logits = logits.clone()
                masked_logits[~mb_masks] = -1e9
                dist = torch.distributions.Categorical(logits=masked_logits)
                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = VALUE_COEF * ((mb_returns - values_pred) ** 2).mean()
                loss = policy_loss + value_loss - entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                self.scheduler.step()

        self.writer.add_scalar("train/entropy_coef", entropy_coef, self.step)
        self.step += 1

    def save(self, tag=None) -> str:
        if tag is None:
            tag = int(time.time())
        path = CHECKPOINT_DIR / f"ppo_{tag}.pth"
        torch.save(
            {
                "policy_state": self.policy.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "step": self.step,
            },
            path,
        )
        with open(CHECKPOINT_DIR / "latest.txt", "w", encoding="utf-8") as f:
            f.write(path.name)
        return str(path)

    def save_best(self) -> str:
        path = str(BEST_CHECKPOINT)
        torch.save(
            {
                "policy_state": self.policy.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "step": self.step,
            },
            path,
        )
        with open(CHECKPOINT_DIR / "latest.txt", "w", encoding="utf-8") as f:
            f.write("best.pth")
        return path

    def load_best_if_any(self) -> bool:
        if not BEST_CHECKPOINT.exists():
            return False
        checkpoint = torch.load(str(BEST_CHECKPOINT), map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state"])
        if "optimizer_state" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            except Exception:
                pass
        self.step = checkpoint.get("step", 0)
        return True

    def load_latest_checkpoint_if_any(self) -> None:
        if self.load_best_if_any():
            print("Загружен best чекпоинт:", BEST_CHECKPOINT)
            return
        latest_file = CHECKPOINT_DIR / "latest.txt"
        if not latest_file.exists():
            return
        try:
            name = latest_file.read_text(encoding="utf-8").strip()
            candidate = CHECKPOINT_DIR / name
            if candidate.exists():
                checkpoint = torch.load(candidate, map_location=self.device)
                self.policy.load_state_dict(checkpoint["policy_state"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                self.step = checkpoint.get("step", 0)
                print("Загружен чекпоинт:", candidate)
        except Exception as exc:
            print("Ошибка загрузки чекпоинта:", exc)

    def load_path(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state"])
        self.step = checkpoint.get("step", 0)
        self.policy.eval()
