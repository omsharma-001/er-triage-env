"""
agent.py — Deep Q-Network (DQN) agent for ERTriageEnv.

Architecture: 3-layer MLP with dueling heads.

  Input  : state vector (OBS_DIM[task] floats)
  Hidden : 256 → 256 → 128 (ReLU, LayerNorm for stability)
  Output : Dueling DQN
    - Value stream    V(s)        : scalar
    - Advantage stream A(s,a)     : ACTION_DIM[task] values
    - Q(s,a) = V(s) + A(s,a) - mean(A)  ← dueling combination

Why dueling?
  In ER triage, many states have a "clearly right" action (admit the dying
  patient). Dueling separates learning state-value from action advantage,
  which speeds convergence when most actions are equivalent.

Replay buffer: prioritised by recency (simple circular buffer, works well
for environments with clear temporal structure).

Training tricks:
  - Target network (hard update every TARGET_UPDATE_FREQ steps)
  - Gradient clipping (norm 1.0) — prevents exploding gradients
  - Huber loss (SmoothL1) — robust to outlier rewards
  - Action masking — -inf on invalid actions before argmax
"""

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from features import OBS_DIM, ACTION_DIM, HOLD_ACTION


# ── Replay Buffer ────────────────────────────────────────────────────────────

@dataclass
class Transition:
    state:      np.ndarray
    action:     int
    reward:     float
    next_state: np.ndarray
    done:       bool
    mask:       np.ndarray        # valid action mask for next_state
    next_mask:  np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buf: Deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition):
        self.buf.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)


# ── Dueling DQN Network ──────────────────────────────────────────────────────

class DuelingDQN(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        # Value stream
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Advantage stream
        self.advantage_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        shared = self.shared(x)
        value     = self.value_head(shared)          # (B, 1)
        advantage = self.advantage_head(shared)       # (B, A)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q                                      # (B, A)


# ── DQN Agent ────────────────────────────────────────────────────────────────

class DQNAgent:
    def __init__(
        self,
        task:               str   = "easy",
        lr:                 float = 1e-3,
        gamma:              float = 0.99,
        epsilon_start:      float = 1.0,
        epsilon_end:        float = 0.05,
        epsilon_decay:      int   = 2_000,   # steps to decay epsilon
        batch_size:         int   = 64,
        target_update_freq: int   = 200,
        buffer_capacity:    int   = 50_000,
        device:             str   = "cpu",
    ):
        assert TORCH_AVAILABLE, "PyTorch not installed. Run: pip install torch"

        self.task = task
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        obs_dim    = OBS_DIM[task]
        action_dim = ACTION_DIM[task]

        self.policy_net = DuelingDQN(obs_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(obs_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimiser = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        self.steps_done    = 0
        self.episodes_done = 0
        self.train_losses: List[float] = []

    # ── epsilon schedule ────────────────────────────────────────────────────
    @property
    def epsilon(self) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.steps_done / self.epsilon_decay)

    # ── action selection ────────────────────────────────────────────────────
    def select_action(
        self,
        state: np.ndarray,
        mask:  np.ndarray,
        greedy: bool = False,
    ) -> int:
        eps = 0.0 if greedy else self.epsilon
        if random.random() < eps:
            # Random valid action
            valid = np.where(mask > 0)[0]
            return int(np.random.choice(valid))

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t).squeeze(0)

        # Mask invalid actions with -inf
        mask_t = torch.FloatTensor(mask).to(self.device)
        q_values = q_values.masked_fill(mask_t == 0, float("-inf"))
        return int(q_values.argmax().item())

    # ── store transition ────────────────────────────────────────────────────
    def store(self, state, action, reward, next_state, done, mask, next_mask):
        self.buffer.push(Transition(
            state=state, action=action, reward=reward,
            next_state=next_state, done=done,
            mask=mask, next_mask=next_mask,
        ))
        self.steps_done += 1

    # ── training step ───────────────────────────────────────────────────────
    def train_step(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)

        states      = torch.FloatTensor(np.stack([t.state      for t in batch])).to(self.device)
        actions     = torch.LongTensor( [t.action               for t in batch]).to(self.device)
        rewards     = torch.FloatTensor([t.reward               for t in batch]).to(self.device)
        next_states = torch.FloatTensor(np.stack([t.next_state  for t in batch])).to(self.device)
        dones       = torch.FloatTensor([float(t.done)          for t in batch]).to(self.device)
        next_masks  = torch.FloatTensor(np.stack([t.next_mask   for t in batch])).to(self.device)

        # Current Q values
        q_current = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target: use policy net to SELECT, target net to EVALUATE
        with torch.no_grad():
            next_q_policy = self.policy_net(next_states)
            next_q_policy = next_q_policy.masked_fill(next_masks == 0, float("-inf"))
            next_actions  = next_q_policy.argmax(dim=1)

            next_q_target = self.target_net(next_states)
            next_q_vals   = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets       = rewards + self.gamma * next_q_vals * (1 - dones)

        loss = F.smooth_l1_loss(q_current, targets)

        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimiser.step()

        # Hard update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        loss_val = loss.item()
        self.train_losses.append(loss_val)
        return loss_val

    # ── save / load ─────────────────────────────────────────────────────────
    def save(self, path: str):
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
            "steps_done": self.steps_done,
            "episodes_done": self.episodes_done,
        }, path)
        print(f"  Saved checkpoint → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_state_dict"])
        self.target_net.load_state_dict(ckpt["target_state_dict"])
        self.optimiser.load_state_dict(ckpt["optimiser_state_dict"])
        self.steps_done    = ckpt["steps_done"]
        self.episodes_done = ckpt["episodes_done"]
        print(f"  Loaded checkpoint ← {path}")
