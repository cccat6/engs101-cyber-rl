"""Frozen teacher DQN architecture.

The matching checkpoints are trained and owned by the course staff. This
module is course implementation support, not part of the student API.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DuelingQNetwork(nn.Module):
    def __init__(self, observation_dim: int, num_actions: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, num_actions)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.features(features)
        value = self.value(hidden)
        advantage = self.advantage(hidden)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class TeacherDQNAgent:
    def __init__(self, config) -> None:
        self.device = torch.device(config.device)
        self.num_actions = config.num_actions
        self.total_steps = max(1, config.total_steps)
        self.model = DuelingQNetwork(
            config.observation_dim,
            config.num_actions,
        ).to(self.device)
        self.target = DuelingQNetwork(
            config.observation_dim,
            config.num_actions,
        ).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.replay = deque(maxlen=5000)
        self.rng = random.Random(config.seed)
        self.steps = 0

    def reset(self, seed: int) -> None:
        self.rng.seed(seed)

    def act(self, observation, training: bool) -> int:
        valid = np.flatnonzero(observation.action_mask)
        if training:
            progress = min(1.0, self.steps / (0.7 * self.total_steps))
            epsilon = 1.0 + progress * (0.05 - 1.0)
            if self.rng.random() < epsilon:
                return int(valid[self.rng.randrange(len(valid))])

        features = torch.as_tensor(
            observation.features,
            dtype=torch.float32,
            device=self.device,
        )
        mask = torch.as_tensor(
            observation.action_mask,
            dtype=torch.bool,
            device=self.device,
        )
        with torch.no_grad():
            values = self.model(features).masked_fill(~mask, -torch.inf)
        return int(values.argmax().item())

    def learn(self, transition) -> None:
        self.steps += 1
        self.replay.append(
            (
                transition.observation.features.copy(),
                int(transition.action),
                float(transition.reward),
                transition.next_observation.features.copy(),
                transition.next_observation.action_mask.copy(),
                float(transition.done),
            )
        )
        if len(self.replay) < 128 or self.steps % 4:
            return

        batch = self.rng.sample(list(self.replay), 64)
        observations = torch.as_tensor(
            np.stack([item[0] for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            [item[1] for item in batch],
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.as_tensor(
            [item[2] for item in batch],
            dtype=torch.float32,
            device=self.device,
        )
        next_observations = torch.as_tensor(
            np.stack([item[3] for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        next_masks = torch.as_tensor(
            np.stack([item[4] for item in batch]),
            dtype=torch.bool,
            device=self.device,
        )
        dones = torch.as_tensor(
            [item[5] for item in batch],
            dtype=torch.float32,
            device=self.device,
        )

        values = self.model(observations).gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)
        with torch.no_grad():
            online_next = self.model(next_observations).masked_fill(
                ~next_masks,
                -torch.inf,
            )
            next_actions = online_next.argmax(dim=1, keepdim=True)
            next_values = self.target(next_observations).gather(
                1,
                next_actions,
            ).squeeze(1)
            targets = rewards + 0.98 * (1.0 - dones) * next_values

        loss = F.smooth_l1_loss(values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        if self.steps % 250 == 0:
            self.target.load_state_dict(self.model.state_dict())


def create_agent(config):
    return TeacherDQNAgent(config)
