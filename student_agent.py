"""Edit this file to implement your agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from cyberrl_eval import AgentConfig, Observation, Transition


class StarterAgent:
    """Runnable example that does not learn."""

    def __init__(self, config: AgentConfig) -> None:
        self.device = torch.device(config.device)
        self.model = nn.Sequential(
            nn.Linear(config.observation_dim, 32),
            nn.Tanh(),
            nn.Linear(32, config.num_actions),
        ).to(self.device)
        self.rng = np.random.default_rng(config.seed)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, observation: Observation, training: bool) -> int:
        valid = np.flatnonzero(observation.action_mask)
        if training:
            return int(self.rng.choice(valid))

        with torch.no_grad():
            features = torch.as_tensor(
                observation.features,
                dtype=torch.float32,
                device=self.device,
            )
            values = self.model(features)
            mask = torch.as_tensor(
                observation.action_mask,
                dtype=torch.bool,
                device=self.device,
            )
            return int(values.masked_fill(~mask, -torch.inf).argmax().item())

    def learn(self, transition: Transition) -> None:
        # Replace this with your own PyTorch update.
        del transition


def create_agent(config: AgentConfig) -> StarterAgent:
    return StarterAgent(config)
