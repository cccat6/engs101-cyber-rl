"""A minimal trainable agent. Students may replace this implementation.

The submitted model may contain at most 1,000,000 registered parameters per
role. The local runner checks the same limit as the course server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from cyberrl_eval import AgentConfig, Observation, Transition


class SimpleAgent:
    """One-layer, one-step policy-gradient example."""

    def __init__(self, config: AgentConfig) -> None:
        self.device = torch.device(config.device)
        self.model = nn.Linear(
            config.observation_dim,
            config.num_actions,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self._log_probability = None

    def reset(self, seed: int) -> None:
        torch.manual_seed(seed)

    def act(self, observation: Observation, training: bool) -> int:
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
        logits = self.model(features).masked_fill(~mask, -torch.inf)

        if not training:
            return int(logits.argmax().item())

        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        self._log_probability = distribution.log_prob(action)
        return int(action.item())

    def learn(self, transition: Transition) -> None:
        reward = torch.tensor(
            transition.reward,
            dtype=torch.float32,
            device=self.device,
        )
        loss = -self._log_probability * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._log_probability = None


def create_agent(config: AgentConfig) -> SimpleAgent:
    return SimpleAgent(config)
