"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], action_dim * chunk_size))
        
        self.mlp = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        return nn.MSELoss()(self.mlp(state).view_as(action_chunk), action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,            # num_steps is not used in MSEPolicy
    ) -> torch.Tensor:
        return self.mlp(state).view(-1, self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
        time_dim: int = 16,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
    
        input_dim = state_dim + action_dim * chunk_size + time_dim     
        self.time_embedding = nn.Sequential(            # time embedding is needed for better performance
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.GELU())    
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dims[-1], action_dim * chunk_size))
        
        self.mlp = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]        
        t = torch.rand(batch_size, device=state.device)     # uniform [0,1)
        
        noise = torch.randn_like(action_chunk)        
        t_broadcast = t.view(batch_size, 1, 1)
        noisy_action = (1 - t_broadcast) * noise + t_broadcast * action_chunk
        
        t_emb = self.time_embedding(t.unsqueeze(1))
        input_state = torch.cat([state, noisy_action.view(batch_size, -1), t_emb], dim=1)
        
        pred_velocity = self.mlp(input_state).view_as(action_chunk)        
        target_velocity = action_chunk - noise
        
        return nn.MSELoss()(pred_velocity, target_velocity)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        dt = 1.0 / num_steps
        
        action = torch.randn(
            batch_size, 
            self.chunk_size * self.action_dim, 
            device=state.device
        )
        
        for i in range(num_steps):
            t = (i / num_steps) * torch.ones(batch_size, 1, device=state.device)
            t_emb = self.time_embedding(t)
            network_input = torch.cat([state, action, t_emb], dim=1)
            velocity = self.mlp(network_input)
            action = action + dt * velocity
        return action.view(-1, self.chunk_size, self.action_dim)


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
