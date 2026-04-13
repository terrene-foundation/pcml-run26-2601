# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for Exercise 8 — Reinforcement Learning.

Contains: CartPole setup, reward plotting helpers, ExperimentTracker/ModelRegistry
setup, custom environment base class, evaluation utilities.
Technique-specific code does NOT belong here.
"""
from __future__ import annotations

import asyncio
import pickle
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym
from gymnasium import spaces

import polars as pl

from kailash.db import ConnectionManager
from kailash_ml import ModelVisualizer
from kailash_ml.engines.experiment_tracker import ExperimentTracker
from kailash_ml.engines.model_registry import ModelRegistry

from shared.kailash_helpers import get_device, setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = get_device()

# Output directory for all visualisation artifacts
OUTPUT_DIR = Path("outputs") / "ex8_reinforcement_learning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# CARTPOLE ENVIRONMENT
# ════════════════════════════════════════════════════════════════════════


def make_cartpole() -> tuple[gym.Env, int, int]:
    """Create CartPole-v1 and return (env, obs_dim, n_actions)."""
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print(f"CartPole-v1  obs_dim={obs_dim}  n_actions={n_actions}")
    return env, obs_dim, n_actions


# ════════════════════════════════════════════════════════════════════════
# KAILASH ENGINE SETUP
# ════════════════════════════════════════════════════════════════════════


async def _setup_engines():
    conn = ConnectionManager("sqlite:///mlfp05_rl.db")
    await conn.initialize()

    tracker = ExperimentTracker(conn)
    exp_name = await tracker.create_experiment(
        name="m5_reinforcement_learning",
        description="RL algorithms: DQN and PPO on CartPole and business envs",
    )

    try:
        registry = ModelRegistry(conn)
        has_registry = True
    except Exception as e:
        registry = None
        has_registry = False
        print(f"  Note: ModelRegistry setup skipped ({e})")

    return conn, tracker, exp_name, registry, has_registry


def setup_engines() -> tuple:
    """Synchronously set up kailash-ml engines."""
    return asyncio.run(_setup_engines())


# ════════════════════════════════════════════════════════════════════════
# REPLAY BUFFER — shared by DQN and custom env training
# ════════════════════════════════════════════════════════════════════════


class ReplayBuffer:
    """Fixed-size buffer storing (state, action, reward, next_state, done)."""

    def __init__(self, capacity: int = 10_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(list(self.buffer), batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.long, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=device),
            torch.tensor(dones, dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)


# ════════════════════════════════════════════════════════════════════════
# DQN NETWORK — shared by DQN training and custom env training
# ════════════════════════════════════════════════════════════════════════


class DQN(nn.Module):
    """Deep Q-Network: maps state -> Q-value for each action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ════════════════════════════════════════════════════════════════════════
# EVALUATION UTILITY
# ════════════════════════════════════════════════════════════════════════


def evaluate_policy(env: gym.Env, policy_fn, n_episodes: int = 30) -> list[float]:
    """Evaluate a policy function over n_episodes. Returns list of total rewards."""
    eval_returns: list[float] = []
    for i in range(n_episodes):
        state, _ = env.reset(seed=1000 + i)
        total = 0.0
        done = False
        while not done:
            action = policy_fn(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        eval_returns.append(total)
    return eval_returns


# ════════════════════════════════════════════════════════════════════════
# REWARD PLOTTING HELPERS
# ════════════════════════════════════════════════════════════════════════


def moving_average(xs: list[float], window: int = 10) -> list[float]:
    """Smooth a time series with a rolling mean."""
    if len(xs) < window:
        return xs
    arr = np.asarray(xs, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    return list(np.convolve(arr, kernel, mode="valid"))


def plot_reward_curve(
    viz: ModelVisualizer,
    rewards: list[float],
    title: str,
    filename: str,
    window: int = 20,
    x_label: str = "Episode",
    y_label: str = "Reward",
) -> None:
    """Plot a reward curve with moving average and save to HTML."""
    metrics = {
        f"{title} reward": rewards,
        f"{title} moving avg ({window})": moving_average(rewards, window),
    }
    fig = viz.training_history(metrics=metrics, x_label=x_label, y_label=y_label)
    out_path = OUTPUT_DIR / filename
    fig.write_html(str(out_path))
    print(f"  Saved: {out_path}")


# ════════════════════════════════════════════════════════════════════════
# MODEL REGISTRATION HELPER
# ════════════════════════════════════════════════════════════════════════


async def _register_rl_model(
    registry: ModelRegistry,
    name: str,
    model: nn.Module,
    metrics_dict: dict[str, float],
):
    """Register a single RL policy network in ModelRegistry."""
    from kailash_ml.types import MetricSpec

    model_bytes = pickle.dumps(model.state_dict())
    metric_specs = [MetricSpec(name=k, value=v) for k, v in metrics_dict.items()]
    version = await registry.register_model(
        name=name,
        artifact=model_bytes,
        metrics=metric_specs,
    )
    print(f"  Registered {name}: version={version.version}")
    return version


def register_rl_model(
    registry: ModelRegistry,
    name: str,
    model: nn.Module,
    metrics_dict: dict[str, float],
):
    """Sync wrapper for RL model registration."""
    return asyncio.run(_register_rl_model(registry, name, model, metrics_dict))
