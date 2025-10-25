"""
AION RL Controller - Autonomous Remediation Agent
Integrates with GNN predictions and RCAEval dataset for intelligent fault remediation
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Dict, List, Tuple, Optional
import pandas as pd
import json
from dataclasses import dataclass
from enum import Enum


class RemediationAction(Enum):
    """Available remediation actions for the RL agent"""
    NO_ACTION = 0
    RESTART_POD = 1
    SCALE_UP = 2
    SCALE_DOWN = 3
    CLEAR_CACHE = 4
    INCREASE_MEMORY = 5
    INCREASE_CPU = 6
    ROLLBACK_DEPLOYMENT = 7


@dataclass
class SystemState:
    """Represents the current state of the microservice system"""
    service_metrics: Dict[str, float]  # CPU, memory, latency, error_rate per service
    gnn_predictions: np.ndarray  # Anomaly predictions from GNN
    causal_scores: Dict[str, float]  # Root cause probabilities
    dependency_health: np.ndarray  # Health of dependent services
    timestamp: float


class ObservabilityEnv(gym.Env):
    """
    Custom Gym environment for observability-driven remediation
    Integrates with RCAEval dataset and GNN predictions
    """
    
    def __init__(self, 
                 num_services: int = 10,
                 max_steps: int = 100,
                 gnn_predictor=None,
                 human_in_loop: bool = True):
        super().__init__()
        
        self.num_services = num_services
        self.max_steps = max_steps
        self.gnn_predictor = gnn_predictor
        self.human_in_loop = human_in_loop
        self.current_step = 0
        
        # Action space: one action per service
        self.action_space = gym.spaces.Discrete(len(RemediationAction) * num_services)
        
        # Observation space: [service_metrics, gnn_predictions, causal_scores, health_status]
        obs_dim = num_services * 6  # 6 features per service
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # System state
        self.state: Optional[SystemState] = None
        self.baseline_metrics = {}
        self.action_history = []
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        self.current_step = 0
        self.action_history = []
        
        # Initialize with baseline metrics from RCAEval dataset
        self.state = self._generate_initial_state()
        obs = self._get_observation()
        
        return obs, {}
    
    def step(self, action: int):
        """Execute action and return new state, reward, done, info"""
        self.current_step += 1
        
        # Decode action
        service_idx = action // len(RemediationAction)
        action_type = RemediationAction(action % len(RemediationAction))
        
        # Human-in-the-loop approval (simulated)
        if self.human_in_loop and self._requires_approval(action_type):
            approved = self._request_approval(service_idx, action_type)
            if not approved:
                action_type = RemediationAction.NO_ACTION
        
        # Execute action and get new state
        self._execute_action(service_idx, action_type)
        self.action_history.append((service_idx, action_type))
        
        # Calculate reward
        reward = self._calculate_reward(service_idx, action_type)
        
        # Check if episode is done
        done = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, done, truncated, info
    
    def _generate_initial_state(self) -> SystemState:
        """Generate initial system state with potential anomalies"""
        # Simulate service metrics
        service_metrics = {}
        for i in range(self.num_services):
            service_metrics[f"service_{i}"] = {
                'cpu': np.random.uniform(0.3, 0.9),
                'memory': np.random.uniform(0.4, 0.8),
                'latency': np.random.uniform(50, 500),
                'error_rate': np.random.uniform(0, 0.1),
                'request_rate': np.random.uniform(10, 100),
                'health': np.random.uniform(0.5, 1.0)
            }
        
        # GNN predictions (simulate or use actual GNN)
        gnn_predictions = np.random.uniform(0, 1, self.num_services)
        
        # Causal scores (which service is likely root cause)
        causal_scores = {f"service_{i}": np.random.uniform(0, 1) 
                        for i in range(self.num_services)}
        
        # Dependency health
        dependency_health = np.random.uniform(0.5, 1.0, self.num_services)
        
        return SystemState(
            service_metrics=service_metrics,
            gnn_predictions=gnn_predictions,
            causal_scores=causal_scores,
            dependency_health=dependency_health,
            timestamp=0
        )
    
    def _get_observation(self) -> np.ndarray:
        """Convert system state to observation vector"""
        obs = []
        for i in range(self.num_services):
            service = f"service_{i}"
            metrics = self.state.service_metrics[service]
            obs.extend([
                metrics['cpu'],
                metrics['memory'],
                metrics['latency'] / 1000.0,  # Normalize
                metrics['error_rate'],
                self.state.gnn_predictions[i],
                self.state.causal_scores[service]
            ])
        return np.array(obs, dtype=np.float32)
    
    def _execute_action(self, service_idx: int, action: RemediationAction):
        """Simulate action execution and update system state"""
        service = f"service_{service_idx}"
        metrics = self.state.service_metrics[service]
        
        if action == RemediationAction.RESTART_POD:
            # Temporary latency spike, then improvement
            metrics['latency'] *= 1.5
            metrics['error_rate'] = max(0, metrics['error_rate'] - 0.05)
            metrics['health'] = min(1.0, metrics['health'] + 0.2)
            
        elif action == RemediationAction.SCALE_UP:
            metrics['cpu'] *= 0.7
            metrics['latency'] *= 0.8
            
        elif action == RemediationAction.SCALE_DOWN:
            metrics['cpu'] *= 1.2
            metrics['latency'] *= 1.1
            
        elif action == RemediationAction.CLEAR_CACHE:
            metrics['memory'] *= 0.6
            metrics['latency'] *= 1.1  # Temporary increase
            
        elif action == RemediationAction.INCREASE_MEMORY:
            metrics['memory'] *= 0.8
            
        elif action == RemediationAction.INCREASE_CPU:
            metrics['cpu'] *= 0.7
            
        elif action == RemediationAction.ROLLBACK_DEPLOYMENT:
            # Restore to baseline
            metrics['health'] = 0.9
            metrics['error_rate'] = 0.01
        
        # Update GNN predictions based on new state
        if self.gnn_predictor:
            # Would call actual GNN here
            pass
        else:
            self.state.gnn_predictions[service_idx] *= 0.8
    
    def _calculate_reward(self, service_idx: int, action: RemediationAction) -> float:
        """
        Reward shaping for safe and effective remediation
        Positive reward for: reducing anomalies, improving health, minimal actions
        Negative reward for: incorrect actions, excessive interventions
        """
        service = f"service_{service_idx}"
        metrics = self.state.service_metrics[service]
        
        reward = 0.0
        
        # Reward for improving anomaly scores
        if self.state.gnn_predictions[service_idx] < 0.3:
            reward += 10.0
        
        # Reward for high service health
        reward += metrics['health'] * 5.0
        
        # Reward for low error rates
        reward += (1.0 - metrics['error_rate']) * 3.0
        
        # Penalty for high resource usage
        if metrics['cpu'] > 0.9 or metrics['memory'] > 0.9:
            reward -= 5.0
        
        # Penalty for excessive actions
        if action != RemediationAction.NO_ACTION:
            reward -= 1.0
        
        # Bonus for targeting high causal score services
        reward += self.state.causal_scores[service] * 2.0
        
        # Penalty for high latency
        if metrics['latency'] > 300:
            reward -= 2.0
        
        return reward
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate (all services healthy)"""
        all_healthy = all(
            metrics['health'] > 0.8 and self.state.gnn_predictions[i] < 0.3
            for i, (_, metrics) in enumerate(self.state.service_metrics.items())
        )
        return all_healthy
    
    def _requires_approval(self, action: RemediationAction) -> bool:
        """Check if action requires human approval"""
        high_risk_actions = [
            RemediationAction.ROLLBACK_DEPLOYMENT,
            RemediationAction.SCALE_DOWN
        ]
        return action in high_risk_actions
    
    def _request_approval(self, service_idx: int, action: RemediationAction) -> bool:
        """Simulate human approval (would be actual UI in production)"""
        print(f"[APPROVAL REQUIRED] Action: {action.name} on service_{service_idx}")
        # In production, this would wait for human input
        return np.random.random() > 0.2  # 80% approval rate for simulation
    
    def _get_info(self) -> Dict:
        """Return additional info about current state"""
        return {
            'step': self.current_step,
            'action_history': self.action_history,
            'anomaly_count': np.sum(self.state.gnn_predictions > 0.5),
            'avg_health': np.mean([m['health'] for m in self.state.service_metrics.values()])
        }


class HumanFeedbackCallback(BaseCallback):
    """Callback to handle human feedback during training"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Log episode statistics
        if len(self.locals['rewards']) > 0:
            self.episode_rewards.append(np.mean(self.locals['rewards']))
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout"""
        if self.verbose > 0:
            print(f"Rollout ended. Mean reward: {np.mean(self.episode_rewards[-100:]):.2f}")


def create_rl_agent(
    num_services: int = 10,
    human_in_loop: bool = True,
    training: bool = True
) -> Tuple[PPO, ObservabilityEnv]:
    """
    Create and configure the RL agent
    
    Args:
        num_services: Number of microservices to monitor
        human_in_loop: Enable human approval for high-risk actions
        training: Whether agent is in training mode
    
    Returns:
        Tuple of (PPO agent, environment)
    """
    # Create environment
    env = ObservabilityEnv(
        num_services=num_services,
        human_in_loop=human_in_loop and not training
    )
    env = DummyVecEnv([lambda: env])
    
    # Create PPO agent
    agent = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        verbose=1,
        tensorboard_log="./rl_logs/"
    )
    
    return agent, env


if __name__ == "__main__":
    # Example usage
    print("Initializing AION RL Controller...")
    
    # Create agent
    agent, env = create_rl_agent(
        num_services=10,
        human_in_loop=True,
        training=True
    )
    
    # Train agent
    print("\nTraining RL agent...")
    callback = HumanFeedbackCallback(verbose=1)
    agent.learn(
        total_timesteps=50000,
        callback=callback,
        progress_bar=True
    )
    
    # Save model
    agent.save("aion_rl_controller")
    print("\nModel saved as 'aion_rl_controller.zip'")
    
    # Test agent
    print("\nTesting trained agent...")
    obs = env.reset()
    for i in range(20):
        action, _states = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(f"Step {i}: Action={action}, Reward={reward:.2f}")
        if done:
            print("Episode finished - all services healthy!")
            break