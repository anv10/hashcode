import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from enum import Enum
from dataclasses import dataclass
import matplotlib.pyplot as plt

# -----------------------------
# Action and State Definitions
# -----------------------------
class RemediationAction(Enum):
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
    service_metrics: dict
    gnn_predictions: np.ndarray
    causal_scores: dict
    dependency_health: np.ndarray
    timestamp: float

# -----------------------------
# Observability Environment
# -----------------------------
class ObservabilityEnv(gym.Env):
    def __init__(self, num_services=5, max_steps=50, gnn_predictor=None):
        super().__init__()
        self.num_services = num_services
        self.max_steps = max_steps
        self.gnn_predictor = gnn_predictor
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(len(RemediationAction)*num_services)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_services*6,), dtype=np.float32
        )
        self.state = None
        self.history = {
            "latency": [], "error_rate": [], "health": [], "reward": [], "actions": []
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = self._generate_initial_state()
        return self._get_observation(), {}

    def step(self, action: int):
        self.current_step += 1
        service_idx = action // len(RemediationAction)
        action_type = RemediationAction(action % len(RemediationAction))

        # Execute action
        self._execute_action(service_idx, action_type)
        reward = self._calculate_reward(service_idx, action_type)

        obs = self._get_observation()
        done = self._check_termination()
        truncated = self.current_step >= self.max_steps

        # Log step info
        self.history["latency"].append(
            np.mean([m['latency'] for m in self.state.service_metrics.values()])
        )
        self.history["error_rate"].append(
            np.mean([m['error_rate'] for m in self.state.service_metrics.values()])
        )
        self.history["health"].append(
            np.mean([m['health'] for m in self.state.service_metrics.values()])
        )
        self.history["reward"].append(reward)
        self.history["actions"].append(action_type.name)

        print(f"Step {self.current_step}: Service_{service_idx}, Action={action_type.name}, "
              f"Latency={self.history['latency'][-1]:.2f}, "
              f"Error={self.history['error_rate'][-1]:.3f}, "
              f"Health={self.history['health'][-1]:.2f}, Reward={reward:.2f}")

        return obs, reward, done, truncated, {}

    def _generate_initial_state(self):
        service_metrics = {}
        for i in range(self.num_services):
            service_metrics[f"service_{i}"] = {
                'cpu': np.random.uniform(0.3, 0.8),
                'memory': np.random.uniform(0.4, 0.8),
                'latency': np.random.uniform(50, 500),
                'error_rate': np.random.uniform(0, 0.1),
                'request_rate': np.random.uniform(10, 100),
                'health': np.random.uniform(0.5, 1.0)
            }
        gnn_predictions = np.random.uniform(0, 1, self.num_services)
        causal_scores = {f"service_{i}": np.random.uniform(0, 1) for i in range(self.num_services)}
        dependency_health = np.random.uniform(0.5, 1.0, self.num_services)
        return SystemState(service_metrics, gnn_predictions, causal_scores, dependency_health, 0)

    def _get_observation(self):
        obs = []
        for i in range(self.num_services):
            metrics = self.state.service_metrics[f"service_{i}"]
            obs.extend([
                metrics['cpu'],
                metrics['memory'],
                metrics['latency'] / 1000.0,
                metrics['error_rate'],
                self.state.gnn_predictions[i],
                self.state.causal_scores[f"service_{i}"]
            ])
        return np.array(obs, dtype=np.float32)

    def _execute_action(self, service_idx, action: RemediationAction):
        metrics = self.state.service_metrics[f"service_{service_idx}"]
        if action == RemediationAction.RESTART_POD:
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
            metrics['latency'] *= 1.1
        elif action == RemediationAction.INCREASE_MEMORY:
            metrics['memory'] *= 0.8
        elif action == RemediationAction.INCREASE_CPU:
            metrics['cpu'] *= 0.7
        elif action == RemediationAction.ROLLBACK_DEPLOYMENT:
            metrics['health'] = 0.9
            metrics['error_rate'] = 0.01

        # GNN prediction decay simulation
        self.state.gnn_predictions[service_idx] *= 0.8

    def _calculate_reward(self, service_idx, action: RemediationAction):
        metrics = self.state.service_metrics[f"service_{service_idx}"]
        reward = 0.0
        if self.state.gnn_predictions[service_idx] < 0.3:
            reward += 10
        reward += metrics['health']*5
        reward += (1 - metrics['error_rate'])*3
        if metrics['cpu']>0.9 or metrics['memory']>0.9:
            reward -= 5
        if action != RemediationAction.NO_ACTION:
            reward -= 1
        reward += self.state.causal_scores[f"service_{service_idx}"]*2
        if metrics['latency']>300:
            reward -= 2
        return reward

    def _check_termination(self):
        all_healthy = all(
            m['health']>0.8 and self.state.gnn_predictions[i]<0.3
            for i,m in enumerate(self.state.service_metrics.values())
        )
        return all_healthy

    def plot_results(self):
        steps = range(1, len(self.history['reward'])+1)
        plt.figure(figsize=(12,6))
        plt.subplot(2,2,1)
        plt.plot(steps, self.history['latency'], label='Latency')
        plt.xlabel('Step'); plt.ylabel('Latency (ms)'); plt.title('Latency over Time'); plt.grid(True)
        plt.subplot(2,2,2)
        plt.plot(steps, self.history['error_rate'], label='Error Rate', color='r')
        plt.xlabel('Step'); plt.ylabel('Error Rate'); plt.title('Error Rate over Time'); plt.grid(True)
        plt.subplot(2,2,3)
        plt.plot(steps, self.history['health'], label='Health', color='g')
        plt.xlabel('Step'); plt.ylabel('Health'); plt.title('Service Health over Time'); plt.grid(True)
        plt.subplot(2,2,4)
        plt.plot(steps, self.history['reward'], label='Reward', color='m')
        plt.xlabel('Step'); plt.ylabel('Reward'); plt.title('Reward over Time'); plt.grid(True)
        plt.tight_layout()
        plt.show()

# -----------------------------
# Main: Train and Demo Agent
# -----------------------------
if __name__ == "__main__":
    env = ObservabilityEnv(num_services=5, max_steps=50)
    env = DummyVecEnv([lambda: env])
    
    agent = PPO("MlpPolicy", env, verbose=1)
    
    print("Training agent...")
    agent.learn(total_timesteps=5000)
    
    print("\nTesting agent...")
    obs = env.reset()
    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
    
    # Plot metrics & reward
    env.envs[0].plot_results()
