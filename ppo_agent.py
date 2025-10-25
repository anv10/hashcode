import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    Actor: outputs action probabilities π(a|s)
    Critic: outputs state value V(s)
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        features = self.shared(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value
    
    def act(self, state):
        """Sample action from policy"""
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob, state_value


class PPOAgent:
    """
    Proximal Policy Optimization agent with clipped objective.
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        K_epochs: int = 10,
        device: str = 'cpu'
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        
        # Initialize networks
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # Memory buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
        self.state_values = []
    
    def select_action(self, state: np.ndarray):
        """Select action using current policy"""
        state = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            action, log_prob, state_value = self.policy_old.act(state)
        
        # Store for training
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.state_values.append(state_value)
        
        return action
    
    def store_reward(self, reward: float, is_terminal: bool):
        """Store reward and terminal flag"""
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
    
    def update(self):
        """PPO policy update with clipped objective"""
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Convert lists to tensors
        old_states = torch.stack(self.states).detach()
        old_actions = torch.tensor(self.actions).to(self.device)
        old_log_probs = torch.stack(self.log_probs).detach()
        old_state_values = torch.stack(self.state_values).squeeze().detach()
        
        # Calculate advantages: A(s,a) = Q(s,a) - V(s)
        advantages = rewards - old_state_values
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Get current policy predictions
            action_probs, state_values = self.policy(old_states)
            dist = Categorical(action_probs)
            log_probs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            # Calculate ratios: π(a|s) / π_old(a|s)
            ratios = torch.exp(log_probs - old_log_probs.detach())
            
            # Clipped surrogate objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Final loss: policy loss + value loss - entropy bonus
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.loss_fn(state_values.squeeze(), rewards)
            entropy_loss = -dist_entropy.mean()
            
            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear memory
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.state_values.clear()
    
    def save(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")
