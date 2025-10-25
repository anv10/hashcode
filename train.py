"""
rl_controller/train.py - Complete training script for RL agent
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from rl_controller.rl_env import K8sRemediationEnv
from rl_controller.ppo_agent import PPOAgent


def train_rl_agent(
    episodes: int = 5000,
    update_timestep: int = 200,
    save_interval: int = 500,
    log_interval: int = 50,
    checkpoint_dir: str = "checkpoints",
    render: bool = False
):
    """
    Train PPO agent on K8s remediation environment.
    
    Args:
        episodes: Number of training episodes
        update_timestep: Update policy every N timesteps
        save_interval: Save checkpoint every N episodes
        log_interval: Print stats every N episodes
        checkpoint_dir: Directory to save checkpoints
        render: Whether to render environment
    """
    print("=" * 70)
    print("AION RL Controller Training")
    print("=" * 70)
    print(f"Episodes: {episodes}")
    print(f"Update interval: {update_timestep} timesteps")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print("=" * 70)
    print()
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Initialize environment safely
    env = K8sRemediationEnv(num_services=5)
    
    # If rendering is requested, define a safe render method
    if render:
        if not hasattr(env, 'render'):
            def render_fn():
                print("Rendering environment (dummy)")
            env.render = render_fn
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        K_epochs=10,
        device='cpu'
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    avg_rewards = []
    success_count = 0
    timestep = 0
    
    # Action distribution tracking
    action_counts = {i: 0 for i in range(action_dim)}
    
    # Training loop
    print("🚀 Starting training...\n")
    
    for episode in tqdm(range(1, episodes + 1), desc="Training"):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state)
            action_counts[action] += 1
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store reward
            agent.store_reward(reward, done)
            
            episode_reward += reward
            episode_length += 1
            timestep += 1
            state = next_state
            
            # Optional rendering
            if render:
                env.render()
            
            # Update policy
            if timestep % update_timestep == 0:
                agent.update()
        
        # Track success
        if info.get('health', 0) > 0.85:
            success_count += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Log progress
        if episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_length = np.mean(episode_lengths[-log_interval:])
            success_rate = (success_count / log_interval) * 100
            avg_rewards.append(avg_reward)
            
            print(f"\nEpisode {episode:5d} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Success Rate: {success_rate:.1f}% | "
                  f"Health: {info.get('health', 0):.3f}")
            
            # Reset success counter
            success_count = 0
        
        # Save checkpoint
        if episode % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"ppo_agent_ep{episode}.pth")
            agent.save(checkpoint_path)
            
            # Save action distribution
            print(f"\nAction Distribution (Episode {episode}):")
            for action_id, count in action_counts.items():
                action_name = env.actions[action_id]
                pct = (count / sum(action_counts.values())) * 100
                print(f"  {action_name:12s}: {count:5d} ({pct:5.1f}%)")
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, "ppo_agent_final.pth")
    agent.save(final_path)
    
    # Plot training curves
    plot_training_curves(episode_rewards, avg_rewards, log_interval, action_counts, env.actions)
    
    # Save training log
    save_training_log(episode_rewards, episode_lengths, checkpoint_dir)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Final Average Reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Best Episode Reward: {max(episode_rewards):.2f}")
    print(f"Total Timesteps: {timestep}")
    print(f"Model saved to: {final_path}")
    print("=" * 70)
    
    env.close()


def plot_training_curves(episode_rewards, avg_rewards, log_interval, action_counts, actions):
    """Plot and save training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.3, label='Episode Reward')
    if avg_rewards:
        x_avg = range(log_interval - 1, len(episode_rewards), log_interval)
        axes[0, 0].plot(x_avg, avg_rewards, linewidth=2, label=f'Avg ({log_interval} eps)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Moving average
    window = 100
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(moving_avg)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].set_title(f'{window}-Episode Moving Average')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Action distribution
    action_names = [actions[i] for i in sorted(action_counts.keys())]
    counts = [action_counts[i] for i in sorted(action_counts.keys())]
    axes[1, 0].bar(action_names, counts)
    axes[1, 0].set_xlabel('Action')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Action Distribution')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Cumulative reward
    cumulative = np.cumsum(episode_rewards)
    axes[1, 1].plot(cumulative)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Cumulative Reward')
    axes[1, 1].set_title('Cumulative Reward Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"logs/training_curves_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Training curves saved to: {plot_path}")
    
    plt.close()


def save_training_log(episode_rewards, episode_lengths, checkpoint_dir):
    """Save training statistics to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(checkpoint_dir, f"training_log_{timestamp}.txt")
    
    with open(log_path, 'w') as f:
        f.write("AION RL Controller Training Log\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Total Episodes: {len(episode_rewards)}\n")
        f.write(f"Total Timesteps: {sum(episode_lengths)}\n")
        f.write("\n")
        
        f.write("Statistics:\n")
        f.write(f"  Mean Reward: {np.mean(episode_rewards):.2f}\n")
        f.write(f"  Std Reward: {np.std(episode_rewards):.2f}\n")
        f.write(f"  Max Reward: {np.max(episode_rewards):.2f}\n")
        f.write(f"  Min Reward: {np.min(episode_rewards):.2f}\n")
        f.write(f"  Mean Episode Length: {np.mean(episode_lengths):.2f}\n")
        f.write("\n")
        
        f.write("Last 100 Episodes:\n")
        f.write(f"  Mean Reward: {np.mean(episode_rewards[-100:]):.2f}\n")
        f.write(f"  Std Reward: {np.std(episode_rewards[-100:]):.2f}\n")
        f.write("=" * 70 + "\n")
    
    print(f"📝 Training log saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AION RL Controller")
    parser.add_argument('--episodes', type=int, default=5000, help='Number of training episodes')
    parser.add_argument('--update-timestep', type=int, default=200, help='Update policy every N timesteps')
    parser.add_argument('--save-interval', type=int, default=500, help='Save checkpoint every N episodes')
    parser.add_argument('--log-interval', type=int, default=50, help='Log stats every N episodes')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--render', action='store_true', help='Render environment')
    
    args = parser.parse_args()
    
    train_rl_agent(
        episodes=args.episodes,
        update_timestep=args.update_timestep,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir,
        render=args.render
    )
