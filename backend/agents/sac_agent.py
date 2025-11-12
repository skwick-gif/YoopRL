"""
sac_agent.py
SAC Agent Wrapper for Leveraged ETF Trading

Purpose:
- Wrapper around Stable-Baselines3 SAC implementation
- Optimized for leveraged ETF trading with continuous actions
- MLP policy with 2 hidden layers [256, 256]
- Integrates with ETFTradingEnv from Phase 2

Why SAC for leveraged ETFs:
- Off-policy algorithm: sample-efficient with replay buffer
- Continuous action space: precise position sizing [-1, 1]
- Entropy regularization: encourages exploration
- High returns: aggressive trading suitable for leveraged ETFs
- Maximum entropy: balances reward and policy diversity

Hyperparameters (from TrainingConfig):
- learning_rate: 0.0001-0.001 (default: 0.0003)
- buffer_size: replay buffer size (default: 1,000,000)
- batch_size: 64-512 (default: 256)
- tau: soft update coefficient (default: 0.005)
- gamma: discount factor 0.95-0.999 (default: 0.99)
- ent_coef: entropy coefficient (default: 'auto')

Wiring:
- Receives environment from Phase 2 (ETFTradingEnv)
- Receives hyperparameters from TrainingConfig
- Saves models via ModelManager
- Used by Training Loop in Phase 4
"""

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import os
from datetime import datetime
from typing import Optional, Dict, Any, Callable, Union
import numpy as np
import torch


class SACAgent:
    """
    SAC Agent for Leveraged ETF Trading
    
    Wraps Stable-Baselines3 SAC with ETF-specific configurations.
    
    Attributes:
        env: Trading environment (ETFTradingEnv)
        model: SAC model instance
        hyperparameters: Dict of hyperparameters
        model_dir: Directory for saving models
    """
    
    def __init__(
        self, 
        env, 
        hyperparameters: Dict[str, Any],
        model_dir: str = 'backend/models/sac',
        *,
        seed: Optional[int] = None
    ):
        """
        Initialize SAC Agent
        
        Args:
            env: Gym environment (ETFTradingEnv)
            hyperparameters: Dict containing:
                - learning_rate: float
                - buffer_size: int
                - batch_size: int
                - tau: float
                - gamma: float
                - ent_coef: str or float ('auto' or value)
                - target_entropy: str or float ('auto' or value)
            model_dir: Path to save models
            seed: Optional random seed for deterministic initialization
        """
        def _make_env():
            if seed is not None:
                if hasattr(env, 'reset'):
                    try:
                        env.reset(seed=seed)
                    except TypeError:
                        env.reset()  # type: ignore[call-arg]
                action_space = getattr(env, 'action_space', None)
                if action_space is not None and hasattr(action_space, 'seed'):
                    action_space.seed(seed)
                observation_space = getattr(env, 'observation_space', None)
                if observation_space is not None and hasattr(observation_space, 'seed'):
                    observation_space.seed(seed)
            return env

        # Wrap environment for vectorization (required by SB3)
        self.env = DummyVecEnv([_make_env])
        self.hyperparameters = hyperparameters
        self.model_dir = model_dir
        self.seed = seed
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Extract hyperparameters with defaults
        learning_rate = hyperparameters.get('learning_rate', 0.0003)
        buffer_size = hyperparameters.get('buffer_size', 1_000_000)
        batch_size = hyperparameters.get('batch_size', 256)
        tau = hyperparameters.get('tau', 0.005)
        gamma = hyperparameters.get('gamma', 0.99)
        ent_coef = hyperparameters.get('ent_coef', 'auto')
        target_entropy = hyperparameters.get('target_entropy', 'auto')
        train_freq = hyperparameters.get('train_freq', 1)
        gradient_steps = hyperparameters.get('gradient_steps', 1)
        learning_starts = hyperparameters.get('learning_starts', 1000)
        target_update_interval = hyperparameters.get('target_update_interval', 1)
        policy_kwargs = hyperparameters.get('policy_kwargs', {'net_arch': [256, 256]})
        
        # Auto-detect CUDA/CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create SAC model
        self.model = SAC(
            policy='MlpPolicy',
            env=self.env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            learning_starts=learning_starts,
            target_update_interval=target_update_interval,
            policy_kwargs=policy_kwargs,
            device=device,  # GPU support
            verbose=1,
            tensorboard_log=f"{model_dir}/tensorboard/",
            seed=seed,
        )

        if seed is not None:
            try:
                self.env.seed(seed)
            except AttributeError:
                pass
        
        print(f"✅ SAC Agent initialized")
        print(f"   Device: {device.upper()}")
        if device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Learning Rate: {learning_rate}")
        print(f"   Buffer Size: {buffer_size:,}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Gamma: {gamma}")
        print(f"   Entropy Coef: {ent_coef}")
        print(f"   Network: {policy_kwargs.get('net_arch', '[default]')}")
    
    def train(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        progress_callback: Optional[Callable] = None,
        *,
        use_progress_bar: bool = False,
    ) -> None:
        """
        Train the SAC agent
        
        Args:
            total_timesteps: Number of timesteps to train
            callback: SB3 callback for logging/checkpointing
            progress_callback: Custom callback for progress updates
        """
        print(f"\n{'='*60}")
        print(f"🏋️ Training SAC Agent...")
        print(f"   Total Timesteps: {total_timesteps:,}")
        print(f"{'='*60}\n")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=use_progress_bar,
        )
        
        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"{'='*60}\n")
    
    def predict(
        self, 
        observation: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Predict action for given observation
        
        Args:
            observation: State vector from environment
            deterministic: If True, use deterministic policy (no exploration)
        
        Returns:
            action: Continuous action in [-1, 1]
                   -1 = Full short, 0 = No position, +1 = Full long
        """
        action, _states = self.model.predict(
            observation, 
            deterministic=deterministic
        )
        return action
    
    def save(
        self, 
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save model with versioning
        
        Args:
            version: Version string (e.g., '1.0', '1.1')
                    If None, uses timestamp
            metadata: Additional metadata to save (Sharpe, episodes, etc.)
        
        Returns:
            filepath: Path to saved model
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        filepath = os.path.join(self.model_dir, f"sac_model_v{version}.zip")
        self.model.save(filepath)
        
        print(f"✅ Model saved: {filepath}")
        
        # Save metadata if provided
        if metadata:
            import json
            metadata_path = filepath.replace('.zip', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✅ Metadata saved: {metadata_path}")
        
        return filepath
    
    def load(self, filepath: str) -> None:
        """
        Load model from file
        
        Args:
            filepath: Path to model .zip file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found: {filepath}")
        
        self.model = SAC.load(filepath, env=self.env)
        print(f"✅ Model loaded: {filepath}")
    
    def evaluate(
        self, 
        eval_env,
        n_eval_episodes: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate agent on evaluation environment
        
        Args:
            eval_env: Evaluation environment
            n_eval_episodes: Number of episodes to evaluate
        
        Returns:
            metrics: Dict with mean_reward, std_reward, mean_episode_length
        """
        from stable_baselines3.common.evaluation import evaluate_policy
        
        mean_reward, std_reward = evaluate_policy(
            self.model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True
        )
        
        return {
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'n_episodes': n_eval_episodes
        }
    
    def get_replay_buffer_size(self) -> int:
        """
        Get current replay buffer size
        
        Returns:
            size: Number of transitions in buffer
        """
        return self.model.replay_buffer.size()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            info: Dict with model details
        """
        return {
            'algorithm': 'SAC',
            'policy': 'MlpPolicy',
            'network_architecture': [256, 256],
            'hyperparameters': self.hyperparameters,
            'action_space': 'Box([-1, 1])',  # Continuous position sizing
            'buffer_size': self.hyperparameters.get('buffer_size', 1000000),
            'current_buffer_size': self.get_replay_buffer_size(),
            'model_dir': self.model_dir
        }
