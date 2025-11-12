"""
ppo_agent.py
PPO Agent Wrapper for Stock Trading

Purpose:
- Wrapper around Stable-Baselines3 PPO implementation
- Optimized for stock trading with discrete actions (HOLD/BUY/SELL)
- MLP policy with 2 hidden layers [64, 64]
- Integrates with StockTradingEnv from Phase 2

Why PPO for stocks:
- On-policy algorithm: stable training
- Discrete action space: binary position logic
- Risk-averse: good for conservative stock trading
- Lower variance: consistent performance

Hyperparameters (from TrainingConfig):
- learning_rate: 0.0001-0.001 (default: 0.0003)
- gamma: discount factor 0.95-0.999 (default: 0.99)
- batch_size: 64-512 (default: 256)
- n_steps: rollout buffer size (default: 2048)
- n_epochs: optimization epochs per update (default: 10)

Wiring:
- Receives environment from Phase 2 (StockTradingEnv)
- Receives hyperparameters from TrainingConfig
- Saves models via ModelManager
- Used by Training Loop in Phase 4
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import os
from datetime import datetime
from typing import Optional, Dict, Any, Callable
import numpy as np
import torch


class PPOAgent:
    """
    PPO Agent for Stock Trading
    
    Wraps Stable-Baselines3 PPO with stock-specific configurations.
    
    Attributes:
        env: Trading environment (StockTradingEnv)
        model: PPO model instance
        hyperparameters: Dict of hyperparameters
        model_dir: Directory for saving models
    """
    
    def __init__(
        self, 
        env, 
        hyperparameters: Dict[str, Any],
        model_dir: str = 'backend/models/ppo',
        *,
        seed: Optional[int] = None
    ):
        """
        Initialize PPO Agent
        
        Args:
            env: Gym environment (StockTradingEnv)
            hyperparameters: Dict containing:
                - learning_rate: float
                - gamma: float
                - batch_size: int
                - n_steps: int
                - n_epochs: int
                - clip_range: float (optional)
                - ent_coef: float (optional)
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
        gamma = hyperparameters.get('gamma', 0.99)
        batch_size = hyperparameters.get('batch_size', 256)
        n_steps = hyperparameters.get('n_steps', 2048)
        n_epochs = hyperparameters.get('n_epochs', 10)
        clip_range = hyperparameters.get('clip_range', 0.2)
        ent_coef = hyperparameters.get('ent_coef', 0.01)
        
        # Auto-detect CUDA/CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create PPO model
        self.model = PPO(
            policy='MlpPolicy',
            env=self.env,
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            n_steps=n_steps,
            n_epochs=n_epochs,
            clip_range=clip_range,
            ent_coef=ent_coef,
            policy_kwargs={
                'net_arch': [64, 64]  # MLP: 2 hidden layers, 64 neurons each
            },
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
        
        print(f"✅ PPO Agent initialized")
        print(f"   Device: {device.upper()}")
        if device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Learning Rate: {learning_rate}")
        print(f"   Gamma: {gamma}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Network: [64, 64]")
    
    def train(
        self, 
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        progress_callback: Optional[Callable] = None
    ) -> None:
        """
        Train the PPO agent
        
        Args:
            total_timesteps: Number of timesteps to train
            callback: SB3 callback for logging/checkpointing
            progress_callback: Custom callback for progress updates
        """
        print(f"\n{'='*60}")
        print(f"🏋️ Training PPO Agent...")
        print(f"   Total Timesteps: {total_timesteps:,}")
        print(f"{'='*60}\n")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"{'='*60}\n")
    
    def predict(
        self, 
        observation: np.ndarray,
        deterministic: bool = True
    ) -> int:
        """
        Predict action for given observation
        
        Args:
            observation: State vector from environment
            deterministic: If True, use deterministic policy (no exploration)
        
        Returns:
            action: Integer action (0=HOLD, 1=BUY, 2=SELL)
        """
        action, _states = self.model.predict(
            observation, 
            deterministic=deterministic
        )
        return int(action)
    
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
        filepath = os.path.join(self.model_dir, f"ppo_model_v{version}.zip")
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
        
        self.model = PPO.load(filepath, env=self.env)
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            info: Dict with model details
        """
        return {
            'algorithm': 'PPO',
            'policy': 'MlpPolicy',
            'network_architecture': [64, 64],
            'hyperparameters': self.hyperparameters,
            'action_space': 'Discrete(3)',  # HOLD, BUY, SELL
            'model_dir': self.model_dir
        }
