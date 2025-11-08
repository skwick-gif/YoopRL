"""
base_agent.py
Abstract Base Agent Interface

Purpose:
- Define consistent interface for all RL agents (PPO, SAC, future agents)
- Abstract base class using ABC (Abstract Base Classes)
- Ensures all agents implement required methods
- Common utilities for logging and metrics tracking

Why Abstract Interface:
- Enforces API consistency across agents
- Easy to add new agents (DQN, A2C, etc.) in future
- Factory pattern can rely on consistent interface
- Simplifies training loop - works with any agent

Required Methods:
- train(): Train the agent
- predict(): Get action from observation
- save(): Save model to disk
- load(): Load model from disk
- evaluate(): Evaluate on test environment
- get_model_info(): Return model metadata

Wiring:
- PPOAgent and SACAgent inherit from this
- AgentFactory creates agents via this interface
- Training Loop uses this interface generically
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class BaseAgent(ABC):
    """
    Abstract Base Class for RL Trading Agents
    
    All trading agents (PPO, SAC, etc.) must inherit from this class
    and implement the abstract methods.
    
    This ensures a consistent API across different agent types.
    """
    
    def __init__(
        self, 
        env, 
        hyperparameters: Dict[str, Any],
        model_dir: str
    ):
        """
        Initialize base agent
        
        Args:
            env: Trading environment (Gym-compatible)
            hyperparameters: Dict of algorithm-specific hyperparameters
            model_dir: Directory to save models
        """
        self.env = env
        self.hyperparameters = hyperparameters
        self.model_dir = model_dir
        self.model = None  # Will be set by child classes
    
    @abstractmethod
    def train(
        self, 
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        progress_callback: Optional[Callable] = None
    ) -> None:
        """
        Train the agent
        
        Args:
            total_timesteps: Number of timesteps to train
            callback: Stable-Baselines3 callback
            progress_callback: Custom progress callback
        """
        pass
    
    @abstractmethod
    def predict(
        self, 
        observation: np.ndarray,
        deterministic: bool = True
    ):
        """
        Predict action for given observation
        
        Args:
            observation: State vector from environment
            deterministic: If True, use deterministic policy
        
        Returns:
            action: Action to take (discrete int or continuous array)
        """
        pass
    
    @abstractmethod
    def save(
        self, 
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save model to disk
        
        Args:
            version: Version identifier
            metadata: Additional metadata to save
        
        Returns:
            filepath: Path to saved model
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """
        Load model from disk
        
        Args:
            filepath: Path to model file
        """
        pass
    
    @abstractmethod
    def evaluate(
        self, 
        eval_env,
        n_eval_episodes: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate agent performance
        
        Args:
            eval_env: Evaluation environment
            n_eval_episodes: Number of episodes to run
        
        Returns:
            metrics: Dict with evaluation metrics
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            info: Dict with model details (algorithm, architecture, etc.)
        """
        pass
    
    # Common utility methods (non-abstract, available to all agents)
    
    def log_training_start(self, total_timesteps: int) -> None:
        """
        Log training start information
        
        Args:
            total_timesteps: Total training timesteps
        """
        print(f"\n{'='*60}")
        print(f"🚀 Starting Training: {self.__class__.__name__}")
        print(f"   Total Timesteps: {total_timesteps:,}")
        print(f"   Hyperparameters: {self.hyperparameters}")
        print(f"{'='*60}\n")
    
    def log_training_end(self) -> None:
        """Log training completion"""
        print(f"\n{'='*60}")
        print(f"✅ Training Complete: {self.__class__.__name__}")
        print(f"{'='*60}\n")
    
    def validate_hyperparameters(self, required_keys: list) -> None:
        """
        Validate that required hyperparameters are present
        
        Args:
            required_keys: List of required hyperparameter keys
        
        Raises:
            ValueError: If required keys are missing
        """
        missing = [k for k in required_keys if k not in self.hyperparameters]
        if missing:
            raise ValueError(
                f"Missing required hyperparameters: {missing}"
            )
