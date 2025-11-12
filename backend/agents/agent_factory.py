"""
agent_factory.py
Agent Factory for Creating RL Agents

Purpose:
- Factory pattern for creating agents dynamically
- Single entry point for agent creation
- Validation of agent type and parameters
- Simplifies agent instantiation in training loop

Why Factory Pattern:
- Decouples agent creation from training logic
- Easy to extend with new agent types
- Centralized validation and error handling
- Consistent interface for all agents

Supported Agents:
- PPO: For stock trading (discrete actions)
- SAC: For leveraged ETF trading (continuous actions)
- Future: DQN, A2C, TD3, etc.

Usage:
    agent = AgentFactory.create_agent(
        agent_type='PPO',
        env=stock_env,
        hyperparameters=config.ppo_hyperparameters
    )

Wiring:
- Used by Training Loop (Phase 4)
- Receives environments from Phase 2
- Receives configs from TrainingConfig
- Returns BaseAgent instances
"""

from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent


class AgentFactory:
    """
    Factory for creating RL trading agents
    
    Provides a single interface for instantiating different agent types.
    """
    
    # Supported agent types
    SUPPORTED_AGENTS = ['PPO', 'SAC']
    
    # Default model directories
    DEFAULT_MODEL_DIRS = {
        'PPO': 'backend/models/ppo',
        'SAC': 'backend/models/sac'
    }
    
    @staticmethod
    def create_agent(
        agent_type: str,
        env,
        hyperparameters: Dict[str, Any],
        model_dir: Optional[str] = None,
        *,
        seed: Optional[int] = None,
    ) -> BaseAgent:
        """
        Create an agent instance
        
        Args:
            agent_type: Type of agent ('PPO' or 'SAC')
            env: Trading environment (Gym-compatible)
            hyperparameters: Dict of algorithm-specific hyperparameters
            model_dir: Directory to save models (optional)
            seed: Optional random seed for deterministic initialization
        
        Returns:
            agent: Instance of BaseAgent (PPOAgent or SACAgent)
        
        Raises:
            ValueError: If agent_type is not supported
            TypeError: If hyperparameters is not a dict
        """
        # Validate agent type
        if agent_type not in AgentFactory.SUPPORTED_AGENTS:
            raise ValueError(
                f"Unsupported agent type: {agent_type}\n"
                f"Supported types: {AgentFactory.SUPPORTED_AGENTS}"
            )
        
        # Validate hyperparameters
        if not isinstance(hyperparameters, dict):
            raise TypeError(
                f"hyperparameters must be a dict, got {type(hyperparameters)}"
            )
        
        # Set default model directory if not provided
        if model_dir is None:
            model_dir = AgentFactory.DEFAULT_MODEL_DIRS[agent_type]
        
        # Create agent based on type
        if agent_type == 'PPO':
            agent = PPOAgent(
                env=env,
                hyperparameters=hyperparameters,
                model_dir=model_dir,
                seed=seed,
            )
        
        elif agent_type == 'SAC':
            agent = SACAgent(
                env=env,
                hyperparameters=hyperparameters,
                model_dir=model_dir,
                seed=seed,
            )
        
        print(f"✅ Agent created: {agent_type}")
        
        return agent
    
    @staticmethod
    def get_supported_agents() -> list:
        """
        Get list of supported agent types
        
        Returns:
            agents: List of agent type strings
        """
        return AgentFactory.SUPPORTED_AGENTS.copy()
    
    @staticmethod
    def is_supported(agent_type: str) -> bool:
        """
        Check if agent type is supported
        
        Args:
            agent_type: Agent type string
        
        Returns:
            supported: True if supported, False otherwise
        """
        return agent_type in AgentFactory.SUPPORTED_AGENTS
    
    @staticmethod
    def get_default_hyperparameters(agent_type: str) -> Dict[str, Any]:
        """
        Get default hyperparameters for agent type
        
        Args:
            agent_type: Type of agent ('PPO' or 'SAC')
        
        Returns:
            hyperparameters: Dict of default hyperparameters
        
        Raises:
            ValueError: If agent_type is not supported
        """
        if agent_type not in AgentFactory.SUPPORTED_AGENTS:
            raise ValueError(
                f"Unsupported agent type: {agent_type}"
            )
        
        if agent_type == 'PPO':
            return {
                'learning_rate': 0.0003,
                'gamma': 0.99,
                'batch_size': 256,
                'n_steps': 2048,
                'n_epochs': 10,
                'clip_range': 0.2,
                'ent_coef': 0.01
            }
        
        elif agent_type == 'SAC':
            return {
                'learning_rate': 0.0003,
                'buffer_size': 1000000,
                'batch_size': 256,
                'tau': 0.005,
                'gamma': 0.99,
                'ent_coef': 'auto',
                'target_entropy': 'auto'
            }
    
    @staticmethod
    def validate_hyperparameters(
        agent_type: str,
        hyperparameters: Dict[str, Any]
    ) -> bool:
        """
        Validate hyperparameters for agent type
        
        Args:
            agent_type: Type of agent ('PPO' or 'SAC')
            hyperparameters: Dict of hyperparameters to validate
        
        Returns:
            valid: True if valid, raises ValueError otherwise
        
        Raises:
            ValueError: If hyperparameters are invalid
        """
        if agent_type not in AgentFactory.SUPPORTED_AGENTS:
            raise ValueError(f"Unsupported agent type: {agent_type}")
        
        # Define required hyperparameters for each agent
        required = {
            'PPO': ['learning_rate', 'gamma', 'batch_size'],
            'SAC': ['learning_rate', 'gamma', 'batch_size', 'buffer_size']
        }
        
        missing = [
            key for key in required[agent_type] 
            if key not in hyperparameters
        ]
        
        if missing:
            raise ValueError(
                f"Missing required hyperparameters for {agent_type}: {missing}"
            )
        
        # Validate value ranges
        lr = hyperparameters.get('learning_rate')
        if lr and (lr <= 0 or lr > 0.1):
            raise ValueError(
                f"learning_rate must be in (0, 0.1], got {lr}"
            )
        
        gamma = hyperparameters.get('gamma')
        if gamma and (gamma < 0 or gamma > 1):
            raise ValueError(
                f"gamma must be in [0, 1], got {gamma}"
            )
        
        batch_size = hyperparameters.get('batch_size')
        if batch_size and batch_size < 1:
            raise ValueError(
                f"batch_size must be >= 1, got {batch_size}"
            )
        
        return True


# Convenience function for backward compatibility
def create_agent(agent_type: str, env, hyperparameters: Dict[str, Any], **kwargs):
    """
    Create an agent using the factory
    
    Args:
        agent_type: 'PPO' or 'SAC'
        env: Gym environment
        hyperparameters: Dict of hyperparameters
        **kwargs: Additional arguments (model_dir, etc.)
    
    Returns:
        Agent instance (PPOAgent or SACAgent)
    """
    return AgentFactory.create_agent(agent_type, env, hyperparameters, **kwargs)
