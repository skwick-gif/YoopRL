"""
training_config.py
Training Configuration Dataclasses and Presets

Purpose:
- Define structured configuration for RL training
- Validate hyperparameters before training
- Provide preset configurations (Conservative, Aggressive, Balanced)
- Serialize/deserialize for saving/loading

Why separate file:
- Type-safe configuration with dataclasses
- Centralized validation logic
- Easy to extend with new parameters
- Consistent config format across codebase

Configuration Sections:
1. Agent hyperparameters (PPO/SAC specific)
2. Training settings (episodes, dates, commission)
3. Feature selection (technical indicators, alternative data)
4. Environment parameters (capital, position size)

Presets:
- Conservative: Low risk, stable returns, high Sharpe
- Aggressive: High risk, high returns, leveraged
- Balanced: Moderate risk-return profile

Wiring:
- Used by train.py to configure training runs
- Received from frontend via API
- Validated before environment/agent creation
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, ClassVar
from datetime import datetime
import json
from copy import deepcopy


@dataclass
class PPOHyperparameters:
    """
    PPO-specific hyperparameters.
    """
    learning_rate: float = 0.0003
    gamma: float = 0.99
    batch_size: int = 256
    n_epochs: int = 10
    clip_range: float = 0.2
    ent_coef: float = 0.01
    risk_penalty: float = -0.5
    episodes: int = 50000
    
    def validate(self) -> List[str]:
        """Validate PPO hyperparameters."""
        errors = []
        
        if self.learning_rate <= 0 or self.learning_rate > 0.1:
            errors.append("Learning rate must be between 0 and 0.1")
        
        if self.gamma < 0 or self.gamma > 1:
            errors.append("Gamma must be between 0 and 1")
        
        if self.batch_size < 32:
            errors.append("Batch size must be at least 32")
        
        if self.episodes < 1000:
            errors.append("Episodes must be at least 1000")
        
        if self.risk_penalty > 0:
            errors.append("Risk penalty should be negative")
        
        return errors


@dataclass
class SACHyperparameters:
    """
    SAC-specific hyperparameters.
    """
    learning_rate: float = 0.0003
    buffer_size: int = 100000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    ent_coef: str = 'auto'  # or float
    target_entropy: str = 'auto'  # or float
    vol_penalty: float = -0.3
    episodes: int = 45000
    
    def validate(self) -> List[str]:
        """Validate SAC hyperparameters."""
        errors = []
        
        if self.learning_rate <= 0 or self.learning_rate > 0.1:
            errors.append("Learning rate must be between 0 and 0.1")
        
        if self.gamma < 0 or self.gamma > 1:
            errors.append("Gamma must be between 0 and 1")
        
        if self.batch_size < 32:
            errors.append("Batch size must be at least 32")
        
        if self.episodes < 1000:
            errors.append("Episodes must be at least 1000")
        
        if self.tau <= 0 or self.tau >= 1:
            errors.append("Tau must be between 0 and 1")
        
        if self.vol_penalty > 0:
            errors.append("Volatility penalty should be negative")
        
        return errors


def _deep_merge_dict(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries without mutating inputs."""
    result = deepcopy(base)

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value

    return result


@dataclass
class FeatureConfig:
    """
    Feature selection configuration.
    """
    # Price data (always enabled)
    price: bool = True
    volume: bool = True
    ohlc: bool = True
    
    # Technical indicators
    rsi: Dict[str, Any] = field(default_factory=lambda: {'enabled': True, 'period': 14})
    macd: Dict[str, Any] = field(default_factory=lambda: {'enabled': True, 'params': '12,26,9'})
    ema: Dict[str, Any] = field(default_factory=lambda: {'enabled': True, 'periods': '10,50'})
    vix: bool = True
    bollinger: Dict[str, Any] = field(default_factory=lambda: {'enabled': False, 'params': '20,2'})
    stochastic: Dict[str, Any] = field(default_factory=lambda: {'enabled': False, 'params': '14,3'})
    multi_asset: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False,
        'symbols': ['SPY', 'QQQ', 'TLT', 'GLD']
    })
    
    # Alternative data
    sentiment: bool = False
    social_media: bool = False
    news_headlines: bool = False
    market_events: bool = False
    fundamental: bool = False
    
    # Agent history
    recent_actions: Any = True
    performance: Dict[str, Any] = field(default_factory=lambda: {'enabled': True, 'period': '30d'})
    position_history: Any = True
    reward_history: Any = False
    
    # LLM integration
    llm: Dict[str, Any] = field(default_factory=lambda: {'enabled': False, 'provider': 'Perplexity API'})

    AGENT_PROFILES: ClassVar[Dict[str, Dict[str, Any]]] = {
        'PPO': {
            'bollinger': {'enabled': False},
            'stochastic': {'enabled': False},
            'sentiment': False,
            'social_media': False,
            'news_headlines': False,
            'market_events': False,
            'fundamental': False,
            'multi_asset': {'enabled': False},
            'recent_actions': True,
            'performance': {'enabled': True, 'period': '30d'},
            'reward_history': False
        },
        'SAC': {
            'bollinger': {'enabled': True, 'params': '20,2'},
            'stochastic': {'enabled': True, 'params': '14,3'},
            'sentiment': True,
            'social_media': True,
            'news_headlines': True,
            'market_events': True,
            'fundamental': False,
            'multi_asset': {
                'enabled': True,
                'symbols': ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']
            },
            'recent_actions': {'enabled': True, 'length': 10},
            'performance': {'enabled': True, 'period': '14d'},
            'position_history': {'enabled': True, 'length': 10},
            'reward_history': {'enabled': True, 'length': 10}
        }
    }

    @classmethod
    def for_agent(cls, agent_type: str, overrides: Optional[Dict[str, Any]] = None) -> 'FeatureConfig':
        """Create a feature configuration tailored to the requested agent."""
        base_data = asdict(cls())
        profile = cls.AGENT_PROFILES.get(agent_type.upper(), {})
        merged = _deep_merge_dict(base_data, profile)

        if overrides:
            merged = _deep_merge_dict(merged, overrides)

        return cls(**merged)

    def with_overrides(self, overrides: Dict[str, Any]) -> 'FeatureConfig':
        """Return a copy of this configuration with overrides applied."""
        merged = _deep_merge_dict(asdict(self), overrides)
        return FeatureConfig(**merged)

    def to_payload(self) -> Dict[str, Any]:
        """Return the configuration as a plain dictionary payload."""
        return asdict(self)


@dataclass
class TrainingSettings:
    """
    General training settings.
    """
    start_date: str = '2023-01-01'
    end_date: str = '2024-11-01'
    commission: float = 1.0
    initial_capital: float = 100000.0
    max_position_size: float = 1.0
    optuna_trials: int = 100
    normalize_obs: bool = True
    total_timesteps: Optional[int] = None
    max_total_timesteps: int = 1_000_000
    episode_budget: Optional[int] = None
    
    def validate(self) -> List[str]:
        """Validate training settings."""
        errors = []
        
        # Validate dates
        try:
            start = datetime.strptime(self.start_date, '%Y-%m-%d')
            end = datetime.strptime(self.end_date, '%Y-%m-%d')
            
            if start >= end:
                errors.append("Start date must be before end date")
        except ValueError:
            errors.append("Invalid date format (use YYYY-MM-DD)")
        
        if self.commission < 0:
            errors.append("Commission cannot be negative")
        
        if self.initial_capital <= 0:
            errors.append("Initial capital must be positive")
        
        if self.max_position_size <= 0 or self.max_position_size > 1:
            errors.append("Max position size must be between 0 and 1")
        
        if self.optuna_trials < 10:
            errors.append("Optuna trials must be at least 10")

        if self.total_timesteps is not None and self.total_timesteps <= 0:
            errors.append("Total timesteps must be positive")

        if self.max_total_timesteps is not None and self.max_total_timesteps <= 0:
            errors.append("Max total timesteps must be positive")

        if self.episode_budget is not None and self.episode_budget <= 0:
            errors.append("Episode budget must be positive when provided")

        if self.total_timesteps and self.max_total_timesteps and self.total_timesteps > self.max_total_timesteps:
            errors.append("Total timesteps cannot exceed max total timesteps")
        
        return errors


@dataclass
class TrainingConfig:
    """
    Complete training configuration.
    """
    agent_type: str  # 'PPO' or 'SAC'
    symbol: str
    ppo_hyperparameters: Optional[PPOHyperparameters] = None
    sac_hyperparameters: Optional[SACHyperparameters] = None
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training_settings: TrainingSettings = field(default_factory=TrainingSettings)
    
    def __post_init__(self):
        """Initialize appropriate hyperparameters based on agent type."""
        if self.agent_type == 'PPO' and self.ppo_hyperparameters is None:
            self.ppo_hyperparameters = PPOHyperparameters()
        elif self.agent_type == 'SAC' and self.sac_hyperparameters is None:
            self.sac_hyperparameters = SACHyperparameters()

        # Normalize feature configuration
        if isinstance(self.features, dict):
            self.features = FeatureConfig(**self.features)
        elif self.features is None:
            self.features = FeatureConfig.for_agent(self.agent_type)
        elif isinstance(self.features, FeatureConfig):
            default_features = FeatureConfig()
            if self.features == default_features:
                self.features = FeatureConfig.for_agent(self.agent_type)
        else:
            # Any unexpected type falls back to agent-specific defaults
            self.features = FeatureConfig.for_agent(self.agent_type)
    
    def validate(self) -> Dict[str, List[str]]:
        """
        Validate entire configuration.
        
        Returns:
            Dictionary with validation errors by section
        """
        errors = {
            'agent_type': [],
            'hyperparameters': [],
            'training_settings': []
        }
        
        # Validate agent type
        if self.agent_type not in ['PPO', 'SAC']:
            errors['agent_type'].append("Agent type must be 'PPO' or 'SAC'")
        
        # Validate hyperparameters
        if self.agent_type == 'PPO' and self.ppo_hyperparameters:
            errors['hyperparameters'] = self.ppo_hyperparameters.validate()
        elif self.agent_type == 'SAC' and self.sac_hyperparameters:
            errors['hyperparameters'] = self.sac_hyperparameters.validate()
        
        # Validate training settings
        errors['training_settings'] = self.training_settings.validate()
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        errors = self.validate()
        return all(len(e) == 0 for e in errors.values())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrainingConfig':
        """Create from dictionary."""
        # Reconstruct nested dataclasses
        if 'ppo_hyperparameters' in data and data['ppo_hyperparameters']:
            data['ppo_hyperparameters'] = PPOHyperparameters(**data['ppo_hyperparameters'])
        
        if 'sac_hyperparameters' in data and data['sac_hyperparameters']:
            data['sac_hyperparameters'] = SACHyperparameters(**data['sac_hyperparameters'])
        
        if 'features' in data:
            features_payload = data['features']
            if isinstance(features_payload, FeatureConfig):
                data['features'] = features_payload
            elif isinstance(features_payload, dict):
                data['features'] = FeatureConfig(**features_payload)
            else:
                data['features'] = FeatureConfig.for_agent(data.get('agent_type', 'PPO'))
        else:
            data['features'] = FeatureConfig.for_agent(data.get('agent_type', 'PPO'))
        
        if 'training_settings' in data:
            data['training_settings'] = TrainingSettings(**data['training_settings'])
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TrainingConfig':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


# ===== PRESET CONFIGURATIONS =====

def get_conservative_preset(symbol: str, agent_type: str = 'PPO') -> TrainingConfig:
    """
    Conservative trading strategy.
    
    Focus: Low risk, stable returns, high Sharpe ratio
    Best for: Risk-averse investors, retirement accounts
    """
    if agent_type == 'PPO':
        features = FeatureConfig.for_agent('PPO').with_overrides({
            'bollinger': {'enabled': False},
            'stochastic': {'enabled': False},
            'recent_actions': {'enabled': True, 'length': 5},
            'performance': {'enabled': True, 'period': '60d'},
            'reward_history': False
        })
        return TrainingConfig(
            agent_type='PPO',
            symbol=symbol,
            ppo_hyperparameters=PPOHyperparameters(
                learning_rate=0.0001,
                gamma=0.99,
                batch_size=512,
                risk_penalty=-0.8,
                episodes=60000
            ),
            features=features,
            training_settings=TrainingSettings(
                commission=1.0,
                max_position_size=0.7,
                optuna_trials=150
            )
        )
    else:  # SAC
        features = FeatureConfig.for_agent('SAC').with_overrides({
            'bollinger': {'enabled': True, 'params': '30,2'},
            'stochastic': {'enabled': True, 'params': '21,3'},
            'sentiment': False,
            'market_events': False,
            'multi_asset': {'enabled': True, 'symbols': ['SPY', 'QQQ', 'TLT']},
            'reward_history': {'enabled': True, 'length': 6}
        })
        return TrainingConfig(
            agent_type='SAC',
            symbol=symbol,
            sac_hyperparameters=SACHyperparameters(
                learning_rate=0.0001,
                vol_penalty=-0.6,
                episodes=55000
            ),
            features=features,
            training_settings=TrainingSettings(
                commission=1.0,
                max_position_size=0.7,
                optuna_trials=150
            )
        )


def get_aggressive_preset(symbol: str, agent_type: str = 'SAC') -> TrainingConfig:
    """
    Aggressive trading strategy.
    
    Focus: High returns, tolerates higher volatility
    Best for: Risk-tolerant traders, growth accounts
    """
    if agent_type == 'SAC':
        features = FeatureConfig.for_agent('SAC').with_overrides({
            'bollinger': {'enabled': True, 'params': '20,3'},
            'stochastic': {'enabled': True, 'params': '10,3'},
            'sentiment': True,
            'market_events': True,
            'social_media': True,
            'multi_asset': {'enabled': True, 'symbols': ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']},
            'recent_actions': {'enabled': True, 'length': 12},
            'reward_history': {'enabled': True, 'length': 12}
        })
        return TrainingConfig(
            agent_type='SAC',
            symbol=symbol,
            sac_hyperparameters=SACHyperparameters(
                learning_rate=0.0005,
                vol_penalty=-0.1,
                episodes=40000
            ),
            features=features,
            training_settings=TrainingSettings(
                commission=1.0,
                max_position_size=1.0,
                optuna_trials=100
            )
        )
    else:  # PPO
        features = FeatureConfig.for_agent('PPO').with_overrides({
            'bollinger': {'enabled': True, 'params': '20,2'},
            'stochastic': {'enabled': True, 'params': '14,3'},
            'recent_actions': {'enabled': True, 'length': 8},
            'reward_history': {'enabled': True, 'length': 6}
        })
        return TrainingConfig(
            agent_type='PPO',
            symbol=symbol,
            ppo_hyperparameters=PPOHyperparameters(
                learning_rate=0.0005,
                risk_penalty=-0.2,
                episodes=40000
            ),
            features=features,
            training_settings=TrainingSettings(
                commission=1.0,
                max_position_size=1.0,
                optuna_trials=100
            )
        )


def get_balanced_preset(symbol: str, agent_type: str = 'PPO') -> TrainingConfig:
    """
    Balanced trading strategy.
    
    Focus: Moderate risk-return profile
    Best for: General investors, balanced portfolios
    """
    if agent_type == 'PPO':
        features = FeatureConfig.for_agent('PPO').with_overrides({
            'bollinger': {'enabled': True, 'params': '20,2'},
            'stochastic': {'enabled': False},
            'recent_actions': {'enabled': True, 'length': 6},
            'reward_history': {'enabled': True, 'length': 4}
        })
        return TrainingConfig(
            agent_type='PPO',
            symbol=symbol,
            ppo_hyperparameters=PPOHyperparameters(
                learning_rate=0.0003,
                gamma=0.99,
                batch_size=256,
                risk_penalty=-0.5,
                episodes=50000
            ),
            features=features,
            training_settings=TrainingSettings(
                commission=1.0,
                max_position_size=0.85,
                optuna_trials=100
            )
        )
    else:  # SAC
        features = FeatureConfig.for_agent('SAC').with_overrides({
            'bollinger': {'enabled': True, 'params': '20,2'},
            'stochastic': {'enabled': True, 'params': '14,3'},
            'sentiment': True,
            'market_events': True,
            'multi_asset': {'enabled': True, 'symbols': ['SPY', 'QQQ', 'IWM']},
            'reward_history': {'enabled': True, 'length': 8}
        })
        return TrainingConfig(
            agent_type='SAC',
            symbol=symbol,
            sac_hyperparameters=SACHyperparameters(
                learning_rate=0.0003,
                vol_penalty=-0.3,
                episodes=45000
            ),
            features=features,
            training_settings=TrainingSettings(
                commission=1.0,
                max_position_size=0.85,
                optuna_trials=100
            )
        )
