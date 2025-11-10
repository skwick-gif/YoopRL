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
from typing import Dict, List, Optional, Any, ClassVar, Union
from datetime import datetime, UTC
import json
from copy import deepcopy

from data_download.intraday_loader import ALLOWED_INTRADAY_SYMBOLS


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
    adx: Dict[str, Any] = field(default_factory=lambda: {'enabled': True, 'period': 14})
    multi_asset: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False,
        'symbols': ['SPY', 'QQQ', 'TLT', 'GLD']
    })

    # Engineered intraday features
    base_trend_context: bool = False
    base_momentum: bool = False
    base_trend_strength: bool = False
    base_extremes: bool = False
    leveraged_volatility: bool = False
    leveraged_momentum_short: bool = False
    time_context: bool = False
    position_context: bool = False
    
    # Alternative data
    sentiment: bool = False
    macro: bool = False
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
            'macro': False,
            'social_media': False,
            'news_headlines': False,
            'market_events': False,
            'fundamental': False,
            'multi_asset': {'enabled': False},
            'base_trend_context': False,
            'base_momentum': False,
            'base_trend_strength': False,
            'base_extremes': False,
            'leveraged_volatility': False,
            'leveraged_momentum_short': False,
            'time_context': False,
            'position_context': False,
            'recent_actions': True,
            'performance': {'enabled': True, 'period': '30d'},
            'reward_history': False
        },
        'SAC': {
            'bollinger': {'enabled': True, 'params': '20,2'},
            'stochastic': {'enabled': True, 'params': '14,3'},
            'sentiment': True,
            'macro': False,
            'social_media': True,
            'news_headlines': True,
            'market_events': True,
            'fundamental': False,
            'multi_asset': {
                'enabled': True,
                'symbols': ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']
            },
            'base_trend_context': False,
            'base_momentum': False,
            'base_trend_strength': False,
            'base_extremes': False,
            'leveraged_volatility': False,
            'leveraged_momentum_short': False,
            'time_context': False,
            'position_context': False,
            'recent_actions': {'enabled': True, 'length': 10},
            'performance': {'enabled': True, 'period': '14d'},
            'position_history': {'enabled': True, 'length': 10},
            'reward_history': {'enabled': True, 'length': 10}
        },
        'SAC_INTRADAY_DSR': {
            'price': True,
            'volume': True,
            'ohlc': True,
            'rsi': {'enabled': True, 'period': 14},
            'macd': {'enabled': True, 'params': '12,26,9'},
            'ema': {'enabled': True, 'periods': '10,50'},
            'vix': False,
            'bollinger': {'enabled': False},
            'stochastic': {'enabled': False},
            'adx': {'enabled': True, 'period': 14},
            'multi_asset': {'enabled': False, 'symbols': []},
            'sentiment': False,
            'macro': False,
            'social_media': False,
            'news_headlines': False,
            'market_events': False,
            'fundamental': False,
            'base_trend_context': True,
            'base_momentum': True,
            'base_trend_strength': True,
            'base_extremes': True,
            'leveraged_volatility': True,
            'leveraged_momentum_short': True,
            'time_context': True,
            'position_context': True,
            'recent_actions': False,
            'performance': False,
            'position_history': False,
            'reward_history': False
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
    commission: Union[float, Dict[str, Any]] = 0.005
    commission_model: str = 'ibkr_tiered_us_equities'
    commission_min_fee: float = 1.0
    commission_max_pct: float = 0.01
    initial_capital: float = 100000.0
    max_position_size: float = 1.0
    optuna_trials: int = 100
    normalize_obs: bool = True
    total_timesteps: Optional[int] = None
    max_total_timesteps: int = 1_000_000
    episode_budget: Optional[int] = None
    data_frequency: str = 'daily'
    interval: str = '1d'
    benchmark_symbol: Optional[str] = None
    benchmark_interval: Optional[str] = None
    reward_mode: Optional[str] = None
    train_split: float = 0.8
    dsr_config: Dict[str, Any] = field(default_factory=dict)
    intraday_enabled: bool = False
    
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
        
        if isinstance(self.commission, dict):
            per_share = float(self.commission.get('per_share', 0.0))
            min_fee = float(self.commission.get('min_fee', 0.0))
            max_pct = float(self.commission.get('max_pct', 0.0))

            if per_share <= 0:
                errors.append("Commission per share must be positive")
            if min_fee < 0:
                errors.append("Commission min_fee cannot be negative")
            if max_pct <= 0:
                errors.append("Commission max_pct must be positive")
        else:
            if self.commission < 0:
                errors.append("Commission cannot be negative")
            if self.commission > 0.05:
                errors.append("Commission per share appears unrealistically high (> $0.05)")

        if self.commission_min_fee < 0:
            errors.append("commission_min_fee cannot be negative")

        if self.commission_max_pct <= 0 or self.commission_max_pct > 0.05:
            errors.append("commission_max_pct must be in (0, 0.05]")
        
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

        frequency = (self.data_frequency or '').lower()
        intraday_flag = bool(self.intraday_enabled)
        if frequency and frequency not in {'daily', 'intraday', '15m', '15min'}:
            errors.append("data_frequency must be 'daily' or an intraday option ('intraday', '15m')")

        if not 0.0 < self.train_split < 1.0:
            errors.append("train_split must be between 0 and 1")

        if self.reward_mode and self.reward_mode.lower() not in {'dsr', 'return', 'default'}:
            errors.append("reward_mode must be one of: dsr, return, default")

        interval_lower = (self.interval or '').lower()
        if frequency in {'intraday', '15m', '15min'} and interval_lower not in {'15m', '15min'}:
            errors.append("Intraday modes currently require interval='15m'")
        if intraday_flag and interval_lower not in {'15m', '15min'}:
            errors.append("intraday_enabled requires interval='15m'")
        if intraday_flag and frequency not in {'intraday', '15m', '15min'}:
            errors.append("intraday_enabled requires data_frequency set to an intraday value")
        
        return errors


@dataclass
class TrainingConfig:
    """
    Complete training configuration.
    """
    agent_type: str  # 'PPO', 'SAC', or 'SAC_INTRADAY_DSR'
    symbol: str
    ppo_hyperparameters: Optional[PPOHyperparameters] = None
    sac_hyperparameters: Optional[SACHyperparameters] = None
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training_settings: TrainingSettings = field(default_factory=TrainingSettings)
    
    def __post_init__(self):
        """Initialize appropriate hyperparameters based on agent type."""
        canonical_agent = self.agent_type
        if self.agent_type == 'SAC_INTRADAY_DSR':
            canonical_agent = 'SAC'

        if canonical_agent == 'PPO' and self.ppo_hyperparameters is None:
            self.ppo_hyperparameters = PPOHyperparameters()
        elif canonical_agent == 'SAC' and self.sac_hyperparameters is None:
            if self.agent_type == 'SAC_INTRADAY_DSR':
                self.sac_hyperparameters = SACHyperparameters(
                    learning_rate=0.0001,
                    buffer_size=200_000,
                    batch_size=256,
                    tau=0.005,
                    gamma=0.99,
                    ent_coef='auto',
                    target_entropy='auto',
                    vol_penalty=-0.0,
                    episodes=250_000,
                )
            else:
                self.sac_hyperparameters = SACHyperparameters()

        # Ensure intraday defaults are set when required
        if self.agent_type == 'SAC_INTRADAY_DSR':
            if not isinstance(self.training_settings, TrainingSettings):
                self.training_settings = TrainingSettings(**asdict(self.training_settings))

            self.training_settings.data_frequency = 'intraday'
            self.training_settings.interval = '15m'
            if not self.training_settings.reward_mode:
                self.training_settings.reward_mode = 'dsr'
            if not self.training_settings.benchmark_symbol:
                self.training_settings.benchmark_symbol = _infer_benchmark_symbol(self.symbol)
            self.training_settings.benchmark_interval = '15m'
            if not self.training_settings.dsr_config:
                self.training_settings.dsr_config = {
                    'decay': 0.94,
                    'epsilon': 1e-9,
                    'warmup_steps': 200,
                    'clip_value': 6.0,
                }
            if self.training_settings.optuna_trials < 1:
                self.training_settings.optuna_trials = 0
            self.training_settings.intraday_enabled = True
        else:
            self.training_settings.intraday_enabled = bool(self.training_settings.intraday_enabled)

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
        supported_agents = {'PPO', 'SAC', 'SAC_INTRADAY_DSR'}
        if self.agent_type not in supported_agents:
            errors['agent_type'].append("Agent type must be one of: PPO, SAC, SAC_INTRADAY_DSR")

        symbol_upper = (self.symbol or '').upper()

        if self.agent_type == 'SAC_INTRADAY_DSR' and symbol_upper not in ALLOWED_INTRADAY_SYMBOLS:
            allowed_list = ', '.join(sorted(ALLOWED_INTRADAY_SYMBOLS))
            errors['training_settings'].append(
                f"Intraday agent symbol must be one of: {allowed_list}"
            )
        
        # Validate hyperparameters
        canonical_agent = self.agent_type if self.agent_type != 'SAC_INTRADAY_DSR' else 'SAC'

        if canonical_agent == 'PPO' and self.ppo_hyperparameters:
            errors['hyperparameters'] = self.ppo_hyperparameters.validate()
        elif canonical_agent == 'SAC' and self.sac_hyperparameters:
            errors['hyperparameters'] = self.sac_hyperparameters.validate()
        
        # Validate training settings
        errors['training_settings'] += self.training_settings.validate()
        
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
                max_position_size=0.85,
                optuna_trials=100
            )
        )


def get_sac_intraday_dsr_preset(symbol: str, benchmark_symbol: Optional[str] = None) -> TrainingConfig:
    """Preset for SAC + DSR intraday (15m) training."""

    benchmark_symbol = benchmark_symbol or _infer_benchmark_symbol(symbol)

    features = FeatureConfig.for_agent('SAC_INTRADAY_DSR')

    sac_params = SACHyperparameters(
        learning_rate=0.0001,
        buffer_size=200_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef='auto',
        target_entropy='auto',
        vol_penalty=0.0,
        episodes=250_000
    )

    now_str = datetime.now(UTC).strftime('%Y-%m-%d')

    training_settings = TrainingSettings(
        start_date='2018-01-01',
        end_date=now_str,
        commission={'per_share': 0.005, 'min_fee': 1.0, 'max_pct': 0.01},
        commission_model='ibkr_tiered_us_equities',
        commission_min_fee=1.0,
        commission_max_pct=0.01,
        max_position_size=1.0,
        optuna_trials=50,
        normalize_obs=True,
        episode_budget=1500,
        max_total_timesteps=1_000_000,
        data_frequency='intraday',
        interval='15m',
        benchmark_symbol=benchmark_symbol,
        benchmark_interval='15m',
        reward_mode='dsr',
        train_split=0.8,
        dsr_config={
            'decay': 0.94,
            'epsilon': 1e-9,
            'warmup_steps': 200,
            'clip_value': 6.0
        }
    )

    return TrainingConfig(
        agent_type='SAC_INTRADAY_DSR',
        symbol=symbol,
        sac_hyperparameters=sac_params,
        features=features,
        training_settings=training_settings
    )


def _infer_benchmark_symbol(symbol: str) -> str:
    mapping = {
        'TQQQ': 'QQQ',
        'SQQQ': 'QQQ',
        'UPRO': 'SPY',
        'SPXL': 'SPY',
        'TNA': 'IWM',
        'TMF': 'TLT'
    }
    return mapping.get(symbol.upper(), 'SPY')
