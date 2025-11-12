"""
model_manager.py
RL Model Management and Versioning System

Purpose:
- Save and load trained RL models with metadata
- Version control for models (v1.0, v1.1, v2.0, etc.)
- Track model performance metrics
- Manage model lifecycle (active, archived, deprecated)

Why separate file:
- Centralized model storage and retrieval
- Consistent metadata format across all models
- Easy model comparison and rollback
- Prevents model file conflicts

Model Metadata:
- Version number and creation date
- Training hyperparameters
- Performance metrics (Sharpe, Sortino, Win Rate)
- Dataset information (symbol, date range)
- Agent type (PPO/SAC)

Storage Structure:
backend/models/
├── ppo/
│   ├── ppo_aapl_v1.0.zip
│   ├── ppo_aapl_v1.0_metadata.json
│   ├── ppo_aapl_v1.1.zip
│   └── ppo_aapl_v1.1_metadata.json
└── sac/
    ├── sac_tna_v1.0.zip
    └── sac_tna_v1.0_metadata.json

Wiring:
- Used by train.py to save models after training
- Used by API endpoints to load models for backtesting/live trading
- Frontend ModelSelector.jsx displays models from list_models()
"""

import os
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import shutil


logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages RL model storage, versioning, and metadata.
    """
    
    def __init__(self, base_dir: str = "backend/models"):
        """
        Initialize model manager.
        
        Args:
            base_dir: Base directory for model storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each agent type
        (self.base_dir / "ppo").mkdir(exist_ok=True)
        (self.base_dir / "sac").mkdir(exist_ok=True)
    
    def save_model(
        self,
        model: Any,
        agent_type: str,
        symbol: str,
        metadata: Dict[str, Any],
        version: Optional[str] = None
    ) -> str:
        """
        Save model with metadata.
        
        Args:
            model: Trained model object (Stable-Baselines3 model)
            agent_type: 'PPO' or 'SAC'
            symbol: Trading symbol (e.g., 'AAPL', 'TNA')
            metadata: Dictionary with training info and metrics
            version: Optional version string (auto-generated if None)
            
        Returns:
            model_id: Unique identifier for saved model
        """
        agent_type = agent_type.lower()
        symbol = symbol.upper()
        
        # Generate version if not provided
        if version is None:
            version = self._generate_next_version(agent_type, symbol)
        
        # Create model ID
        model_id = f"{agent_type}_{symbol}_{version}"
        
        # Model directory
        model_dir = self.base_dir / agent_type
        model_path = model_dir / f"{model_id}.zip"
        metadata_path = model_dir / f"{model_id}_metadata.json"
        
        # Save model (Stable-Baselines3 format)
        try:
            model.save(str(model_path.with_suffix('')))  # .save() adds .zip automatically
        except AttributeError:
            # Fallback for non-SB3 models
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Prepare metadata
        full_metadata = {
            'model_id': model_id,
            'agent_type': agent_type.upper(),
            'symbol': symbol,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'file_path': str(model_path),
            **metadata
        }
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
        
        print(f"✅ Model saved: {model_id}")
        return model_id

    def update_model_metadata(
        self,
        agent_type: str,
        symbol: str,
        version: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update metadata for an existing model without touching the model file."""

        agent_dir = self.base_dir / agent_type.lower()
        metadata_path = agent_dir / f"{agent_type.lower()}_{symbol.upper()}_{version}_metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        merged = metadata.copy()
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            else:
                merged[key] = value

        with open(metadata_path, 'w') as f:
            json.dump(merged, f, indent=2)

        return merged
    
    def load_model(
        self,
        model_id: str = None,
        model_path: str = None
    ) -> tuple:
        """
        Load model and metadata.
        
        Args:
            model_id: Model identifier (e.g., 'ppo_aapl_v1.0')
            model_path: Direct path to model file (alternative to model_id)
            
        Returns:
            (model, metadata): Loaded model and its metadata
        """
        if model_path:
            # Load from direct path
            metadata_path = Path(model_path).with_name(
                Path(model_path).stem + '_metadata.json'
            )
        else:
            # Find model by ID
            agent_type = model_id.split('_')[0]
            model_path = self.base_dir / agent_type / f"{model_id}.zip"
            metadata_path = self.base_dir / agent_type / f"{model_id}_metadata.json"
        
        # Load metadata
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load model
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Try Stable-Baselines3 load first
        try:
            from stable_baselines3 import PPO, SAC
            agent_type = metadata['agent_type'].upper()
            if agent_type == 'SAC_INTRADAY_DSR':
                agent_type = 'SAC'
            
            if agent_type == 'PPO':
                model = PPO.load(str(Path(model_path).with_suffix('')))
            elif agent_type == 'SAC':
                model = SAC.load(str(Path(model_path).with_suffix('')))
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
        except Exception as e:
            # Fallback to pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        print(f"✅ Model loaded: {metadata['model_id']}")
        return model, metadata
    
    def list_models(
        self,
        agent_type: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all available models with metadata.
        
        Args:
            agent_type: Filter by 'PPO' or 'SAC' (optional)
            symbol: Filter by symbol (optional)
            
        Returns:
            List of metadata dictionaries
        """
        models = []
        
        # Determine which directories to search
        if agent_type:
            search_dirs = [self.base_dir / agent_type.lower()]
        else:
            search_dirs = [self.base_dir / "ppo", self.base_dir / "sac"]
        
        # Find all metadata files
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            
            for metadata_file in search_dir.glob("*_metadata.json"):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Apply symbol filter if specified
                if symbol and metadata.get('symbol') != symbol.upper():
                    continue
                
                models.append(metadata)
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return models
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a trained model, its metadata, and optional artifacts."""

        if not model_id:
            return False

        agent_type = (model_id.split('_')[0] or '').lower()
        model_dir = self.base_dir / agent_type

        candidates = [
            model_dir / f"{model_id}.zip",
            model_dir / f"{model_id}.pkl",
            model_dir / f"{model_id}.bin",
        ]

        metadata_path = model_dir / f"{model_id}_metadata.json"
        deleted_any = False

        for candidate in candidates:
            try:
                if candidate.exists():
                    candidate.unlink()
                    deleted_any = True
            except Exception as exc:
                logger.warning("Failed to remove model artifact %s: %s", candidate, exc)

        normalizer_candidate: Optional[Path] = None
        if metadata_path.exists():
            try:
                with metadata_path.open('r', encoding='utf-8') as handle:
                    metadata = json.load(handle)
                normalizer_raw = metadata.get('normalizer_path') or metadata.get('normalizer')
                if normalizer_raw:
                    normalizer_candidate = Path(str(normalizer_raw))
                    if not normalizer_candidate.is_absolute():
                        root_dir = self.base_dir.parent.parent if self.base_dir.parent else Path('.')
                        normalizer_candidate = (root_dir / normalizer_candidate).resolve()
                    try:
                        normalizer_candidate.relative_to(self.base_dir.parent)
                    except ValueError:
                        normalizer_candidate = None
            except Exception as exc:  # pragma: no cover - metadata best effort
                logger.warning("Failed to parse metadata for %s before deletion: %s", model_id, exc)
            try:
                metadata_path.unlink()
                deleted_any = True
            except Exception as exc:
                logger.warning("Failed to remove metadata file %s: %s", metadata_path, exc)

        if normalizer_candidate and normalizer_candidate.exists():
            try:
                normalizer_candidate.unlink()
            except Exception as exc:  # pragma: no cover - cleanup safeguard
                logger.warning("Failed to remove normalizer artifact %s: %s", normalizer_candidate, exc)

        if deleted_any:
            print(f"✅ Model deleted: {model_id}")
        else:
            print(f"⚠️ Model artifacts not found: {model_id}")

        return deleted_any
    
    def archive_model(self, model_id: str) -> bool:
        """
        Move model to archive directory.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if successful
        """
        agent_type = model_id.split('_')[0]
        archive_dir = self.base_dir / agent_type / "archive"
        archive_dir.mkdir(exist_ok=True)
        
        model_path = self.base_dir / agent_type / f"{model_id}.zip"
        metadata_path = self.base_dir / agent_type / f"{model_id}_metadata.json"
        
        if model_path.exists():
            shutil.move(str(model_path), str(archive_dir / model_path.name))
        
        if metadata_path.exists():
            shutil.move(str(metadata_path), str(archive_dir / metadata_path.name))
        
        print(f"✅ Model archived: {model_id}")
        return True
    
    def _generate_next_version(self, agent_type: str, symbol: str) -> str:
        """
        Generate next version number for a symbol.
        
        Args:
            agent_type: 'ppo' or 'sac'
            symbol: Trading symbol
            
        Returns:
            Version string (e.g., 'v1.0', 'v1.1', 'v2.0')
        """
        # Find existing versions
        existing_models = self.list_models(agent_type=agent_type, symbol=symbol)
        
        if not existing_models:
            return "v1.0"
        
        # Extract version numbers
        versions = []
        for model in existing_models:
            version_str = model.get('version', 'v1.0')
            try:
                # Parse version (e.g., 'v1.2' -> [1, 2])
                major, minor = version_str.lstrip('v').split('.')
                versions.append((int(major), int(minor)))
            except:
                continue
        
        if not versions:
            return "v1.0"
        
        # Get latest version and increment
        latest = max(versions)
        new_minor = latest[1] + 1
        
        return f"v{latest[0]}.{new_minor}"
    
    def get_best_model(
        self,
        agent_type: str,
        symbol: str,
        metric: str = 'sharpe_ratio'
    ) -> Optional[Dict[str, Any]]:
        """
        Get best performing model for a symbol based on a metric.
        
        Args:
            agent_type: 'PPO' or 'SAC'
            symbol: Trading symbol
            metric: Metric to compare ('sharpe_ratio', 'total_return', etc.)
            
        Returns:
            Metadata of best model, or None if no models found
        """
        models = self.list_models(agent_type=agent_type, symbol=symbol)
        
        if not models:
            return None
        
        # Filter models that have the metric
        models_with_metric = [m for m in models if metric in m.get('metrics', {})]
        
        if not models_with_metric:
            return None
        
        # Find best
        best_model = max(models_with_metric, key=lambda x: x.get('metrics', {}).get(metric, 0))
        return best_model
    
    def compare_models(
        self,
        model_ids: List[str],
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple models across various metrics.
        
        Args:
            model_ids: List of model identifiers to compare
            metrics: List of metrics to compare (default: all)
            
        Returns:
            Dictionary with comparison data:
            {
                'models': [...],
                'comparison': {...},
                'winner': {...}
            }
        """
        if metrics is None:
            metrics = ['sharpe_ratio', 'sortino_ratio', 'max_drawdown', 
                      'win_rate', 'profit_factor', 'total_return']
        
        # Load metadata for all models
        models_data = []
        for model_id in model_ids:
            try:
                _, metadata = self.load_model(model_id=model_id)
                models_data.append(metadata)
            except Exception as e:
                print(f"⚠️ Could not load {model_id}: {e}")
                continue
        
        if not models_data:
            return {'error': 'No valid models found'}
        
        # Build comparison
        comparison = {}
        for metric in metrics:
            comparison[metric] = {}
            for model in models_data:
                model_id = model['model_id']
                value = model.get('metrics', {}).get(metric, None)
                comparison[metric][model_id] = value
        
        # Determine winner (best Sharpe ratio)
        winner = max(models_data, key=lambda x: x.get('metrics', {}).get('sharpe_ratio', -999))
        
        return {
            'models': models_data,
            'comparison': comparison,
            'winner': winner['model_id'],
            'winner_sharpe': winner.get('metrics', {}).get('sharpe_ratio', 0)
        }
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full metadata for a specific model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Metadata dictionary or None if not found
        """
        try:
            _, metadata = self.load_model(model_id=model_id)
            return metadata
        except:
            return None
    
    def cleanup_old_models(
        self,
        agent_type: str,
        symbol: str,
        keep_last: int = 10
    ) -> int:
        """
        Archive old models, keeping only the N most recent.
        
        Args:
            agent_type: 'PPO' or 'SAC'
            symbol: Trading symbol
            keep_last: Number of recent models to keep
            
        Returns:
            Number of models archived
        """
        models = self.list_models(agent_type=agent_type, symbol=symbol)
        
        if len(models) <= keep_last:
            print(f"ℹ️ Only {len(models)} models exist, no cleanup needed")
            return 0
        
        # Sort by creation date
        models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        # Archive old models
        models_to_archive = models[keep_last:]
        archived_count = 0
        
        for model in models_to_archive:
            model_id = model['model_id']
            if self.archive_model(model_id):
                archived_count += 1
        
        print(f"✅ Cleanup complete: kept {keep_last} models, archived {archived_count}")
        return archived_count

