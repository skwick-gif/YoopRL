"""
Training Module for RL Trading System

Provides core training functionality including:
- Main training workflow
- Progress tracking callbacks
- Data loading and preparation

Author: YoopRL System
Date: November 8, 2025
"""

from importlib import import_module

__all__ = [
	'train_agent',
	'TrainingProgressCallback',
	'WalkForwardWindow',
	'generate_walk_forward_windows',
	'run_walk_forward_evaluation',
	'run_walk_forward_training_pipeline',
]


def __getattr__(name):
	if name in {'train_agent', 'TrainingProgressCallback'}:
		module = import_module('.train', __name__)
		return getattr(module, name)
	if name in {
		'WalkForwardWindow',
		'generate_walk_forward_windows',
		'run_walk_forward_evaluation',
		'run_walk_forward_training_pipeline',
	}:
		module = import_module('.walk_forward', __name__)
		return getattr(module, name)
	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
