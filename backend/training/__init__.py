"""
Training Module for RL Trading System

Provides core training functionality including:
- Main training workflow
- Progress tracking callbacks
- Data loading and preparation

Author: YoopRL System
Date: November 8, 2025
"""

from .train import train_agent, TrainingProgressCallback

__all__ = ['train_agent', 'TrainingProgressCallback']
