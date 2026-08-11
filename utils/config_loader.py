"""Configuration loading utilities"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from omegaconf import OmegaConf


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file with environment variable substitution

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Use OmegaConf for advanced features
    config = OmegaConf.create(config)

    # Resolve environment variables
    config = OmegaConf.to_container(config, resolve=True)

    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple config dictionaries"""
    merged = OmegaConf.create({})
    for config in configs:
        merged = OmegaConf.merge(merged, config)
    return OmegaConf.to_container(merged, resolve=True)
