"""Input/Output utility functions for the relativistic ray tracer.

This module handles configuration loading and asset management pipelines.
"""

import os
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads and parses the simulation YAML configuration file.

    Args:
      config_path: The path to the configuration file relative to the project root.

    Returns:
      A nested dictionary containing configuration sections and parameters.

    Raises:
      FileNotFoundError: If the configuration file does not exist at the path.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found at: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
