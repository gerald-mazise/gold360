import os
from pathlib import Path
from typing import Any, Dict
import yaml


_CONFIG_CACHE: Dict[str, Any] = {}


def _find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
        if (parent.parent / "pyproject.toml").exists():
            return parent.parent
    return Path.cwd()


def load_config(name: str = "default") -> Dict[str, Any]:
    if name in _CONFIG_CACHE:
        return _CONFIG_CACHE[name]

    root = _find_project_root()
    config_path = root / "config" / f"{name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration '{name}' not found at {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    env = os.getenv("GOLD360_ENV", "development")
    env_config_path = root / "config" / f"{name}.{env}.yaml"
    if env_config_path.exists():
        with open(env_config_path) as f:
            env_overrides = yaml.safe_load(f)
        if env_overrides:
            _deep_merge(config, env_overrides)

    _CONFIG_CACHE[name] = config
    return config


def _deep_merge(base: Dict, override: Dict) -> None:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def get_project_root() -> Path:
    return _find_project_root()
