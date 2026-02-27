"""
ConfigParser — YAML → SimpleNamespace Loader
================================================
Loads YAML configuration files and converts them into
nested SimpleNamespace objects for attribute-style access
(e.g. config.model.d_model instead of config['model']['d_model']).
"""

import yaml
from types import SimpleNamespace
from typing import Any, Union


class ConfigParser:
    """Loads YAML configs into nested SimpleNamespace objects."""

    @staticmethod
    def load_and_build_config(config_path: str) -> SimpleNamespace:
        """
        Load a YAML file and return a nested SimpleNamespace.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            SimpleNamespace with attribute-style access to all config values.
        """
        with open(config_path, 'r') as f:
            raw = yaml.safe_load(f)

        return ConfigParser._dict_to_namespace(raw)

    @staticmethod
    def _dict_to_namespace(d: Any) -> Any:
        """Recursively convert dicts to SimpleNamespace."""
        if isinstance(d, dict):
            converted = {}
            for k, v in d.items():
                converted[k] = ConfigParser._dict_to_namespace(v)
            return SimpleNamespace(**converted)
        elif isinstance(d, list):
            return [ConfigParser._dict_to_namespace(item) for item in d]
        else:
            return d

    @staticmethod
    def merge_configs(base: SimpleNamespace, override: SimpleNamespace
                      ) -> SimpleNamespace:
        """Merge override config on top of base config."""
        base_dict = ConfigParser._namespace_to_dict(base)
        override_dict = ConfigParser._namespace_to_dict(override)
        merged = ConfigParser._deep_merge(base_dict, override_dict)
        return ConfigParser._dict_to_namespace(merged)

    @staticmethod
    def _namespace_to_dict(ns: Any) -> Any:
        """Convert SimpleNamespace back to dict."""
        if isinstance(ns, SimpleNamespace):
            return {k: ConfigParser._namespace_to_dict(v)
                    for k, v in vars(ns).items()}
        elif isinstance(ns, list):
            return [ConfigParser._namespace_to_dict(item) for item in ns]
        return ns

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Deep merge two dicts (override wins on conflicts)."""
        merged = base.copy()
        for k, v in override.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = ConfigParser._deep_merge(merged[k], v)
            else:
                merged[k] = v
        return merged
