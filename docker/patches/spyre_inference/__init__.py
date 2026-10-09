# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.metadata
import json
import os
from logging.config import dictConfig
from typing import Any

# Defer torch_spyre's autoload until we explicitly trigger it inside
# `TorchSpyreWorker.init_device`. Autoload loads `libspyre_comms.so`,
# which captures `RANK`/`WORLD_SIZE`/`LOCAL_RANK`/`LOCAL_WORLD_SIZE`
# at dlopen time — those env vars are only known per-worker, so the
# library can't load before init_device runs.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

# torch-spyre#3707's pool planner aliases allocations across a fallback boundary
# and corrupts compiled output; our eager RoPE op makes every RoPE call one.
os.environ.setdefault("HBM_POOL_PLANNING", "0")

__version__ = importlib.metadata.version("spyre_inference")


def register():
    """Register the Spyre platform."""
    # Register all VLLM_SPYRE_* environment variables with vLLM env validation
    # so that validate_environ() does not emit Unknown variable warnings for
    # legitimate spyre-inference env vars. Runs before vLLM arg_utils.py calls
    # validate_environ().
    try:
        import os as _os
        import vllm.envs as _vllm_envs
        _SPYRE_ENV_VARS = {
            # Controls compile backend: 'eager' maps to --enforce-eager behaviour;
            # any other value (or unset) uses the default inductor path.
            'VLLM_SPYRE_DYNAMO_BACKEND': lambda: _os.environ.get('VLLM_SPYRE_DYNAMO_BACKEND'),
            # Enable per-step performance metric logging (DEBUG t_token output).
            'VLLM_SPYRE_PERF_METRIC_LOGGING_ENABLED': lambda: _os.environ.get('VLLM_SPYRE_PERF_METRIC_LOGGING_ENABLED'),
        }
        for _k, _v in _SPYRE_ENV_VARS.items():
            if _k not in _vllm_envs.environment_variables:
                _vllm_envs.environment_variables[_k] = _v
    except Exception:
        pass  # Gracefully degrade if vllm.envs layout changes

    # Register Spyre-specific NIXL KV-transfer connectors so that
    # --kv-connector SpyreNixlConnector is accepted by KVConnectorFactory.
    try:
        from spyre_inference.distributed.kv_transfer.kv_connector.v1.nixl.spyre_connector import (
            register_spyre_nixl_connectors,
        )
        register_spyre_nixl_connectors()
    except Exception:
        pass  # Gracefully degrade if vllm KVConnectorFactory layout changes

    return 'spyre_inference.platform.TorchSpyrePlatform'


def register_ops():
    """Register the Spyre OOT custom ops and Transformers backend."""
    from vllm.model_executor.models import ModelRegistry

    from spyre_inference.custom_ops import register_all

    register_all()

    # So that ``model_impl="transformers"`` picks up the Spyre RoPE adaptation.
    ModelRegistry.register_model(
        "TransformersForCausalLM",
        "spyre_inference.transformers_backend:SpyreTransformersForCausalLM",
    )


def _init_logging():
    """Setup logging, extending from the vLLM logging config"""
    from vllm.envs import VLLM_CONFIGURE_LOGGING, VLLM_LOGGING_CONFIG_PATH
    from vllm.logger import DEFAULT_LOGGING_CONFIG, init_logger

    global logger
    logger = init_logger(__name__)

    config: dict[str, Any] = {}

    if VLLM_CONFIGURE_LOGGING:
        config = {**DEFAULT_LOGGING_CONFIG}

    if VLLM_LOGGING_CONFIG_PATH:
        # Error checks must already be done in vllm.logger
        with open(VLLM_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
            config = json.loads(file.read())

    if VLLM_CONFIGURE_LOGGING:
        # Copy the vLLM logging configurations for our package
        if "spyre_inference" not in config["formatters"]:
            if "vllm" in config["formatters"]:
                config["formatters"]["spyre_inference"] = config["formatters"]["vllm"]
            else:
                config["formatters"]["spyre_inference"] = DEFAULT_LOGGING_CONFIG["formatters"][
                    "vllm"
                ]

        if "spyre_inference" not in config["handlers"]:
            if "vllm" in config["handlers"]:
                handler_config = config["handlers"]["vllm"]
            else:
                handler_config = DEFAULT_LOGGING_CONFIG["handlers"]["vllm"]
            handler_config["formatter"] = "spyre_inference"
            config["handlers"]["spyre_inference"] = handler_config

        if "spyre_inference" not in config["loggers"]:
            if "vllm" in config["loggers"]:
                logger_config = config["loggers"]["vllm"]
            else:
                logger_config = DEFAULT_LOGGING_CONFIG["loggers"]["vllm"]
            logger_config["handlers"] = ["spyre_inference"]
            config["loggers"]["spyre_inference"] = logger_config

    dictConfig(config)


_init_logging()
