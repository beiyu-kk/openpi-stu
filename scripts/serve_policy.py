import dataclasses
import enum
import logging
import pathlib
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import download as _download
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str
    # Normalization asset id. If omitted, it is inferred when the checkpoint contains exactly one set of stats.
    asset_id: str | None = None


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def _configured_asset_id(train_config: _config.TrainConfig) -> str | None:
    if train_config.data.assets.asset_id is not None:
        return train_config.data.assets.asset_id
    if train_config.data.repo_id is not tyro.MISSING:
        return train_config.data.repo_id
    return None


def _prepare_checkpoint(checkpoint: Checkpoint) -> tuple[_config.TrainConfig, pathlib.Path]:
    train_config = _config.get_config(checkpoint.config)
    checkpoint_dir = _download.maybe_download(checkpoint.dir)
    asset_id = checkpoint.asset_id or _configured_asset_id(train_config)

    if asset_id is None:
        assets_dir = checkpoint_dir / "assets"
        asset_ids = sorted(
            {path.parent.relative_to(assets_dir).as_posix() for path in assets_dir.rglob("norm_stats.json")}
        )
        if not asset_ids:
            raise FileNotFoundError(
                f"No normalization statistics found under {assets_dir}. "
                "Expected a file at assets/<asset_id>/norm_stats.json."
            )
        if len(asset_ids) > 1:
            raise ValueError(
                f"Multiple normalization asset ids found under {assets_dir}: {asset_ids}. "
                "Select one with --policy.asset-id."
            )
        asset_id = asset_ids[0]
        logging.info("Auto-detected normalization asset id: %s", asset_id)

    data_config = dataclasses.replace(
        train_config.data,
        assets=dataclasses.replace(train_config.data.assets, asset_id=asset_id),
    )
    return dataclasses.replace(train_config, data=data_config), checkpoint_dir


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            train_config, checkpoint_dir = _prepare_checkpoint(args.policy)
            return _policy_config.create_trained_policy(
                train_config, checkpoint_dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
