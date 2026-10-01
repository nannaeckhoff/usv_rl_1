# Defines make_env(): builds and returns a configured environment based on
# config.py. Single source of truth for how the environment is constructed.
#
# Two variants are needed:
# - for_training=False (default): uses safety_gymnasium.make(), which returns
#   6 values per step (obs, reward, cost, terminated, truncated, info).
# - for_training=True: uses gymnasium.make() on the "...Gymnasium-v0"
#   variant, which returns the standard 5 values -- required by
#   stable-baselines3.

import gymnasium
import safety_gymnasium
import config

# Sentinel object used to detect "no render_mode argument was passed at
# all", distinct from "render_mode=None was passed explicitly" (which
# means "no rendering", and must NOT fall back to config.RENDER_MODE).
_UNSET = object()


def make_env(render_mode=_UNSET, for_training: bool = False):
    """Create and return the configured environment.

    Args:
        render_mode: "human" to open a live viewer window, None to run
            headless (no window). If omitted entirely, falls back to
            config.RENDER_MODE.
        for_training: if True, returns the standard 5-value Gymnasium API
            (for use with stable-baselines3). If False, returns the
            safety-gymnasium 6-value API including the cost signal.
    """
    mode = config.RENDER_MODE if render_mode is _UNSET else render_mode

    if for_training:
        env = gymnasium.make(config.ENV_ID_TRAINING, render_mode=mode)
    else:
        env = safety_gymnasium.make(config.ENV_ID, render_mode=mode)

    return env