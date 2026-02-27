from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  alex_v1_flat_env_cfg,
  alex_v1_rough_env_cfg,
)
from .rl_cfg import alex_v1_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Alex-V1",
  env_cfg=alex_v1_rough_env_cfg(),
  play_env_cfg=alex_v1_rough_env_cfg(play=True),
  rl_cfg=alex_v1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Alex-V1",
  env_cfg=alex_v1_flat_env_cfg(),
  play_env_cfg=alex_v1_flat_env_cfg(play=True),
  rl_cfg=alex_v1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
