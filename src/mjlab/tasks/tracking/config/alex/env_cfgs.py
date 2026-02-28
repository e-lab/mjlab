"""Alex V1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  ALEX_ACTION_SCALE,
  get_alex_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def alex_v1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Alex V1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_alex_robot_cfg()}

  # Alex exposes pelvis-prefixed IMU sensor names.
  cfg.observations["actor"].terms["base_lin_vel"].params["sensor_name"] = (
    "robot/imu_pelvis_linear_velocity"
  )
  cfg.observations["critic"].terms["base_lin_vel"].params["sensor_name"] = (
    "robot/imu_pelvis_linear_velocity"
  )
  cfg.observations["actor"].terms["base_ang_vel"].params["sensor_name"] = (
    "robot/imu_pelvis_gyro"
  )
  cfg.observations["critic"].terms["base_ang_vel"].params["sensor_name"] = (
    "robot/imu_pelvis_gyro"
  )

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = ALEX_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso"
  motion_cmd.body_names = (
    "pelvis_link",
    "left_hip_x",
    "left_thigh",
    "left_shin",
    "left_foot",
    "right_hip_x",
    "right_thigh",
    "right_shin",
    "right_foot",
    "torso",
    "left_shoulder_z",
    "left_elbow_y",
    "left_wrist_z",
    "right_shoulder_z",
    "right_elbow_y",
    "right_wrist_z",
  )

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
    r"^(left|right)_foot_collision$"
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle",
    "right_ankle",
    "left_wrist_z",
    "right_wrist_z",
  )

  cfg.viewer.body_name = "torso"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg
