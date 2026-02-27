"""Alex V1 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

ALEX_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "alex_V1_description"
  / "mjcf"
  / "alex_v1_humanoid_train.xml"
)
assert ALEX_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  meshes_root = ALEX_XML.parent.parent / "meshes"
  for f in meshes_root.rglob("*"):
    if not f.is_file():
      continue
    rel_path = f.relative_to(meshes_root).as_posix()
    assets[f"{meshdir}/{rel_path}"] = f.read_bytes()
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ALEX_XML))
  spec.assets = get_assets(spec.meshdir)
  # We replace XML motors with built-in position actuators below.
  for actuator in tuple(spec.actuators):
    spec.delete(actuator)
  return spec


##
# Actuator config.
##

# Joints actuated in alex_v1_humanoid_train.xml.
ACTUATED_JOINTS = (
  "spine_z",
  "left_hip_x",
  "left_hip_z",
  "left_hip_y",
  "left_knee_y",
  "right_hip_x",
  "right_hip_z",
  "right_hip_y",
  "right_knee_y",
  "left_shoulder_y",
  "left_shoulder_x",
  "left_elbow_y",
  "right_shoulder_y",
  "right_shoulder_x",
  "right_elbow_y",
)

ALEX_POSITION_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=ACTUATED_JOINTS,
  stiffness=120.0,
  damping=8.0,
  effort_limit=200.0,
)

##
# Keyframe config.
##

ALEX_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 1.05),
  joint_pos={
    "left_hip_y": -0.30,
    "left_knee_y": 0.60,
    "right_hip_y": -0.30,
    "right_knee_y": 0.60,
    "left_shoulder_y": 0.20,
    "right_shoulder_y": 0.20,
    "left_elbow_y": -0.60,
    "right_elbow_y": -0.60,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot_collision$": 1},
  friction={r"^(left|right)_foot_collision$": (0.8,)},
)

##
# Final config.
##

ALEX_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(ALEX_POSITION_ACTUATOR,),
  soft_joint_pos_limit_factor=0.9,
)


def get_alex_robot_cfg() -> EntityCfg:
  """Get a fresh Alex V1 robot configuration instance."""
  return EntityCfg(
    init_state=ALEX_INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=ALEX_ARTICULATION,
  )


ALEX_ACTION_SCALE: dict[str, float] = {}
assert ALEX_POSITION_ACTUATOR.effort_limit is not None
for name in ACTUATED_JOINTS:
  ALEX_ACTION_SCALE[name] = 0.25 * (
    ALEX_POSITION_ACTUATOR.effort_limit / ALEX_POSITION_ACTUATOR.stiffness
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_alex_robot_cfg())

  viewer.launch(robot.spec.compile())
