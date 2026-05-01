from .garnet import GarnetEnv
from .fourrooms import FourRoomsTabular

# Reacher requires gymnasium[mujoco] + mujoco>=3.x.
# Import lazily so the rest of the package works without it.
try:
    from .reacher import ReacherDiscrete
except ImportError:
    pass

# CarGoal is in a separate optional-dependency group (pinned gymnasium==0.26.3).
# Import lazily so the rest of the package still works without it.
try:
    from .cargoal import CarGoalDiscrete
except ImportError:
    pass
