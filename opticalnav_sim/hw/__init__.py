"""Hardware in the simulator: robots, cameras and sensors defined in Python and attached to one another.

    from opticalnav_sim import hw
    world = hw.World()
    robot = world.spawn(hw.robots.RangerMini2(), at=hw.pose2d(1.0, 2.0))
    cam = world.spawn(hw.cameras.LucidTDR054S(), attach_to=robot, link="top_plate",
                      mount=hw.Transform((0.2, 0.0, 0.1), (0.0, 8.0, 0.0)))

See docs/hardware_platform_plan.md.
"""
from . import cameras, optics, robots
from .actor import Actor, Spec, World
from .robots import RangerControl, Twist
from .sim import Frame, SimWorld
from .transform import Transform, camera_to_world, pose2d, to_mitsuba

__all__ = ["Actor", "Frame", "RangerControl", "SimWorld", "Spec", "Transform", "Twist", "World", "camera_to_world", "cameras", "optics",
           "pose2d", "robots", "to_mitsuba"]
