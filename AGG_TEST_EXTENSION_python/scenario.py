# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import numpy as np
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid, GroundPlane
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils import distance_metrics
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats, quats_to_rot_matrices
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.motion_generation import ArticulationMotionPolicy, RmpFlow
from omni.isaac.motion_generation.interface_config_loader import load_supported_motion_policy_config
from omni.isaac.nucleus import get_assets_root_path

import omni
import omni.graph.core as og
import omni.kit.commands
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics
import carb
from omni.isaac.sensor import _sensor


class PhotoeyeConveyorScript:
    def __init__(self):

        self._script_generator = None
        self.sensor_1_path = "/World/Sensors/LightBeam_Sensor"
        self.sensor_2_path = "/World/Sensors/LightBeam_Sensor_01"
        self.sensor_3_path = "/World/Sensors/LightBeam_Sensor_02"
        self.conveyor_11_path = "/World/Conveyors/ConveyorTrack_11/ConveyorBeltGraph"

        self._ls = _sensor.acquire_lightbeam_sensor_interface()

    def load_example_assets(self):
        """Load assets onto the stage and return them so they can be registered with the
        core.World.

        This function is called from ui_builder._setup_scene()

        The position in which things are loaded is also the position to which
        they will be returned on reset.
        """

        # Return assets that were added to the stage so that they can be registered with the core.World
        return []

    def setup(self):
        """
        This function is called after assets have been loaded from ui_builder._setup_scenario().
        """
        # Set a camera view that looks good
        set_camera_view(eye=[25, 1.1, 3.5], target=[21.1, -0.64, 1.96], camera_prim_path="/OmniverseKit_Persp")


        result1, sensor1 = omni.kit.commands.execute(
            "IsaacSensorCreateLightBeamSensor",
            path=self.sensor_1_path,
            parent=None,
            min_range=0.02,
            max_range=2.0,
            translation=Gf.Vec3d(0, 0, 0),
            orientation=Gf.Quatd(1, 0, 0, 0),
            forward_axis=Gf.Vec3d(1, 0, 0),
            num_rays=1,
            curtain_length=0.5,
        )

        if not result1:
            carb.log_error("Could not create Light Beam Sensor")
            return

        result2, sensor2 = omni.kit.commands.execute(
            "IsaacSensorCreateLightBeamSensor",
            path=self.sensor_2_path,
            parent=None,
            min_range=0.20,
            max_range=2.0,
            translation=Gf.Vec3d(0, 0, 0),
            orientation=Gf.Quatd(1, 0, 0, 0),
            forward_axis=Gf.Vec3d(1, 0, 0),
            num_rays=1,
            curtain_length=0.5,
        )

        if not result2:
            carb.log_error("Could not create Light Beam Sensor")
            return

        result3, sensor3 = omni.kit.commands.execute(
            "IsaacSensorCreateLightBeamSensor",
            path=self.sensor_3_path,
            parent=None,
            min_range=0.02,
            max_range=2.0,
            translation=Gf.Vec3d(0, 0, 0),
            orientation=Gf.Quatd(1, 0, 0, 0),
            forward_axis=Gf.Vec3d(1, 0, 0),
            num_rays=1,
            curtain_length=0.5,
        )

        if not result3:
            carb.log_error("Could not create Light Beam Sensor")
            return

        (self.action_graph1, new_nodes, _, _) = og.Controller.edit(
            {"graph_path": "/ActionGraph_1", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("IsaacReadLightBeam", "omni.isaac.sensor.IsaacReadLightBeam"),
                    ("DebugDrawRayCast", "omni.isaac.debug_draw.DebugDrawRayCast"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("IsaacReadLightBeam.inputs:lightbeamPrim", self.sensor_1_path),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "IsaacReadLightBeam.inputs:execIn"),
                    ("IsaacReadLightBeam.outputs:execOut", "DebugDrawRayCast.inputs:exec"),
                    ("IsaacReadLightBeam.outputs:beamOrigins", "DebugDrawRayCast.inputs:beamOrigins"),
                    ("IsaacReadLightBeam.outputs:beamEndPoints", "DebugDrawRayCast.inputs:beamEndPoints"),
                    ("IsaacReadLightBeam.outputs:numRays", "DebugDrawRayCast.inputs:numRays"),
                ],
            },
        )

        (self.action_graph2, new_nodes, _, _) = og.Controller.edit(
            {"graph_path": "/ActionGraph_2", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("IsaacReadLightBeam", "omni.isaac.sensor.IsaacReadLightBeam"),
                    ("DebugDrawRayCast", "omni.isaac.debug_draw.DebugDrawRayCast"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("IsaacReadLightBeam.inputs:lightbeamPrim", self.sensor_2_path),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "IsaacReadLightBeam.inputs:execIn"),
                    ("IsaacReadLightBeam.outputs:execOut", "DebugDrawRayCast.inputs:exec"),
                    ("IsaacReadLightBeam.outputs:beamOrigins", "DebugDrawRayCast.inputs:beamOrigins"),
                    ("IsaacReadLightBeam.outputs:beamEndPoints", "DebugDrawRayCast.inputs:beamEndPoints"),
                    ("IsaacReadLightBeam.outputs:numRays", "DebugDrawRayCast.inputs:numRays"),
                ],
            },
        )

        (self.action_graph3, new_nodes, _, _) = og.Controller.edit(
            {"graph_path": "/ActionGraph_3", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("IsaacReadLightBeam", "omni.isaac.sensor.IsaacReadLightBeam"),
                    ("DebugDrawRayCast", "omni.isaac.debug_draw.DebugDrawRayCast"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("IsaacReadLightBeam.inputs:lightbeamPrim", self.sensor_3_path),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "IsaacReadLightBeam.inputs:execIn"),
                    ("IsaacReadLightBeam.outputs:execOut", "DebugDrawRayCast.inputs:exec"),
                    ("IsaacReadLightBeam.outputs:beamOrigins", "DebugDrawRayCast.inputs:beamOrigins"),
                    ("IsaacReadLightBeam.outputs:beamEndPoints", "DebugDrawRayCast.inputs:beamEndPoints"),
                    ("IsaacReadLightBeam.outputs:numRays", "DebugDrawRayCast.inputs:numRays"),
                ],
            },
        )


        self.conveyor_11_graph = og.get_graph_by_path(self.conveyor_11_path)

        og.Controller.edit(self.conveyor_11_graph,
                   {
                og.Controller.Keys.CREATE_NODES: [
                    ("WriteVariable", "omni.graph.core.WriteVariable"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("WriteVariable.inputs:graph", "/World/Conveyors/ConveyorTrack_11/ConveyorBeltGraph"),
                    ("WriteVariable.inputs:variableName", "Velocity"),
                    ("WriteVariable.inputs:value", 2.0)
                ]
                   }     
        )


        og.Controller.connect("/World/Conveyors/ConveyorTrack_11/ConveyorBeltGraph/OnTick.outputs:tick", "/World/Conveyors/ConveyorTrack_11/ConveyorBeltGraph/WriteVariable.inputs:execIn")



        # Create a script generator to execute my_script().
        self._script_generator = self.my_script()

    def reset(self):
        """
        This function is called when the reset button is pressed.
        In this example the core.World takes care of all necessary resetting
        by putting everything back in the position it was in when loaded.

        In more complicated scripts, e.g. scripts that modify or create USD properties
        or attributes at runtime, the user will need to implement necessary resetting
        behavior to ensure their script runs deterministically.
        """
        # Start the script over by recreating the generator.
        self._script_generator = self.my_script()

    """
    The following two functions demonstrate the mechanics of running code in a script-like way
    from a UI-based extension.  This takes advantage of Python's yield/generator framework.  

    The update() function is tied to a physics subscription, which means that it will be called
    one time on every physics step (usually 60 frames per second).  Each time it is called, it
    queries the script generator using next().  This makes the script generator execute until it hits
    a yield().  In this case, no value need be yielded.  This behavior can be nested into subroutines
    using the "yield from" keywords.
    """

    def update(self, step: float):
        try:
            result = next(self._script_generator)
        except StopIteration:
            return True

    def my_script(self):    

        yield from self.check_sensor()




    ################################### Functions

    def check_sensor(self):
        hit_flag = False
        while True:
            print("Checking Sensor")
            print(self._ls.get_beam_hit_data(self.sensor_1_path).astype(bool))
            og.Controller.attribute("/World/Conveyors/ConveyorTrack_11/ConveyorBeltGraph/WriteVariable.inputs:value").set(2.0)

            #existing_value = og.Controller.attribute("/World/Conveyors/ConveyorTrack_11/ConveyorBeltGraph/write_variable.value").get()
            #print(existing_value)
            if self._ls.get_beam_hit_data(self.sensor_1_path).astype(bool) == True:
                print("beam hit!")
                # Assuming you have graph_context available in your script: https://docs.omniverse.nvidia.com/kit/docs/omni.graph/latest/omni.graph.core/omni.graph.core.Graph.html#omni.graph.core.Graph.get_context
                

                if hit_flag == False:
                    initial_time = og.Controller.attribute("/World/Conveyors/ConveyorTrack_11/ConveyorBeltGraph/OnTick.outputs:time").get()
                    print(f"Initial time: {initial_time}")
                    hit_flag = True
                #og.Controller.attribute("/World/Conveyors/ConveyorTrack_11/ConveyorBeltGraph/WriteVariable.inputs:value").set(0.0)

                
            elif hit_flag == True:
                hit_flag = False
                final_time = og.Controller.attribute("/World/Conveyors/ConveyorTrack_11/ConveyorBeltGraph/OnTick.outputs:time").get()
                print(f"Final time: {final_time}")
                belt_velocity = og.Controller.attribute("/World/Conveyors/ConveyorTrack_11/ConveyorBeltGraph/read_speed.outputs:value").get()
                cube_length = (final_time - initial_time)*belt_velocity
                print(f"Cube length: {cube_length}")

            yield ()
