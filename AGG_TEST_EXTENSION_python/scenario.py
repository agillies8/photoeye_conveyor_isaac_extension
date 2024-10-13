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
import random
from omni.isaac.core.materials import PhysicsMaterial

#import random


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
                    ("WriteVariable.inputs:value", 1.0)
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
        robot_prim_path = "/World/Kickers/kicker/"

        self.articulation = Articulation(prim_path=robot_prim_path)
        print(f"Articulation joints: {self.articulation.dof_names}")


        sensor_1 = SensorChecker(self._ls, self.sensor_1_path)
        sensor_2 = SensorChecker(self._ls, self.sensor_2_path)
        sensor_3 = SensorChecker(self._ls, self.sensor_3_path)
        spawner = BoxSpawner()

        open_kicker = ArticulationAction(joint_positions=np.array([45.0]), joint_indices=np.array([0]))

        while True:

            sensor_1.check_sensor_once()
            sensor_2.check_sensor_once()
            sensor_3.check_sensor_once()
            #print(sensor_1.cube_length)

            if 0.2< sensor_1.cube_length < 0.6:
                print(f"first gate triggered. Box length: {sensor_1.cube_length}")
                self.articulation.apply_action(open_kicker)
                sensor_1.update_cube_length(0.0)





            spawner.spawn_box_once()

            yield()




    ################################### Functions

class SensorChecker:
    def __init__(self, ls, sensor_1_path):
        self._ls = ls
        self.sensor_1_path = sensor_1_path
        self.hit_flag = False
        self.initial_time = None
        self.cube_length = 0.0

    # This method runs one iteration of the check_sensor loop
    def check_sensor_once(self):
        
        # Get beam hit data
        beam_hit_data = self._ls.get_beam_hit_data(self.sensor_1_path).astype(bool)

        # Set conveyor belt speed
        #og.Controller.attribute("/World/Conveyors/ConveyorTrack_11/ConveyorBeltGraph/WriteVariable.inputs:value").set(2.0)

        # Check if beam hit
        if beam_hit_data == True:
            print("Beam hit!")

            if self.hit_flag == False:
                # Capture the initial time when the beam was first hit
                self.initial_time = og.Controller.attribute("/World/Conveyors/ConveyorTrack_11/ConveyorBeltGraph/OnTick.outputs:time").get()
                self.hit_flag = True
            
            current_time = og.Controller.attribute("/World/Conveyors/ConveyorTrack_11/ConveyorBeltGraph/OnTick.outputs:time").get()
            belt_velocity = og.Controller.attribute("/World/Conveyors/ConveyorTrack_11/ConveyorBeltGraph/read_speed.outputs:value").get()
            current_length = (current_time - self.initial_time) * belt_velocity
            print(f"Measuring:{current_length}")

        elif self.hit_flag == True:
            # Reset the flag when the beam hit ends
            self.hit_flag = False
            # Capture the final time when the beam hit ends
            final_time = og.Controller.attribute("/World/Conveyors/ConveyorTrack_11/ConveyorBeltGraph/OnTick.outputs:time").get()
            
            # Get the belt velocity and calculate cube length
            belt_velocity = og.Controller.attribute("/World/Conveyors/ConveyorTrack_11/ConveyorBeltGraph/read_speed.outputs:value").get()
            self.cube_length = (final_time - self.initial_time) * belt_velocity
            print(f"Cube length: {self.cube_length}")
    
    # This method can simulate continuous monitoring if needed
    def check_sensor_continuously(self):
        while True:
            self.check_sensor_once()
            yield ()  # To maintain generator compatibility

    def update_cube_length(self, new_length):
        self.cube_length = new_length
        print(f"Cube length updated to: {self.cube_length}")

class BoxSpawner:
    def __init__(self):
        # Initialize the box types with dimensions and colors
        self.box_types = [
            {'dims': np.array([0.4, 0.4, 0.4]), 'color': np.array([1.0, 0.0, 0.0])},  # Red Box
            {'dims': np.array([0.4, 0.4, 0.4]), 'color': np.array([0.0, 1.0, 0.0])},  # Green Box
            {'dims': np.array([0.4, 0.4, 0.4]), 'color': np.array([0.0, 0.0, 1.0])}   # Blue Box
        ]
        
        # Initialize the initial time
        self.initial_time = og.Controller.attribute("/World/Conveyors/ConveyorTrack_11/ConveyorBeltGraph/OnTick.outputs:time").get()

        #init box count:
        self.box_count = 1
        self.base_path = "/World/Objects/spawn_point/Cube"

    # Method to run one iteration of the box spawn loop
    def spawn_box_once(self):
        # Get the current time
        current_time = og.Controller.attribute("/World/Conveyors/ConveyorTrack_11/ConveyorBeltGraph/OnTick.outputs:time").get()
        
        # Check if more than 3 seconds have passed since the last box was spawned
        if (current_time - self.initial_time) > 2.0:
            # Update the initial time to the current time
            self.initial_time = og.Controller.attribute("/World/Conveyors/ConveyorTrack_11/ConveyorBeltGraph/OnTick.outputs:time").get()
            
            # Select a random box type from the list
            selected_box = random.choice(self.box_types)
            physics_material="/World/PhysicsMaterial"

            p_path = f"{self.base_path}_{self.box_count}"
            self.box_count +=1

            cube_material = PhysicsMaterial(physics_material)
            # Spawn the dynamic cuboid with the chosen box type's dimensions and color
            prim = DynamicCuboid(prim_path=p_path,
                                    scale=selected_box['dims'], 
                                    color=selected_box['color'], 
                                    mass=1.0,
                                    physics_material=cube_material)
            print(f"Spawned a box with dims: {selected_box['dims']} and color: {selected_box['color']}")

    # Generator-style continuous spawner (optional, if you want to use it as a generator)
    def spawn_box_continuously(self):
        while True:
            self.spawn_box_once()
            yield ()  # To maintain compatibility with generator functions if needed



            
        

        

