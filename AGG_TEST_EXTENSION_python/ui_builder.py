# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import omni.timeline
import omni.ui as ui
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.stage import create_new_stage, get_current_stage
from omni.isaac.core.world import World
from omni.isaac.ui.element_wrappers import CollapsableFrame, StateButton
from omni.isaac.ui.element_wrappers.core_connectors import LoadButton, ResetButton
from omni.isaac.ui.ui_utils import get_style
from omni.usd import StageEventType
from pxr import Sdf, UsdLux
import omni.isaac.core.utils.stage as stage_utils

from .scenario import PhotoeyeConveyorScript
from pathlib import Path
from omni.isaac.ui.ui_utils import LABEL_WIDTH, get_style, setup_ui_headers
from omni.isaac.sensor import _sensor
import omni.physx as _physx
import omni

class UIBuilder:
    def __init__(self):
        # Frames are sub-windows that can contain multiple UI elements
        self.frames = []
        # UI elements created using a UIElementWrapper instance
        self.wrapped_ui_elements = []

        # Get access to the timeline to control stop/pause/play programmatically
        self._timeline = omni.timeline.get_timeline_interface()

        # Run initialization for the provided example
        self._on_init()

        self.num_rays = 3
    ###################################################################################
    #           The Functions Below Are Called Automatically By extension.py
    ###################################################################################

    def on_menu_callback(self):
        """Callback for when the UI is opened from the toolbar.
        This is called directly after build_ui().
        """
        pass

    def on_timeline_event(self, event):
        """Callback for Timeline events (Play, Pause, Stop)

        Args:
            event (omni.timeline.TimelineEventType): Event Type
        """
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            # When the user hits the stop button through the UI, they will inevitably discover edge cases where things break
            # For complete robustness, the user should resolve those edge cases here
            # In general, for extensions based off this template, there is no value to having the user click the play/stop
            # button instead of using the Load/Reset/Run buttons provided.
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False

    def on_physics_step(self, step: float):
        """Callback for Physics Step.
        Physics steps only occur when the timeline is playing

        Args:
            step (float): Size of physics step
        """
        pass

    def on_stage_event(self, event):
        """Callback for Stage Events

        Args:
            event (omni.usd.StageEventType): Event Type
        """
        if event.type == int(StageEventType.OPENED):
            # If the user opens a new stage, the extension should completely reset
            self._reset_extension()

    def cleanup(self):
        """
        Called when the stage is closed or the extension is hot reloaded.
        Perform any necessary cleanup such as removing active callback functions
        Buttons imported from omni.isaac.ui.element_wrappers implement a cleanup function that should be called
        """
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()

    def build_ui(self):
        """
        Build a custom UI tool to run your extension.
        This function will be called any time the UI window is closed and reopened.
        """
        world_controls_frame = CollapsableFrame("World Controls", collapsed=False)

        self._ls = _sensor.acquire_lightbeam_sensor_interface()

        self._timeline = omni.timeline.get_timeline_interface()
        self.sub = _physx.get_physx_interface().subscribe_physics_step_events(self._on_update)

        self.beam_hit_labels = []
        self.linear_depth_labels = []
        self.hit_pos_labels = []

        self.colors = [
            0xFFBBBBFF,
            0xFFBBFFBB,
            0xBBFFBBBB,
            0xBBAAEEFF,
            0xAABBFFEE,
            0xFFEEAABB,
            0xFFC8D5D0,
            0xFFC89BD0,
            0xFFAF9BA7,
            0xFFA4B99A,
        ]

        style = {"background_color": 0xFF888888, "color": 0xFF333333, "secondary_color": self.colors[0]}

        with world_controls_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._load_btn = LoadButton(
                    "Load Button", "LOAD", setup_scene_fn=self._setup_scene, setup_post_load_fn=self._setup_scenario
                )
                self._load_btn.set_world_settings(physics_dt=1 / 60.0, rendering_dt=1 / 60.0)
                self.wrapped_ui_elements.append(self._load_btn)

                self._reset_btn = ResetButton(
                    "Reset Button", "RESET", pre_reset_fn=None, post_reset_fn=self._on_post_reset_btn
                )
                self._reset_btn.enabled = False
                self.wrapped_ui_elements.append(self._reset_btn)

        run_scenario_frame = CollapsableFrame("Run Scenario")

        with run_scenario_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._scenario_state_btn = StateButton(
                    "Run Scenario",
                    "RUN",
                    "STOP",
                    on_a_click_fn=self._on_run_scenario_a_text,
                    on_b_click_fn=self._on_run_scenario_b_text,
                    physics_callback_fn=self._update_scenario,
                )
                self._scenario_state_btn.enabled = False
                self.wrapped_ui_elements.append(self._scenario_state_btn)

        #adding sensor frame here:

        sensor_frame = ui.CollapsableFrame(
                        title="Sensor Readings",
                        height=0,
                        collapsed=False,
                        style=get_style(),
                        style_type_name_override="CollapsableFrame",
                        horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                        vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON
                        )

        with sensor_frame:
            with ui.VStack(style=get_style(), spacing=5):
                for i in range(self.num_rays):
                    # Displaying light beam number and data for each light beam
                    with ui.HStack():
                        ui.Label(
                            f"Lightbeam {i+1}",
                            width=LABEL_WIDTH / 2,
                            tooltip="Light beam number",
                            style={"secondary_color": self.colors[i]},
                        )

                        # Displaying beam hit status (initially empty, to be updated in _on_update)
                        self.beam_hit_labels.append(
                            ui.Label(
                                "",
                                width=LABEL_WIDTH / 1.7,
                                tooltip="beam hit t/f",
                                style={"secondary_color": self.colors[i]},
                            )
                        )

                        # Displaying linear depth (initially empty, to be updated in _on_update)
                        self.linear_depth_labels.append(
                            ui.Label(
                                "",
                                width=LABEL_WIDTH * 1.3,
                                tooltip="linear depth in meters",
                                style={"secondary_color": self.colors[i]},
                            )
                        )

                        # Displaying hit position with specific labels for x, y, and z (initially empty, to be updated in _on_update)
                        self.hit_pos_labels.append(
                            ui.Label(
                                "",
                                width=LABEL_WIDTH,
                                tooltip="hit position in meters",
                                style={"secondary_color": self.colors[i]},
                            )
                        )


    ######################################################################################
    # Functions Below This Point Support The Provided Example And Can Be Deleted/Replaced
    ######################################################################################

    #on update is added from lightbeam_sensor example
    def _on_update(self, dt):
        if self._timeline.is_playing():
            lin_depth = [
                self._ls.get_linear_depth_data(self._scenario.sensor_1_path),
                self._ls.get_linear_depth_data(self._scenario.sensor_2_path),
                self._ls.get_linear_depth_data(self._scenario.sensor_3_path)
            ]

            hit_pos = [
                self._ls.get_hit_pos_data(self._scenario.sensor_1_path),
                self._ls.get_hit_pos_data(self._scenario.sensor_2_path),
                self._ls.get_hit_pos_data(self._scenario.sensor_3_path)
            ]

            # cast from uint8 to bool
            beam_hit = [
                self._ls.get_beam_hit_data(self._scenario.sensor_1_path).astype(bool),
                self._ls.get_beam_hit_data(self._scenario.sensor_2_path).astype(bool),
                self._ls.get_beam_hit_data(self._scenario.sensor_3_path).astype(bool)
            ]

            for i in range(self.num_rays):

                # Update UI labels with the new data
                self.beam_hit_labels[i].text = f"beamhit: {beam_hit[i]}"
                self.linear_depth_labels[i].text = f"linearDepth: {lin_depth[i]}"
                #self.hit_pos_labels[
                #    i
                #].text = f"hitPos x: {hit_pos[i][0]}, hitPos y: {hit_pos[i][1]}, hitPos z: {hit_pos[i][2]}"

    def _on_init(self):
        self._articulation = None
        self._cuboid = None
        self._scenario = PhotoeyeConveyorScript()

    def _add_light_to_stage(self):
        """
        A new stage does not have a light by default.  This function creates a spherical light
        """
        sphereLight = UsdLux.SphereLight.Define(get_current_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        XFormPrim(str(sphereLight.GetPath())).set_world_pose([6.5, 0, 12])

    def _setup_scene(self):
        """
        This function is attached to the Load Button as the setup_scene_fn callback.
        On pressing the Load Button, a new instance of World() is created and then this function is called.
        The user should now load their assets onto the stage and add them to the World Scene.
        """
        #create_new_stage()
        #self._add_light_to_stage()


        # Relative path to the assets folder
        relative_path = Path("agg_omni_extensions/photoeye_conveyor_isaac_extension/usd/conveyor_photoeye_box.usd")

        # Convert to absolute path
        absolute_path = relative_path.resolve()

        stage_utils.add_reference_to_stage(
            usd_path="/home/agillies8/agg_omni_extensions/photoeye_conveyor_isaac_extension/usd/conveyor_photoeye_box.usd",
            prim_path="/World"
            )

        loaded_objects = self._scenario.load_example_assets()

        # Add user-loaded objects to the World
        world = World.instance()
        for loaded_object in loaded_objects:
            world.scene.add(loaded_object)

    def _setup_scenario(self):
        """
        This function is attached to the Load Button as the setup_post_load_fn callback.
        The user may assume that their assets have been loaded by their setup_scene_fn callback, that
        their objects are properly initialized, and that the timeline is paused on timestep 0.
        """
        self._scenario.setup()

        # UI management
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True
        self._reset_btn.enabled = True

    def _on_post_reset_btn(self):
        """
        This function is attached to the Reset Button as the post_reset_fn callback.
        The user may assume that their objects are properly initialized, and that the timeline is paused on timestep 0.

        They may also assume that objects that were added to the World.Scene have been moved to their default positions.
        I.e. the cube prim will move back to the position it was in when it was created in self._setup_scene().
        """
        self._scenario.reset()

        # UI management
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True

    def _update_scenario(self, step: float):
        """This function is attached to the Run Scenario StateButton.
        This function was passed in as the physics_callback_fn argument.
        This means that when the a_text "RUN" is pressed, a subscription is made to call this function on every physics step.
        When the b_text "STOP" is pressed, the physics callback is removed.

        This function will repeatedly advance the script in scenario.py until it is finished.

        Args:
            step (float): The dt of the current physics step
        """
        done = self._scenario.update(step)
        if done:
            self._scenario_state_btn.enabled = False

    def _on_run_scenario_a_text(self):
        """
        This function is attached to the Run Scenario StateButton.
        This function was passed in as the on_a_click_fn argument.
        It is called when the StateButton is clicked while saying a_text "RUN".

        This function simply plays the timeline, which means that physics steps will start happening.  After the world is loaded or reset,
        the timeline is paused, which means that no physics steps will occur until the user makes it play either programmatically or
        through the left-hand UI toolbar.
        """
        self._timeline.play()

    def _on_run_scenario_b_text(self):
        """
        This function is attached to the Run Scenario StateButton.
        This function was passed in as the on_b_click_fn argument.
        It is called when the StateButton is clicked while saying a_text "STOP"

        Pausing the timeline on b_text is not strictly necessary for this example to run.
        Clicking "STOP" will cancel the physics subscription that updates the scenario, which means that
        the robot will stop getting new commands and the cube will stop updating without needing to
        pause at all.  The reason that the timeline is paused here is to prevent the robot being carried
        forward by momentum for a few frames after the physics subscription is canceled.  Pausing here makes
        this example prettier, but if curious, the user should observe what happens when this line is removed.
        """
        self._timeline.pause()

    def _reset_extension(self):
        """This is called when the user opens a new stage from self.on_stage_event().
        All state should be reset.
        """
        self._on_init()
        self._reset_ui()

    def _reset_ui(self):
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = False
        self._reset_btn.enabled = False
