#!/usr/bin/python3

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image as ROS_Image, CameraInfo
from tf.transformations import euler_from_quaternion, quaternion_from_euler

import tf2_ros
from geometry_msgs.msg import TransformStamped

import argparse

import math
import multiprocessing
import os
import random
import time
from enum import Enum

import numpy as np
from PIL import Image

import habitat_sim
import habitat_sim.agent
from habitat_sim.utils.common import (
    d3_40_colors_rgb,
    quat_from_coeffs,
)

from scipy.spatial.transform import Rotation as R

default_sim_settings = {
    "frame_rate": 30, # image frame rate
    "width": 640, # horizontal resolution
    "height": 360, # vertical resolution
    "hfov": 114.591560981, # horizontal FOV
    "camera_offset_z": 0, # camera z-offset
    "color_sensor": True,  # RGB sensor
    "depth_sensor": True,  # depth sensor
    "semantic_sensor": True,  # semantic sensor
    "scene": "../../vehicle_simulator/mesh/matterport/segmentations/matterport.glb",
}

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=str, default=default_sim_settings["scene"])
args = parser.parse_args()

def make_settings():
    settings = default_sim_settings.copy()
    settings["scene"] = args.scene

    return settings

settings = make_settings()

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.frustum_culling = True
    sim_cfg.gpu_device_id = 0
    if not hasattr(sim_cfg, "scene_id"):
        raise RuntimeError(
            "Error: Please upgrade habitat-sim. SimulatorConfig API version mismatch"
        )
    sim_cfg.scene_id = settings["scene"]

    sensor_specs = []
    def create_camera_spec(**kw_args):
        camera_sensor_spec = habitat_sim.CameraSensorSpec()
        camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        camera_sensor_spec.resolution = [settings["height"], settings["width"]]
        camera_sensor_spec.position = [0, 0, 0]
        for k in kw_args:
            setattr(camera_sensor_spec, k, kw_args[k])
        return camera_sensor_spec

    if settings["color_sensor"]:
        color_sensor_spec = create_camera_spec(
            uuid="color_sensor",
            hfov=settings["hfov"],
            sensor_type=habitat_sim.SensorType.COLOR,
            sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
        )
        sensor_specs.append(color_sensor_spec)

    if settings["depth_sensor"]:
        depth_sensor_spec = create_camera_spec(
            uuid="depth_sensor",
            hfov=settings["hfov"],
            sensor_type=habitat_sim.SensorType.DEPTH,
            channels=1,
            sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
        )
        sensor_specs.append(depth_sensor_spec)

    if settings["semantic_sensor"]:
        semantic_sensor_spec = create_camera_spec(
            uuid="semantic_sensor",
            hfov=settings["hfov"],
            sensor_type=habitat_sim.SensorType.SEMANTIC,
            channels=1,
            sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
        )
        sensor_specs.append(semantic_sensor_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

class DemoRunnerType(Enum):
    BENCHMARK = 1
    EXAMPLE = 2
    AB_TEST = 3

class ABTestGroup(Enum):
    CONTROL = 1
    TEST = 2

class DemoRunner:
    def __init__(self, sim_settings, simulator_demo_type):
        if simulator_demo_type == DemoRunnerType.EXAMPLE:
            self.set_sim_settings(sim_settings)
        self._demo_type = simulator_demo_type

    def set_sim_settings(self, sim_settings):
        self._sim_settings = sim_settings.copy()

    def publish_color_observation(self, obs):
        color_obs = obs["color_sensor"]
        color_img = Image.fromarray(color_obs, mode="RGBA")
        self.color_image.data = np.array(color_img.convert("RGB")).tobytes()
        # print(f"shape: {np.array(color_img).shape}")
        # print(f"after convert RGB: {np.array(color_img.convert('RGB')).shape}")
        self.color_image.header.stamp = rospy.Time.from_sec(self.time)
        self.color_image_pub.publish(self.color_image)
        
        # Publish CameraInfo
        self.color_camera_info.header.stamp = self.color_image.header.stamp
        self.color_camera_info_pub.publish(self.color_camera_info)

    def publish_semantic_observation(self, obs):
        semantic_obs = obs["semantic_sensor"]
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        self.semantic_image.data = np.array(semantic_img.convert("RGB")).tobytes()
        self.semantic_image.header.stamp = rospy.Time.from_sec(self.time)
        self.semantic_image_pub.publish(self.semantic_image)
        
        # Publish CameraInfo
        self.semantic_camera_info.header.stamp = self.semantic_image.header.stamp
        self.semantic_camera_info_pub.publish(self.semantic_camera_info)

    def publish_depth_observation(self, obs):
        depth_obs = obs["depth_sensor"]
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        self.depth_image.data = np.array(depth_img.convert("L")).tobytes()
        self.depth_image.header.stamp = rospy.Time.from_sec(self.time)
        self.depth_image_pub.publish(self.depth_image)
        
        # Publish CameraInfo
        self.depth_camera_info.header.stamp = self.depth_image.header.stamp
        self.depth_camera_info_pub.publish(self.depth_camera_info)

    def init_common(self):
        self._cfg = make_cfg(self._sim_settings)
        scene_file = self._sim_settings["scene"]

        self._sim = habitat_sim.Simulator(self._cfg)

        if not self._sim.pathfinder.is_loaded:
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            self._sim.recompute_navmesh(self._sim.pathfinder, navmesh_settings)

    def state_estimation_callback(self, msg):
        self.time = msg.header.stamp.to_sec()
        orientation = msg.pose.pose.orientation
        (self.camera_roll, self.camera_pitch, self.camera_yaw) = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.camera_x = msg.pose.pose.position.x
        self.camera_y = msg.pose.pose.position.y
        self.camera_z = msg.pose.pose.position.z

    # TODO: this is me added, I think this could be using config... maybe check hydra habitat binding
    def init_camera_info(self, width, height, hfov, frame_id):
        camera_info = CameraInfo()
        camera_info.header.frame_id = frame_id
        camera_info.width = width
        camera_info.height = height
        camera_info.distortion_model = "plumb_bob"
        cx = width / 2.0
        cy = height / 2.0
        fx = fy = width / (2.0 * math.tan(hfov * math.pi / 360.0))
        camera_info.K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        camera_info.P = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
        return camera_info

    def listener(self):
        start_state = self.init_common()

        rospy.init_node("habitat_online")

        # self.br = tf2_ros.TransformBroadcaster()
        
        rospy.Subscriber("/state_estimation", Odometry, self.state_estimation_callback)
        self.time = 0
        self.camera_roll = 0
        self.camera_pitch = 0
        self.camera_yaw = 0
        self.camera_x = 0
        self.camera_y = 0
        self.camera_z = 0.5

        if self._sim_settings["color_sensor"]:
            self.color_image_pub = rospy.Publisher("/habitat_camera/color/image", ROS_Image, queue_size=2)
            self.color_camera_info_pub = rospy.Publisher("/habitat_camera/color/camera_info", CameraInfo, queue_size=2)
            self.color_image = ROS_Image()
            self.color_image.header.frame_id = "habitat_camera"
            self.color_image.height = settings["height"]
            self.color_image.width  = settings["width"]
            self.color_image.encoding = "rgb8"
            self.color_image.step = 3 * self.color_image.width
            self.color_image.is_bigendian = False
            self.color_camera_info = self.init_camera_info(settings["width"], settings["height"], settings["hfov"], "habitat_camera")

        if self._sim_settings["depth_sensor"]:
            self.depth_image_pub = rospy.Publisher("/habitat_camera/depth/image", ROS_Image, queue_size=2)
            self.depth_camera_info_pub = rospy.Publisher("/habitat_camera/depth/camera_info", CameraInfo, queue_size=2)
            self.depth_image = ROS_Image()
            self.depth_image.header.frame_id = "habitat_camera"
            self.depth_image.height = settings["height"]
            self.depth_image.width  = settings["width"]
            self.depth_image.encoding = "mono8"
            self.depth_image.step = self.depth_image.width
            self.depth_image.is_bigendian = False
            self.depth_camera_info = self.init_camera_info(settings["width"], settings["height"], settings["hfov"], "habitat_camera")

        if self._sim_settings["semantic_sensor"]:
            self.semantic_image_pub = rospy.Publisher("/habitat_camera/semantic/image", ROS_Image, queue_size=2)
            self.semantic_camera_info_pub = rospy.Publisher("/habitat_camera/semantic/camera_info", CameraInfo, queue_size=2)
            self.semantic_image = ROS_Image()
            self.semantic_image.header.frame_id = "habitat_camera"
            self.semantic_image.height = settings["height"]
            self.semantic_image.width  = settings["width"]
            self.semantic_image.encoding = "rgb8"
            self.semantic_image.step = 3 * self.semantic_image.width
            self.semantic_image.is_bigendian = False
            self.semantic_camera_info = self.init_camera_info(settings["width"], settings["height"], settings["hfov"], "habitat_camera")

        r = rospy.Rate(default_sim_settings["frame_rate"])
        while not rospy.is_shutdown():
            roll = -self.camera_roll
            pitch = self.camera_pitch
            yaw = 1.5708 - self.camera_yaw

            qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
            qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
            qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
            qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

            position = np.array([self.camera_x, self.camera_y, self.camera_z])
            position[1], position[2] = position[2], -position[1]
            
            agent_state = self._sim.get_agent(0).get_state()
            for sensor in agent_state.sensor_states:
                agent_state.sensor_states[sensor].position = position + np.array([0, default_sim_settings["camera_offset_z"], 0])
                agent_state.sensor_states[sensor].rotation = quat_from_coeffs(np.array([-qy, -qz, qx, qw]))

            self._sim.get_agent(0).set_state(agent_state, infer_sensor_states = False)
            observations = self._sim.step("move_forward")

            if self._sim_settings["color_sensor"]:
                self.publish_color_observation(observations)
            if self._sim_settings["depth_sensor"]:
                self.publish_depth_observation(observations)
            if self._sim_settings["semantic_sensor"]:
                self.publish_semantic_observation(observations)

            state = self._sim.last_state()
            print("Publishing at time: " + str(self.time))
            r.sleep()

        self._sim.close()
        del self._sim

demo_runner = DemoRunner(settings, DemoRunnerType.EXAMPLE)
demo_runner.listener()
