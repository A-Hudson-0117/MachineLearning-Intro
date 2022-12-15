# -*- coding: utf-8 -*-


import gym
import numpy as np
import matplotlib.pyplot as plt 
import os
import math
import gym
import matplotlib.pyplot as plt
import pygame
import pygame.camera
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt 
import random
import rospy

from torch.distributions import Categorical
from gym import spaces
from itertools import count
from collections import namedtuple
from geometry_msgs.msg import Twist, PoseStamped

rospy.init_node('velocity_publisher')
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1000)
move = Twist() 

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

global pos_zb, pos_zc
def currentPos1(msg):
        global pos_zb
        pos_zb = msg.pose.position

def currentPos2(msg):
        global pos_zc
        pos_zc = msg.pose.position

rospy.Subscriber("/vrpn_client_node/zb/pose", PoseStamped, currentPos1)
rospy.Subscriber("/vrpn_client_node/zc/pose", PoseStamped, currentPos2)
ititial_pose = rospy.wait_for_message("/vrpn_client_node/zb/pose", PoseStamped, rospy.Duration.from_sec(10))
ititial_pose = rospy.wait_for_message("/vrpn_client_node/zc/pose", PoseStamped, rospy.Duration.from_sec(10))


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class DiffDriveEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=4):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window'

        self.max_v = 1
        self.max_w = 2
        self._time_elapsed = 0
        self.high = self.size/2        
        # self.observation_space = spaces.Box(low=-high, high=high)
        self.observation_space = spaces.Dict({
                "agent": spaces.Box(-self.high, self.high, shape=(2,), dtype=np.float32),
                "target": spaces.Box(-self.high, self.high, shape=(2,), dtype=np.float32),
                "iterations": spaces.Box(0, 10000, shape=(2,), dtype=int)
            })
        
        self.action_space = spaces.Discrete(8)
        
        self._action_to_velocity = {
            0: np.array([self.max_v, 0]),
            1: np.array([-self.max_v, 0]),
            2: np.array([self.max_v, self.max_w]),
            3: np.array([self.max_v, -self.max_w]),
            4: np.array([-self.max_v, self.max_w]),
            5: np.array([-self.max_v, -self.max_w]),
            6: np.array([0, self.max_w]),
            7: np.array([0, -self.max_w]),
        }
        
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
    def _get_obs(self):
        # return np.concatenate([self._agent_location[[0,1]], self._target_location], axis=0, dtype=np.float32)
        return {"agent": self._agent_location, "target": self._target_location
                , "time": self._time_elapsed}

    def _get_info(self):
        # return np.linalg.norm(self._agent_location[[0,1]] - self._target_location, ord=1)
        return {"distance": np.linalg.norm(self._agent_location[[0,1]] - self._target_location, ord=1)}
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        high = self.size/2

        xb_pos = pos_zb.x; yb_pos = pos_zb.y
        xc_pos = pos_zc.x; yc_pos = pos_zc.y

        heading = math.atan2(yb_pos -yc_pos,xb_pos -xc_pos)

        # Choose the agent's location uniformly at random
        self._agent_location = np.array([xc_pos, yc_pos, heading],dtype=np.float32)
        # self._agent_location[2] = self.np_random.uniform(low=0, high=0)
        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = np.array([1.5,1],dtype=np.float32)
        # while np.array_equal(self._target_location, self._agent_location[[0,1]]):
        #     self._target_location = self.np_random.uniform(low=-high, high=high,size=2)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def step(self, action):
        h = .1
        rate = rospy.Rate(1/h)

        terminated = False
        reward = 0
        pix_square_size = self.window_size / self.size
        circle_size = pix_square_size / 8

        actionvw = self._action_to_velocity[action]

        move.linear.x = actionvw[0]
        move.angular.z = actionvw[1]
        pub.publish(move)

        print(move)
        rate.sleep()

        distance_from_object_old = np.sqrt((self._agent_location[0]-self._target_location[0])**2
                             +(self._agent_location[1]-self._target_location[1])**2)
        
        xb_pos = pos_zb.x; yb_pos = pos_zb.y
        xc_pos = pos_zc.x; yc_pos = pos_zc.y

        heading = math.atan2(yb_pos -yc_pos,xb_pos -xc_pos)

        self._agent_location[0] = yc_pos
        self._agent_location[1] = yc_pos
        self._agent_location[2] = heading
        
        distance_from_object = np.sqrt((self._agent_location[0]-self._target_location[0])**2
                             +(self._agent_location[1]-self._target_location[1])**2)
        
        desired_heading = math.atan2(self._target_location[1]-self._agent_location[1], 
                                   self._target_location[0]-self._agent_location[0])

        heading_error = desired_heading-self._agent_location[2]
        
        reward = -distance_from_object-2*heading_error**2
        
        if distance_from_object < circle_size/pix_square_size*.5:  
            reward += 100
            terminated = True
            # print("completed")
            
        elif abs(self._agent_location[0]) > self.size/2 or abs(self._agent_location[1] > self.size/2):
            terminated = True
            reward += -50
            
        elif self._time_elapsed > 50:
            terminated = True
            reward += -50
    
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
            
        if terminated:
            self._time_elapsed = 0
        else:
            self._time_elapsed += 1
        

            
        return observation, reward, terminated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels
        
        # First we draw the target
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (np.array([self._target_location[0] + self.size/2,
                       -self._target_location[1] + self.size/2])) * pix_square_size,
            pix_square_size / 8,
        )
    
        circle_size = pix_square_size / 8
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (np.array([self._agent_location[0] + self.size/2,
                       -self._agent_location[1] + self.size/2])) * pix_square_size,
            circle_size,
        )
        # Draw heading 
        pygame.draw.circle(
            canvas,
            (0, 0, 0),
            (np.array([self._agent_location[0]+ (circle_size/pix_square_size*.9)*np.cos(self._agent_location[2]) + self.size/2,
                       -(self._agent_location[1]+ (circle_size/pix_square_size*.9)*np.sin(self._agent_location[2])) + self.size/2])) * pix_square_size,
            circle_size/3,
        )

        # Finally, add some gridlines
        # for x in range(self.size + 1):
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (0, pix_square_size * x),
        #         (self.window_size, pix_square_size * x),
        #         width=3,
        #     )
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (pix_square_size * x, 0),
        #         (pix_square_size * x, self.window_size),
        #         width=3,
        #     )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
