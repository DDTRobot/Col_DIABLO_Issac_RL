# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from wheeled_bipedal_gym.envs.diablo.diablo_config import (
    DiabloCfg,
    DiabloCfgPPO,
)


class DiabloVMCCfg(DiabloCfg):

    class env(DiabloCfg.env):
        num_privileged_obs = (DiabloCfg.env.num_observations + 7 * 11 + 3 +
                              6 * 7 + 3 + 3)
        fail_to_terminal_time_s = 1
        episode_length_s = 20

    class terrain(DiabloCfg.terrain):
        mesh_type = "trimesh"
        # mesh_type = "trimesh"  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 1
        dynamic_friction = 0.
        restitution = 0.5
        # rough terrain only:
        measure_heights = True
        measured_points_x = [
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        selected = True  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 0  # starting curriculum state
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 8  # number of terrain rows (levels)
        num_cols = 8  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        # trimesh only:
        slope_treshold = (
            0.75  # slopes above this threshold will be corrected to vertical surfaces
        )

    class rewards(DiabloCfg.rewards):

        class scales:
            tracking_lin_vel = 1.0
            tracking_lin_vel_enhance = 1
            tracking_ang_vel = 1.0

            base_height = 0.5
            nominal_state = -0.1
            lin_vel_z = -0.1e-3
            ang_vel_xy = -0.05
            orientation = -100.0

            dof_vel = -5e-2
            dof_acc = -2.5e-3
            torques = -0.1e-5
            action_rate = -0.01
            action_smooth = -0.01

            collision = -1.0
            dof_pos_limits = -1

            theta_limit = -0.1e-8
            same_l = -0.1e-8
            # special for wheel
            wheel_vel = -0.001

        base_height_target = 0.30

    class init_state(DiabloCfg.init_state):
        pos = [0.0, 0.0, 0.15]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            "left_fake_hip_joint": -0.5,
            "left_fake_knee_joint": 1.0,
            "left_wheel_joint": 0.0,
            "right_fake_hip_joint": -0.5,
            "right_fake_knee_joint": 1.0,
            "right_wheel_joint": 0.0,
        }

    class control(DiabloCfg.control):
        action_scale_theta = 0.5
        action_scale_l0 = 0.1
        action_scale_vel = 10.0

        l0_offset = 0.20
        feedforward_force = 60.0  # [N]

        # kp_theta = 60.0  # [N*m/rad]
        # kd_theta = 10.0  # [N*m*s/rad]
        # kp_l0 = 1000.0  # [N/m]
        # kd_l0 = 50.0  # [N*s/m]

        # real max
        kp_theta = 10.0  # [N*m/rad]
        kd_theta = 1.  # [N*m*s/rad]
        kp_l0 = 300.0  # [N/m]
        kd_l0 = 8.0  # [N*s/m]

        # PD Drive parameters:
        stiffness = {"f0": 0.0, "f1": 0.0, "wheel": 0}  # [N*m/rad]
        damping = {"f0": 0.0, "f1": 0.0, "wheel": 0.8}  # [N*m*s/rad]

    class normalization(DiabloCfg.normalization):

        class obs_scales(DiabloCfg.normalization.obs_scales):
            l0 = 5.0
            l0_dot = 0.25
            # wheel pos should be zero!
            dof_pos = 0.0

    class noise(DiabloCfg.noise):

        class noise_scales(DiabloCfg.noise.noise_scales):
            l0 = 0.02
            l0_dot = 0.1

    class commands(DiabloCfg.commands):

        class ranges:
            lin_vel_x = [-5.0, 5.0]  # min max [m/s]
            ang_vel_yaw = [-3.14, 3.14]  # min max [rad/s]
            height = [0.20, 0.35]
            heading = [-3.14, 3.14]

    class domain_rand(DiabloCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        randomize_base_mass = True
        added_mass_range = [-2.0, 3.0]
        randomize_inertia = True
        randomize_inertia_range = [0.8, 1.2]
        randomize_base_com = True
        rand_com_vec = [0.05, 0.05, 0.05]
        push_robots = True
        push_interval_s = 7
        max_push_vel_xy = 2.0
        randomize_Kp = True
        randomize_Kp_range = [0.8, 1.2]
        randomize_Kd = True
        randomize_Kd_range = [0.8, 1.2]
        randomize_motor_torque = True
        randomize_motor_torque_range = [0.8, 1.2]
        randomize_default_dof_pos = True
        randomize_default_dof_pos_range = [-0.05, 0.05]
        randomize_action_delay = True
        delay_ms_range = [0, 10]


class DiabloVMCCfgPPO(DiabloCfgPPO):

    class algorithm(DiabloCfgPPO.algorithm):
        kl_decay = (DiabloCfgPPO.algorithm.desired_kl -
                    0.002) / DiabloCfgPPO.runner.max_iterations

    class runner(DiabloCfgPPO.runner):
        # logging
        experiment_name = "diablo_vmc"
