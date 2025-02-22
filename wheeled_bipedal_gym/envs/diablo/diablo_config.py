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

from wheeled_bipedal_gym.envs.base.wheeled_bipedal_config import WheeledBipedalCfg, WheeledBipedalCfgPPO


class DiabloCfg(WheeledBipedalCfg):

    class terrain(WheeledBipedalCfg.terrain):
        mesh_type = "plane"
        # mesh_type = "trimesh"
        # mesh_type = "trimesh"  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 0.8
        dynamic_friction = 0.8
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
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.0, 0.5, 0.5, 0.0, 0.0, 0.0]
        # trimesh only:
        slope_treshold = (
            0.75  # slopes above this threshold will be corrected to vertical surfaces
        )

    class commands(WheeledBipedalCfg.commands):
        curriculum = True
        basic_max_curriculum = 1
        advanced_max_curriculum = 1.5
        curriculum_threshold = 0.7
        num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 3.0  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges(WheeledBipedalCfg.commands.ranges):
            lin_vel_x = [ -0.1, 0.1 ]  # min max [m/s]
            ang_vel_yaw = [-1.5, 1.5]  # min max [rad/s]
            height = [0.14, 0.35]
            heading = [-3.14, 3.14]

    class init_state(WheeledBipedalCfg.init_state):
        pos = [0.0, 0.0, 0.14]  # x,y,z [m]
        default_joint_angles = {  # target angles when action = 0.0
            "left_fake_hip_joint": 0.0,
            "left_fake_knee_joint": 0.0,
            "left_wheel_joint": 0.0,
            "right_fake_hip_joint": 0.0,
            "right_fake_knee_joint": 0.0,
            "right_wheel_joint": 0.0,
        }

    class control(WheeledBipedalCfg.control):
        control_type = "P"  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {"hip": 20.0, "knee": 25.0, "wheel": 0}  # [N*m/rad]
        damping = {"hip": 0.12, "knee": 0.1, "wheel": 0.22}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 5
        pos_action_scale = 0.5
        vel_action_scale = 10.0
        use_feedforward = False
        feedforward_force = 60.0

    class asset(WheeledBipedalCfg.asset):
        file = "{WHEELED_BIPEDAL_GYM_ROOT_DIR}/resources/robots/diablo/urdf/diablo_asm.urdf"
        name = "diablo"
        offset = 0.
        l1 = 0.14
        l2 = 0.14
        penalize_contacts_on = ["shank", "thigh", "diablo_base_link"]
        terminate_after_contacts_on = ["diablo_base_link"]

    class domain_rand(WheeledBipedalCfg.domain_rand):
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
        push_interval_s = 8
        max_push_vel_xy = 2.0
        max_push_vel_z = 0.0
        randomize_Kp = True
        randomize_Kp_range = [0.9, 1.1]
        randomize_Kd = True
        randomize_Kd_range = [0.9, 1.1]
        randomize_motor_torque = True
        randomize_motor_torque_range = [0.9, 1.1]
        randomize_default_dof_pos = True
        randomize_default_dof_pos_range = [-0.02, 0.02]

        # delay
        randomize_action_delay = True
        delay_ms_range = [0, 5]

        add_joint_delay = False
        randomize_joint_delay = False
        randomize_joint_delay_perstep = False
        joint_delay_ms_range = [0, 2]

        add_imu_delay = False
        randomize_imu_delay = False
        randomize_imu_delay_perstep = False
        imu_delay_ms_range = [0, 2]

        action_noise = 0.02
        action_inertia = 0.1

    class rewards(WheeledBipedalCfg.rewards):

        class scales(WheeledBipedalCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_lin_vel_enhance = 1
            tracking_ang_vel = 0.8

            base_height = 1.0
            base_height_enhance = 2
            nominal_state = -0.5
            lin_vel_z = -0.5
            ang_vel_xy = -0.05
            orientation = -200.0

            dof_vel = -5e-5
            dof_acc = -2.5e-7
            torques = -1e-5
            action_rate = -0.03
            action_smooth = -0.03

            collision = -20.0
            dof_pos_limits = -1.0
            dof_vel_limits = -1.0

            theta_limit = -0.01
            same_l = 0.1e-5
            wheel_vel = 0.0
            # termination = -1e5

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        clip_single_reward = 1
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = (
            0.97  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        max_contact_force = 100.0  # forces above this value are penalized

    class normalization(WheeledBipedalCfg.normalization):

        class obs_scales(WheeledBipedalCfg.normalization.obs_scales):
            lin_vel = 10.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025
            height_measurements = 5.0
            torque = 0.05

        clip_observations = 100.0
        clip_actions = 100.0

    class noise(WheeledBipedalCfg.noise):
        add_noise = True
        noise_type = "Uniform"  # Gaussian or Uniform
        noise_level = 1.0  # scales other values

        class noise_scales(WheeledBipedalCfg.noise.noise_scales):
            dof_pos = 0.05
            dof_vel = 1.0
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
class DiabloCfgPPO(WheeledBipedalCfgPPO):
    class runner(WheeledBipedalCfgPPO.runner):
        # logging
        policy_class_name = (
            "ActorCriticSequence"  # could be ActorCritic, ActorCriticSequence
        )
        experiment_name = "diablo"

    class algorithm(WheeledBipedalCfgPPO.algorithm):
        # training params
        extra_learning_rate = 5e-4
        max_grad_norm = 0.1
