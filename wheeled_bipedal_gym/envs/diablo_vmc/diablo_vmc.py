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

from wheeled_bipedal_gym import WHEELED_BIPEDAL_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from wheeled_bipedal_gym import WHEELED_BIPEDAL_GYM_ROOT_DIR
from wheeled_bipedal_gym.envs.base.wheeled_bipedal import WheeledBipedal
from wheeled_bipedal_gym.utils.terrain import Terrain
from wheeled_bipedal_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float,
)
from wheeled_bipedal_gym.utils.helpers import class_to_dict
from .diablo_vmc_config import DiabloVMCCfg


class DiabloVMC(WheeledBipedal):

    def __init__(self, cfg: DiabloVMCCfg, sim_params, physics_engine,
                 sim_device, headless):
        """Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        super().__init__(self.cfg, sim_params, physics_engine, sim_device,
                         headless)

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions,
                                  clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        self.pre_physics_step()
        for _ in range(self.cfg.control.decimation):
            self.leg_post_physics_step()
            self.envs_steps_buf += 1
            self.action_fifo = torch.cat(
                (self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]),
                dim=1)
            self.torques = self._compute_torques(
                self.action_fifo[torch.arange(self.num_envs),
                self.action_delay_idx, :]).view(
                self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques))
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf,
                                                 -clip_obs, clip_obs)

        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history,
        )

    def compute_observations(self):
        """Computes observations"""
        self.obs_buf = self.compute_proprioception_observations()

        if self.cfg.env.num_privileged_obs is not None:
            heights = (
                    torch.clip(
                        self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                        -1,
                        1.0,
                        )
                    * self.obs_scales.height_measurements
            )
            self.privileged_obs_buf = torch.cat(
                (
                    self.base_lin_vel * self.obs_scales.lin_vel,
                    self.obs_buf,
                    self.last_actions[:, :, 0],
                    self.last_actions[:, :, 1],
                    self.dof_acc * self.obs_scales.dof_acc,
                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                    self.dof_vel * self.obs_scales.dof_vel,
                    heights,
                    self.torques * self.obs_scales.torque,
                    (self.base_mass - self.base_mass.mean()).view(self.num_envs, 1),
                    self.base_com,
                    self.default_dof_pos - self.raw_default_dof_pos,
                    self.friction_coef.view(self.num_envs, 1),
                    self.restitution_coef.view(self.num_envs, 1),
                ),
                dim=-1,
            )

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (
                                    2 * torch.rand_like(self.obs_buf) - 1
                            ) * self.noise_scale_vec

        self.obs_history = torch.cat(
            (self.obs_history[:, self.num_obs:], self.obs_buf), dim=-1
        )

    def compute_proprioception_observations(self):
        # note that observation noise need to modified accordingly !!!
        obs_buf = torch.cat(
            (
                # self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                self.theta0 * self.obs_scales.dof_pos,
                self.theta0_dot * self.obs_scales.dof_vel,
                self.L0 * self.obs_scales.l0,
                self.L0_dot * self.obs_scales.l0_dot,
                self.dof_pos[:, [2, 5]] * self.obs_scales.wheel_pos,
                self.dof_vel[:, [2, 5]] * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
        return obs_buf

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        theta0_ref = (torch.cat(
            (
                (actions[:, 0]).unsqueeze(1),
                (actions[:, 3]).unsqueeze(1),
            ),
            axis=1,
        ) * self.cfg.control.action_scale_theta)
        l0_ref = (torch.cat(
            (
                (actions[:, 1]).unsqueeze(1),
                (actions[:, 4]).unsqueeze(1),
            ),
            axis=1,
        ) * self.cfg.control.action_scale_l0) + self.cfg.control.l0_offset
        wheel_vel_ref = (torch.cat(
            (
                (actions[:, 2]).unsqueeze(1),
                (actions[:, 5]).unsqueeze(1),
            ),
            axis=1,
        ) * self.cfg.control.action_scale_vel)

        self.torque_leg = (self.theta_kp * (theta0_ref - self.theta0) -
                           self.theta_kd * self.theta0_dot)
        self.force_leg = self.l0_kp * (l0_ref -
                                       self.L0) - self.l0_kd * self.L0_dot
        self.torque_wheel = self.d_gains[:, [2, 5]] * (wheel_vel_ref -
                                                       self.dof_vel[:, [2, 5]])
        T1, T2 = self.compute_motor_torque(
            self.force_leg +
            self.cfg.control.feedforward_force * torch.cos(self.theta0),self.torque_leg)

        torques = torch.cat(
            (
                T1[:, 0].unsqueeze(1),
                T2[:, 0].unsqueeze(1),
                self.torque_wheel[:, 0].unsqueeze(1),
                T1[:, 1].unsqueeze(1),
                T2[:, 1].unsqueeze(1),
                self.torque_wheel[:, 1].unsqueeze(1),
            ),
            axis=1,
        )

        return torch.clip(torques * self.torques_scale, -self.torque_limits,
                          self.torque_limits)


    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:
                  3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:8] = 0.0  # commands
        noise_vec[
            8:
            10] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[
            10:
            12] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12:14] = noise_scales.l0 * noise_level * self.obs_scales.l0
        noise_vec[
            14:16] = noise_scales.l0_dot * noise_level * self.obs_scales.l0_dot
        noise_vec[
            16:
            18] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[
            18:
            20] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[20:26] = 0.0  # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = (noise_scales.height_measurements *
                                 noise_level *
                                 self.obs_scales.height_measurements)
        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(
            self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[...,
                                                                           0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[...,
                                                                           1]
        self.dof_acc = torch.zeros_like(self.dof_vel)
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx),
                                    device=self.device).repeat(
                                        (self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0],
                                    device=self.device).repeat(
                                        (self.num_envs, 1))
        self.torques = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.torques_scale = torch.ones(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.p_gains = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.d_gains = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.theta_kp = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.theta_kd = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.l0_kp = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.l0_kd = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.base_position = self.root_states[:, :3]
        self.last_base_position = self.base_position.clone()
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands + 1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
                self.obs_scales.height_measurements,
            ],
            device=self.device,
            requires_grad=False,
        )  # TODO change this
        self.command_ranges["lin_vel_x"] = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["lin_vel_x"][:] = torch.tensor(
            self.cfg.commands.ranges.lin_vel_x)
        self.command_ranges["ang_vel_yaw"] = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["ang_vel_yaw"][:] = torch.tensor(
            self.cfg.commands.ranges.ang_vel_yaw)
        self.command_ranges["height"] = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["height"][:] = torch.tensor(
            self.cfg.commands.ranges.height)
        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.base_lin_vel = quat_rotate_inverse(self.base_quat,
                                                self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat,
                                                self.root_states[:, 10:13])
        self.rigid_body_external_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3),
            device=self.device,
            requires_grad=False)
        self.rigid_body_external_torques = torch.zeros(
            (self.num_envs, self.num_bodies, 3),
            device=self.device,
            requires_grad=False)
        self.projected_gravity = quat_rotate_inverse(self.base_quat,
                                                     self.gravity_vec)
        self.action_delay_idx = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        delay_max = np.int64(
            np.ceil(self.cfg.domain_rand.delay_ms_range[1] / 1000 /
                    self.sim_params.dt))
        self.action_fifo = torch.zeros(
            (self.num_envs, delay_max, self.cfg.env.num_actions),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        self.base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) -
                                      self.measured_heights,
                                      dim=1)

        self.L0 = torch.zeros(self.num_envs,
                              2,
                              dtype=torch.float,
                              device=self.device,
                              requires_grad=False)
        self.L0_dot = torch.zeros(self.num_envs,
                                  2,
                                  dtype=torch.float,
                                  device=self.device,
                                  requires_grad=False)
        self.theta0 = torch.zeros(self.num_envs,
                                  2,
                                  dtype=torch.float,
                                  device=self.device,
                                  requires_grad=False)
        self.theta0_dot = torch.zeros(self.num_envs,
                                      2,
                                      dtype=torch.float,
                                      device=self.device,
                                      requires_grad=False)
        self.theta1 = torch.zeros(self.num_envs,
                                  2,
                                  dtype=torch.float,
                                  device=self.device,
                                  requires_grad=False)
        self.theta2 = torch.zeros(self.num_envs,
                                  2,
                                  dtype=torch.float,
                                  device=self.device,
                                  requires_grad=False)

        # joint positions offsets and PD gains
        self.raw_default_dof_pos = torch.zeros(
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.default_dof_pos = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.raw_default_dof_pos[i] = angle
            self.default_dof_pos[:, i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[:, i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[:, i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[:, i] = 0.0
                self.d_gains[:, i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(
                        f"PD gain of joint {name} were not defined, setting them to zero"
                    )
        self.theta_kp[:] = self.cfg.control.kp_theta
        self.theta_kd[:] = self.cfg.control.kd_theta
        self.l0_kp[:] = self.cfg.control.kp_l0
        self.l0_kd[:] = self.cfg.control.kd_l0
        if self.cfg.domain_rand.randomize_Kp:
            (
                p_gains_scale_min,
                p_gains_scale_max,
            ) = self.cfg.domain_rand.randomize_Kp_range
            self.p_gains *= torch_rand_float(
                p_gains_scale_min,
                p_gains_scale_max,
                self.p_gains.shape,
                device=self.device,
            )
            self.theta_kp *= torch_rand_float(
                p_gains_scale_min,
                p_gains_scale_max,
                self.theta_kp.shape,
                device=self.device,
            )
            self.l0_kp *= torch_rand_float(
                p_gains_scale_min,
                p_gains_scale_max,
                self.l0_kp.shape,
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_Kd:
            (
                d_gains_scale_min,
                d_gains_scale_max,
            ) = self.cfg.domain_rand.randomize_Kd_range
            self.d_gains *= torch_rand_float(
                d_gains_scale_min,
                d_gains_scale_max,
                self.d_gains.shape,
                device=self.device,
            )
            self.theta_kd *= torch_rand_float(
                d_gains_scale_min,
                d_gains_scale_max,
                self.theta_kd.shape,
                device=self.device,
            )
            self.l0_kd *= torch_rand_float(
                d_gains_scale_min,
                d_gains_scale_max,
                self.l0_kd.shape,
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_motor_torque:
            (
                torque_scale_min,
                torque_scale_max,
            ) = self.cfg.domain_rand.randomize_motor_torque_range
            self.torques_scale *= torch_rand_float(
                torque_scale_min,
                torque_scale_max,
                self.torques_scale.shape,
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_default_dof_pos:
            self.default_dof_pos += torch_rand_float(
                self.cfg.domain_rand.randomize_default_dof_pos_range[0],
                self.cfg.domain_rand.randomize_default_dof_pos_range[1],
                (self.num_envs, self.num_dof),
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_action_delay:
            action_delay_idx = torch.round(
                torch_rand_float(
                    self.cfg.domain_rand.delay_ms_range[0] / 1000 /
                    self.sim_params.dt,
                    self.cfg.domain_rand.delay_ms_range[1] / 1000 /
                    self.sim_params.dt,
                    (self.num_envs, 1),
                    device=self.device,
                )).squeeze(-1)
            self.action_delay_idx = action_delay_idx.long()

    # ------------ reward functions----------------
    def _reward_theta_limit(self):
        # Penalize theta is too huge
        return torch.sum(torch.square(self.theta0[:, :2]), dim=1)

    def _reward_same_l(self):
        # Penalize l is too dif
        return torch.square(self.L0[:, 0] - self.L0[:, 1])

    def _reward_wheel_vel(self):
        # Penalize dof velocities
        # left_wheel_vel = self.commands[:,0]/2 - self.commands[:,1]
        # right_wheel_vel = self.commands[:,0]/2 + self.commands[:,1]
        # return torch.sum(torch.square(self.dof_vel[:, 2] - left_wheel_vel) + torch.square(self.dof_vel[:, 5]) - right_wheel_vel)
        return torch.sum(
            torch.square(self.dof_vel[:, 2]) +
            torch.square(self.dof_vel[:, 5]))

    def _reward_static_action_rate(self):
        # When the order remains unchanged, the punishment action rate is higher
        return torch.square(self.L0[:, 0] - self.L0[:, 1])
