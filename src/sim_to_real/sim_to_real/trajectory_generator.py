import torch
from matplotlib import pyplot as plt


class TrajectoryGenerator:

    def __init__(self, num_envs=1, num_dof=8, device='cpu', episode_length_s=5, task='stand_up', hz=1, n=4):

        """
        :param num_envs: The number of robots, for the real robot must be 1
        :param num_dof: The number of actuated joints of the robot, for Mulinex must be 8
        :param device: Whether to use cpu or gpu, for the real robot use cpu
        :param episode_length_s: The length of the experiment
        :param task: The name of the task, one of ['stand_up', 'push_ups', 'pitching', 'rolling', 'trotting_in_place']
        :param hz: Frequency pitching(1), rolling(1) and trotting_in_place(5) (suggested values)
        :param n: Used for task=push_ups only, the number of push-ups.
        """

        self._tasks = ['stand_up', 'push_ups', 'pitching', 'rolling', 'trotting_in_place']
        if task not in self._tasks:
            raise ValueError(f'Unknown task "{task}"! Task value must be in {self._tasks}')

        self._num_envs = num_envs
        self._num_dof = num_dof
        self._device = device
        self._episode_length_s = episode_length_s
        self._task = task
        self._hz = torch.tensor(hz)
        self._n = torch.tensor(n)

        self._a1 = 0.19
        self._a2 = 0.19

        if task in ['stand_up', 'push_ups']:
            # default_dof_pos_REST
            self._default_dof_pos = torch.tensor([[3.141592653589793, -3.141592653589793, -3.141592653589793, 3.141592653589793, -3.141592653589793, 3.141592653589793, 3.141592653589793, -3.141592653589793]] * self._num_envs, dtype=torch.float, device=self._device)
        else:
            # DEFAULT_dof_pos
            self._default_dof_pos = torch.tensor([[2.0944, -2.0944, -2.0944, 2.0944, -1.0472, 1.0472, 1.0472, -1.0472]] * self._num_envs, dtype=torch.float, device=self._device)

        self._dof_pos_cmd = torch.zeros((self._num_envs, self._num_dof), dtype=torch.float, device=self._device)

        self._x = torch.zeros((self._num_envs, 4), dtype=torch.float, device=self._device)
        self._z = torch.zeros((self._num_envs, 4), dtype=torch.float, device=self._device)

        self._q = torch.zeros((self._num_envs, self._num_dof), dtype=torch.float, device=self._device)
        self._prev_q = torch.zeros((self._num_envs, self._num_dof), dtype=torch.float, device=self._device)
        self._q_out_cs = torch.zeros((self._num_envs, self._num_dof), dtype=torch.bool, device=self._device)

    def get_dof_pos_cmd(self, t):
        if self._task == 'stand_up':
            self._set_dof_pos_cmd_for_stand_up(t)  # Requires "default_dof_pos_REST" and base_init_pos_z = 0.04
        if self._task == 'push_ups':
            self._set_dof_pos_cmd_for_n_push_ups(self._n, t)  # Requires "default_dof_pos_REST" and base_init_pos_z = 0.04
        if self._task == 'pitching':
            self._set_dof_pos_cmd_for_pitching_at_hz(self._hz, t)  # Requires "DEFAULT_dof_pos" and base_init_pos_z = 0.33
        if self._task == 'rolling':
            self._set_dof_pos_cmd_for_rolling_at_hz(self._hz, t)  # Requires "DEFAULT_dof_pos" and base_init_pos_z = 0.33
        if self._task == 'trotting_in_place':
            self._set_dof_pos_cmd_for_trotting_in_place_at_hz(self._hz, t)  # Requires "DEFAULT_dof_pos" and base_init_pos_z = 0.33
        return self._dof_pos_cmd.squeeze().clone()

    def _set_dof_pos_cmd_for_stand_up(self, t):
        if t > 0:
            lf_foot_z, lh_foot_z, rf_foot_z, rh_foot_z = [- 0.33 / self._episode_length_s * t] * 4
            self._dof_pos_cmd[:] = self._get_dof_pos_cmd_for_feet_pos(lf_foot_z=lf_foot_z, lh_foot_z=lh_foot_z, rf_foot_z=rf_foot_z, rh_foot_z=rh_foot_z)
        else:
            self._dof_pos_cmd[:] = self._default_dof_pos

    def _set_dof_pos_cmd_for_n_push_ups(self, n, t):
        if t not in [i * self._episode_length_s / n for i in range(n)]:
            lf_foot_z, lh_foot_z, rf_foot_z, rh_foot_z = [- 0.33 * torch.sin(2 * torch.pi * n / 10 * t).abs()] * 4
            self._dof_pos_cmd[:] = self._get_dof_pos_cmd_for_feet_pos(lf_foot_z=lf_foot_z, lh_foot_z=lh_foot_z, rf_foot_z=rf_foot_z, rh_foot_z=rh_foot_z)
        else:
            self._dof_pos_cmd[:] = self._default_dof_pos

    def _set_dof_pos_cmd_for_pitching_at_hz(self, hz, t):
        lf_foot_z, rf_foot_z = [- 0.33 + 0.03 * torch.sin(2 * torch.pi * hz * t)] * 2
        lh_foot_z, rh_foot_z = [- 0.33 - 0.03 * torch.sin(2 * torch.pi * hz * t)] * 2
        self._dof_pos_cmd[:] = self._get_dof_pos_cmd_for_feet_pos(lf_foot_z=lf_foot_z, lh_foot_z=lh_foot_z, rf_foot_z=rf_foot_z, rh_foot_z=rh_foot_z)

    def _set_dof_pos_cmd_for_rolling_at_hz(self, hz, t):
        lf_foot_z, lh_foot_z = [- 0.33 + 0.02 * torch.sin(2 * torch.pi * hz * t)] * 2
        rf_foot_z, rh_foot_z = [- 0.33 - 0.02 * torch.sin(2 * torch.pi * hz * t)] * 2
        self._dof_pos_cmd[:] = self._get_dof_pos_cmd_for_feet_pos(lf_foot_z=lf_foot_z, lh_foot_z=lh_foot_z, rf_foot_z=rf_foot_z, rh_foot_z=rh_foot_z)

    def _set_dof_pos_cmd_for_trotting_in_place_at_hz(self, hz, t):
        sin_1 = 0.05 * torch.sin(2 * torch.pi * hz * t)
        sin_2 = 0.05 * torch.sin(2 * torch.pi * hz * t + torch.pi)
        sin_1 *= sin_1 > 0
        sin_2 *= sin_2 > 0
        lf_foot_z, rh_foot_z = [- 0.33 + sin_1] * 2
        rf_foot_z, lh_foot_z = [- 0.33 + sin_2] * 2
        self._dof_pos_cmd[:] = self._get_dof_pos_cmd_for_feet_pos(lf_foot_z=lf_foot_z, lh_foot_z=lh_foot_z, rf_foot_z=rf_foot_z, rh_foot_z=rh_foot_z)

    def _inv_kin(self):

        self._q[:, -4:] = torch.acos((torch.square(self._x) + torch.square(self._z) - self._a1 ** 2 - self._a2 ** 2) / (2 * self._a1 * self._a2))

        self._q[:, 0] = torch.atan2(- self._z[:, 0], self._x[:, 0]) + torch.atan2(self._a2 * torch.sin(self._q[:, 4]), self._a1 + self._a2 * torch.cos(self._q[:, 4]))
        self._q[:, 1] = torch.atan2(self._z[:, 1], - self._x[:, 1]) - torch.atan2(self._a2 * torch.sin(self._q[:, 5]), self._a1 + self._a2 * torch.cos(self._q[:, 5]))
        self._q[:, 2] = torch.atan2(self._z[:, 2], self._x[:, 2]) - torch.atan2(self._a2 * torch.sin(self._q[:, 6]), self._a1 + self._a2 * torch.cos(self._q[:, 6]))
        self._q[:, 3] = torch.atan2(- self._z[:, 3], - self._x[:, 3]) + torch.atan2(self._a2 * torch.sin(self._q[:, 7]), self._a1 + self._a2 * torch.cos(self._q[:, 7]))

        self._q[:, [4, 7]] *= -1.

        self._q[self._q > torch.pi] -= 2 * torch.pi
        self._q[self._q < - torch.pi] += 2 * torch.pi

        self._q_out_cs[:] = (torch.sqrt(torch.square(self._x) + torch.square(self._z)) > (self._a1 + self._a2)).repeat(1, self._num_dof // 4)
        self._q[self._q_out_cs] = self._prev_q[self._q_out_cs]

        self._prev_q[:] = self._q

    def _get_dof_pos_cmd_for_feet_pos(self, *, lf_foot_x=0, lh_foot_x=0, rf_foot_x=0, rh_foot_x=0, lf_foot_z=0, lh_foot_z=0, rf_foot_z=0, rh_foot_z=0):
        self._x[:, 0] = lf_foot_x
        self._x[:, 1] = lh_foot_x
        self._x[:, 2] = rf_foot_x
        self._x[:, 3] = rh_foot_x
        self._z[:, 0] = lf_foot_z
        self._z[:, 1] = lh_foot_z
        self._z[:, 2] = rf_foot_z
        self._z[:, 3] = rh_foot_z
        self._inv_kin()
        return self._q


# if __name__ == '__main__':

#     episode_length_sec = 5
#     robot_num_dof = 8
#     trajectory_generator = TrajectoryGenerator(num_dof=robot_num_dof, episode_length_s=episode_length_sec, task='trotting_in_place', hz=5)

#     dof_pos = [[] for _ in range(robot_num_dof)]
#     times = []

#     dt = 0.005
#     t_ = 0
#     while t_ <= episode_length_sec:
#         dof_pos_cmd = trajectory_generator.get_dof_pos_cmd(t_)
#         times.append(t_)
#         for i in range(robot_num_dof):
#             dof_pos[i].append(dof_pos_cmd[i])
#         t_ += dt

#     fig, axes = plt.subplots(4, 2)
#     fig.suptitle('Joint Positions')

#     for i in range(robot_num_dof):
#         axes[i % 4, i // 4].plot(times, dof_pos[i])
#         axes[i % 4, i // 4].set_xlim([0., episode_length_sec])

#     plt.show()
