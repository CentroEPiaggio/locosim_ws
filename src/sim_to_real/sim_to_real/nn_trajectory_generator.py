import torch


class NNTrajectoryGenerator:

    def __init__(self, model, num_envs=1, num_dof=8, device='cpu', model_device='cuda:0', homing_time=10., no_policy_time=5.,
                 policy_dt=.02, num_obs=36, default_dof_pos=(2.0944, -2.0944, -2.0944, 2.0944, -1.0472, 1.0472, 1.0472, -1.0472),
                 deterministic=True, lin_vel_scale=2., ang_vel_scale=.25, dof_pos_scale=1., dof_vel_scale=.05, action_scale=.5):

        self._model = model

        self._num_envs = num_envs
        self._num_dof = num_dof
        self._device = device
        self._model_device = model_device
        self._homing_time = homing_time
        self._no_policy_time = no_policy_time
        self._policy_dt = policy_dt
        self._num_obs = num_obs
        self._deterministic = deterministic

        self._default_dof_pos = torch.tensor([default_dof_pos] * self._num_envs, dtype=torch.float, device=self._device)
        self._dof_pos_cmd = torch.zeros((self._num_envs, self._num_dof), dtype=torch.float, device=self._device)
        self._obs = torch.zeros((self._num_envs, self._num_obs), dtype=torch.float, device=self._device)

        self._ang_vel_scale = ang_vel_scale
        self._cmd_scale = torch.tensor([lin_vel_scale, ang_vel_scale], dtype=torch.float, device=self._device)
        self._dof_pos_scale = dof_pos_scale
        self._dof_vel_scale = dof_vel_scale
        self._action_scale = action_scale

        self._gravity_vec = torch.tensor([[0., 0., -1.]] * self._num_envs, dtype=torch.float, device=self._device)
        self._orientation = torch.tensor([[1., 0., 0., 0.]] * self._num_envs, dtype=torch.float, device=self._device)
        self._projected_gravity = self._quat_rotate_inverse(self._orientation, self._gravity_vec)
        self._ang_vel = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device)
        self._cmd = torch.zeros((self._num_envs, 2), dtype=torch.float, device=self._device)
        self._prev_dof_pos = torch.zeros((self._num_envs, self._num_dof), dtype=torch.float, device=self._device)
        self._dof_pos = torch.zeros((self._num_envs, self._num_dof), dtype=torch.float, device=self._device)
        self._dof_vel = torch.zeros((self._num_envs, self._num_dof), dtype=torch.float, device=self._device)
        self._actions = torch.zeros((self._num_envs, self._num_dof), dtype=torch.float, device=self._device)

        self._init_state()

    @staticmethod
    @torch.jit.script
    def _quat_rotate_inverse(q, v):
        shape = q.shape
        q_w = q[:, 0]
        q_vec = q[:, 1:]
        a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
        return a - b + c

    def _init_state(self):
        self._dof_pos[:] = self._default_dof_pos

    def _set_cmd(self, t, vx, wz):
        if t < self._homing_time:
            self._cmd[:] = torch.tensor([0., 0.], dtype=torch.float, device=self._device)
        else:
            self._cmd[:] = torch.tensor([vx, wz], dtype=torch.float, device=self._device)

    def _get_obs(self):
        self._obs[:] = torch.cat((self._projected_gravity,
                                  self._orientation,
                                  self._ang_vel * self._ang_vel_scale,
                                  self._cmd * self._cmd_scale,
                                  self._dof_pos * self._dof_pos_scale,
                                  self._dof_vel * self._dof_vel_scale,
                                  self._actions), dim=-1)

    def _run_inference(self):
        input_dict = {'is_train': False,
                      'prev_actions': None,
                      'obs': self._obs.to(self._model_device),
                      'rnn_states': None}

        with torch.no_grad():
            res_dict = self._model(input_dict)

        if self._deterministic:
            self._actions[:] = res_dict['mus']
        else:
            self._actions[:] = res_dict['actions']

    def _set_dof_pos_cmd(self, t):
        if t < self._no_policy_time:
            self._dof_pos_cmd[:] = self._default_dof_pos
        else:
            self._dof_pos_cmd[:] = self._action_scale * self._actions + self._default_dof_pos

    def _refresh_state(self):
        self._ang_vel[:, 2] = self._cmd[:, 1]
        self._prev_dof_pos[:] = self._dof_pos
        self._dof_pos[:] = self._dof_pos_cmd
        self._dof_vel[:] = (self._dof_pos - self._prev_dof_pos) / self._policy_dt

    def get_dof_pos_cmd(self, t, vx=0., wz=0.):
        self._set_cmd(t, vx, wz)
        self._get_obs()
        self._run_inference()
        self._set_dof_pos_cmd(t)
        self._refresh_state()
        return self._dof_pos_cmd.squeeze().clone()
