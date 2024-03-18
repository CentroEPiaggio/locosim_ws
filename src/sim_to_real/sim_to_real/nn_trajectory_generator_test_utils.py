from matplotlib import pyplot as plt
from nn_trajectory_generator import NNTrajectoryGenerator


def test(model):

    nn_traj_gen = NNTrajectoryGenerator(model, homing_time=0., no_policy_time=0.)
    episode_length_sec = 5.
    num_dof = 8
    dof_pos = [[] for _ in range(num_dof)]
    times = []
    policy_dt = 0.02
    decimation = 4
    sim_dt = policy_dt / decimation
    t = 0.

    while t < episode_length_sec:
        dof_pos_cmd = nn_traj_gen.get_dof_pos_cmd(t, 1., 0.)
        times.extend([t + i * sim_dt for i in range(decimation)])
        for i in range(num_dof):
            dof_pos[i].extend([dof_pos_cmd[i]] * 4)
        t += policy_dt

    fig, axes = plt.subplots(4, 2)
    fig.suptitle('Joint Positions')

    for i in range(num_dof):
        axes[i % 4, i // 4].plot(times, dof_pos[i])
        axes[i % 4, i // 4].set_xlim([0., episode_length_sec])

    plt.show()
