import time
from typing import Optional
import numpy as np

import wandb
from src.tools.buffer import PPOBuffer
from src.reps.environment_hea import HEA
from src.tools.util import to_numpy
from torch import nn
from ase.io import write
from ase.io.trajectory import TrajectoryWriter


def rollout(agent: nn.Module,  # An ArtiSAN
            env: HEA,
            buffer: PPOBuffer,
            training: bool,
            num_steps: Optional[int] = None,
            num_episodes: Optional[int] = None,
            evaluation: bool = False,
            simulate_traj: bool = False,
            simulate_traj_path: str = None,
            entropy_traj: bool = False,
            ):
    global direct_writer
    assert num_steps or num_episodes

    num_steps = num_steps if num_steps is not None else np.inf
    num_episodes = num_episodes if num_episodes is not None else np.inf

    if evaluation:
        obs = env
    else:
        obs = env.reset()

    ep_returns = []
    ep_lengths = []

    ep_length = 0
    ep_counter = 0
    step = 0
    if evaluation:
        energy_trajectory = np.zeros(env.horizon + 1)
        entropy_trajectory = np.zeros(env.horizon + 1)
    if simulate_traj:
        direct_writer = TrajectoryWriter(filename=simulate_traj_path,
                                                 mode="a",
                                                 atoms=obs.current_atoms)
    while step < num_steps and ep_counter < num_episodes:
        pred = agent.step(obs=[obs], training=training)

        a = to_numpy(pred['a'][0])  # Get action

        # Copy obs before action is taken to be stored in buffer
        obs_buf = obs.copy()
        if evaluation:
            energy_trajectory[step] = env.prior_energy
            if env.num_swaps % 30 == 0:
                current_entropy = env.get_eom_proxy()
            elif env.num_swaps == (env.horizon - 1):
                current_entropy = env.get_eom_proxy()
            entropy_trajectory[step] = current_entropy

        state_prior_energy = float(obs.prior_energy)

        next_obs, reward, done, post_swap_energy, post_swap_entropy = env.environment_step(focus_idx=a[0], swap_idx=a[1])

        print(f"SWITCH {env.num_swaps} |  {env.current_atoms.symbols[a[1]]} at {a[0]} <---> {env.current_atoms.symbols[a[0]]} at {a[1]} | Energy change: {-reward} eV")

        if simulate_traj:
            direct_writer.write(obs.current_atoms, energy=obs.prior_energy)

        if done:
            print("Horizon reached")

        buffer.store(obs=obs_buf,
                     act=a,
                     reward=reward,
                     en=state_prior_energy,
                     next_obs=next_obs,
                     terminal=done,
                     value=pred['v'].item(),
                     logp=pred['logp'].item())

        obs = next_obs

        step += 1
        ep_length += 1

        last_step = step == num_steps - 1
        if done or last_step:
            # if trajectory didn't reach terminal state, bootstrap value target of next observation
            if last_step:
                if next_obs.terminal_only:
                    next_obs.set_calc()
                    last_reward = next_obs.init_state_energy - next_obs.current_atoms.get_potential_energy()
                else:
                    last_reward = reward
            else:
                if next_obs.terminal_only:
                    last_reward = 0
                else:
                    last_reward = reward

            ep_return = buffer.finish_path(last_reward, obs)

            if done:
                ep_returns.append(ep_return)
                ep_lengths.append(ep_length)
                ep_counter += 1
                if training:
                    wandb.log({"Training system energy change for {} system".format(env.unit_cells): -ep_return})
                    print("Training system energy change for {} system: {}".format(env.unit_cells, -ep_return))
            print(f"FINISHED ROLLOUT {ep_counter}")
            print(
                "................................................................................................................................................................................................................................................")
            if training:
                obs = env.reset()
            ep_length = 0
    if evaluation:
        if entropy_traj:
            return energy_trajectory, entropy_trajectory
        else:
            return energy_trajectory
