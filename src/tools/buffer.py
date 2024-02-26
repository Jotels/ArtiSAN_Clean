# The content of this file is based on: OpenAI Spinning Up https://spinningup.openai.com/.
from typing import Optional, List

import numpy as np
from src.reps.environment_hea import HEA
from src.tools import util
import wandb

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, int_act_dim: int, size: int, gamma=0.99, lam=0.95, reward_scaling="hardscaling", en_atom_scaling='atoms') -> None:
        self.obs_buf: List[Optional[HEA]] = [None] * size  # observation buffer
        self.act_buf = np.empty((size, int_act_dim), dtype=np.float32)  # action buffer
        self.rew_buf_scaled = np.zeros(size, dtype=np.float32)  # reward buffer
        self.rew_buf_unscaled = np.zeros(size, dtype=np.float32)  # reward buffer
        self.next_obs_buf: List[Optional[HEA]] = [None] * size  # next observation buffer
        self.term_buf = np.zeros(size, dtype=bool)  # termination buffer

        self.val_buf = np.zeros(size, dtype=np.float32)  # value function buffer
        self.logp_buf = np.zeros(size, dtype=np.float32)  # logarithm of probs buffer
        self.en_buf = np.zeros(size, dtype=np.float32)

        # Filled when path is finished
        self.ret_buf = np.zeros(size, dtype=np.float32)  # return buffer

        self.gamma = gamma  # discount factors
        self.lam = lam  # Lambda for GAE

        self.ptr = 0  # pointer
        self.path_start_idx = 0
        self.max_size = size
        self.reward_scaling = reward_scaling
        self.en_atom_scaling = en_atom_scaling

    def store(self,
              obs: HEA,
              act: np.ndarray,
              reward: float,
              next_obs: HEA,
              en: float,
              terminal: bool,
              value: float,
              logp: float):
        """Append one time step of agent-environment interaction to the buffer."""
        assert self.ptr < self.max_size  # buffer has to have room so you can store

        self.act_buf[self.ptr] = act
        self.obs_buf[self.ptr] = obs
        self.rew_buf_unscaled[self.ptr] = reward
        self.rew_buf_scaled[self.ptr] = reward / obs.num_atoms
        self.next_obs_buf[self.ptr] = next_obs
        self.term_buf[self.ptr] = terminal
        self.en_buf[self.ptr] = en

        self.val_buf[self.ptr] = value
        self.logp_buf[self.ptr] = logp

        self.ptr += 1

    def finish_path(self, last_reward: float, last_system: HEA) -> float:
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf_unscaled[path_slice], last_reward)

        self.ret_buf[path_slice] = util.discount_cumsum(rews, self.gamma)[:-1]

        episodic_return = self.ret_buf[self.path_start_idx]

        self.path_start_idx = self.ptr

        self.last_reward_scaled = last_reward / last_system.num_atoms

        return episodic_return

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.is_full()  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        self.rew_buf_scaled[-1] = self.last_reward_scaled

        return dict(obs=self.obs_buf,
                    act=self.act_buf,
                    ret=self.ret_buf,
                    en=self.en_buf,
                    logp=self.logp_buf,
                    rew=self.rew_buf_scaled,)

    def is_full(self) -> bool:
        return self.ptr == self.max_size
