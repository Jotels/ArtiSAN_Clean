import logging
import time
from typing import Dict
import torch
import numpy as np
from torch.optim import Adam
from src.training.loss import compute_loss
from torch import nn
from src.tools.mpi import sync_params, mpi_avg_grads, mpi_avg, set_barrier
import wandb


# Train policy with multiple steps of gradient descent
def train(agent: nn.Module,
          current_system: list,
          optimizer: Adam,
          scheduler: torch.optim.lr_scheduler,
          data: Dict[str, torch.Tensor],
          clip_ratio: float,
          target_kl: float,
          vf_coef: float,
          p_coef: float,
          entropy_coef: float,
          gradient_clip: float,
          max_num_steps: int,
          multithread: bool,
          num_atoms: int,
          device: str,
          ) -> dict:
    infos = {}

    start_time = time.time()

    for i in range(max_num_steps):
        # Compute loss

        loss, loss_info = compute_loss(agent=agent,
                                       current_system=current_system,
                                       data=data,
                                       vf_coef=vf_coef,
                                       p_coef=p_coef,
                                       entropy_coef=entropy_coef,
                                       clip_ratio=clip_ratio,
                                       multithread=multithread,
                                       num_atoms=num_atoms,
                                       device=device)
        kl_ap = loss_info['approx_kl']

        if kl_ap > 1.5 * target_kl:  # Means we stray  too far from a previous policy, thus violating PPO
            # Note that we use the approx_kl from above
            print(f'Early stopping at step {i} due to reaching max KL- KL is {kl_ap}.')
            break
        else:
            print(f'Continuing training - KL is {kl_ap}.')

        # Take gradient step
        optimizer.zero_grad()
        loss.backward()

        if multithread:
            set_barrier()
            mpi_avg_grads(agent)

        # Clip gradients, just to be sure
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=gradient_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            wandb.log({"Learning rate": scheduler.get_last_lr()[0]})

        # Logging
        logging.debug(f'Optimization step {i}: {loss_info}')
        infos.update(loss_info)

    infos['num_opt_steps'] = i
    wandb.log({"Number of optimization steps": i})
    infos['time'] = time.time() - start_time
    return infos
