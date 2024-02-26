from typing import Tuple
import torch
import wandb
from torch import nn
from src.tools.util import to_numpy
from src.tools.mpi import mpi_avg


def compute_loss(agent: nn.Module,
                 current_system: list,
                 data: dict,
                 vf_coef: float,
                 p_coef: float,
                 entropy_coef: float,
                 clip_ratio: float,
                 multithread: bool,
                 num_atoms: int,
                 device) -> Tuple[torch.Tensor, dict]:
    print("Computing loss..")

    """Compute the loss for a batch of data."""

    pred = agent.step(obs=data['obs'],
                      actions=data['act'])

    old_logp = torch.as_tensor(data['logp'], device=device)  # Get the logp from previous step

    en = torch.as_tensor(data['en'], device=device)
    rew = torch.as_tensor(data['rew'], device=device)

    # Policy loss
    ratio = torch.exp(pred['logp'] - old_logp)  # This is the same as saying p/old_p
    obj = ratio * rew  # Objective defined as the p-ratio times performance measure
    clipped_obj = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * rew  # Clipped objective as defined in paper
    policy_loss = - (num_atoms/108) * p_coef * torch.min(obj, clipped_obj).mean()  # The PPO policy loss is then the smallest between the real objective
    # and the clipped objective

    # Entropy loss
    entropy_loss = -entropy_coef * pred['ent'].mean()

    # Value loss
    vf_loss = vf_coef * (pred['v'] + en).pow(2).mean()  # <-- Reverse sign because pred['v'] is the negative predicted energy

    # Total loss
    loss = policy_loss + vf_loss + entropy_loss

    # Approximate KL for early stopping
    approx_kl = (old_logp - pred['logp']).abs().mean()  # KL divergence approximation formula
    wandb.log({"KL": approx_kl})

    # NOTE: CAN BE NEGATIVE - SEE EXPLANATION Below from https://github.com/openai/spinningup/issues/137
    # In the event that a training step results in a policy that is more likely to pick a previously executed action,
    # the approximate KL will be close to zero or slightly negative (the negative may sometimes be a result of floating point calculation, but this is irrelevant).
    # My note: Thus, we change to absolute approx KL

    # Extra info
    clipped = ratio.lt(1 - clip_ratio) | ratio.gt(1 + clip_ratio)
    clip_fraction = torch.as_tensor(clipped, dtype=torch.float32).mean()

    wandb.log({"Policy loss - {} systems".format(current_system): policy_loss,
               "Value loss - {} systems".format(current_system): vf_loss,
               "Entropy loss - {} systems".format(current_system): entropy_loss,
               "Total loss - {} systems".format(current_system): loss})

    print("Total loss for {} systems is {}".format(current_system, loss))

    if multithread:
        info = dict(  # Provides info about the full step, for printing
            policy_loss=mpi_avg(to_numpy(policy_loss)).item(),
            entropy_loss=mpi_avg(to_numpy(entropy_loss)).item(),
            vf_loss=mpi_avg(to_numpy(vf_loss)).item(),
            total_loss=mpi_avg(to_numpy(loss)).item(),
            approx_kl=mpi_avg(to_numpy(approx_kl)).item(),
            clip_fraction=mpi_avg(to_numpy(clip_fraction)).item(),
        )
    else:
        info = dict(  # Provides info about the full step, for printing
            policy_loss=to_numpy(policy_loss).item(),
            entropy_loss=to_numpy(entropy_loss).item(),
            vf_loss=to_numpy(vf_loss).item(),
            total_loss=to_numpy(loss).item(),
            approx_kl=to_numpy(approx_kl).item(),
            clip_fraction=to_numpy(clip_fraction).item(),
        )

    return loss, info