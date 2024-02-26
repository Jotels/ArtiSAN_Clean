# PROXIMAL POLICY OPTIMIZATION
from typing import Optional, List
import torch
import yaml
import wandb
from src.ArtiSAN import ArtiSAN
from src.tools.buffer import PPOBuffer
import os
from torch.optim import Adam, lr_scheduler
from src.reps.environment_hea import HEA
from src.tools.util import ModelIO, get_tag
from src.training.train import train
import numpy as np
from src.tools import mpi
from src.tools.mpi import set_barrier
from mpi4py import MPI


def ppo(
    env: HEA,
    env_list: List[HEA],
    agent: ArtiSAN,
    config: dict,
    model_handler: Optional[ModelIO] = None,
):
    """
    Proximal Policy Optimization (by clipping), with early stopping based on approximate KL

    Args:
        :param config:

        :param env: HEA Environment for training.

        param eval_envs: HEA Environments for evaluation.

        param agent: Instance of an agent

        param num_steps_per_iter: Number of agent-environment interaction steps per iteration.

        param start_num_steps: Initial number of steps

        param max_num_steps: Maximum number of steps

        param gamma: Discount factor. (Always between 0 and 1.)

        param clip_ratio: Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.)

        :param learning_rate: Learning rate for policy optimizer.

        param vf_coef: coefficient for value function loss term

        param entropy_coef: coefficient for entropy loss term

        param learning_rate: Learning rate for optimizer.

        param gradient_clip: clip norm of gradients before optimization step is taken

        param max_num_train_iters: Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer to take fewer than this.)

        :param lam: Lambda for GAE-Lambda. (Always between 0 and 1, close to 1.)

        :param target_kl: Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.)

        :param eval_freq: How often to evaluate the policy

        param num_eval_episodes: Number of evaluation episodes

        param model_handler: Save model to file

        param save_freq: How often the model is saved

        param rollout_saver: Saves rollout buffers

        param info_saver: Save statistics
    """
    device = config["device"]
    gamma = config["gamma"]
    start_num_steps = config["start_num_steps"]
    max_num_steps = config["max_num_steps"]
    learning_rate = float(config["learning_rate"])
    lam = config["lambda"]
    max_num_train_iters = config["max_num_train_iters"]
    save_freq = config["save_freq"]
    weight_decay = config["weight_decay"]
    num_steps_per_iter = config["num_steps_per_iter"]
    amsgrad = config["amsgrad"]
    multithread = config["multithread"]
    en_atom_scaling = config["en_atom_scaling"]

    # Set up experience buffer

    # Total number of steps
    total_num_steps = start_num_steps

    local_steps_per_iter = int(num_steps_per_iter)
    num_steps_per_iter = local_steps_per_iter

    train_buffer = PPOBuffer(
        size=local_steps_per_iter,
        int_act_dim=2,
        gamma=gamma,
        lam=lam,
        en_atom_scaling=en_atom_scaling,
    )

    optimizer = Adam(
        agent.parameters(), lr=learning_rate, amsgrad=amsgrad, weight_decay=weight_decay
    )

    if config["anneal_lr"]:
        anneal_milestones = list(config["anneal_every"] * np.arange(1, 20))

        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=anneal_milestones, gamma=config["gamma_lr"]
        )
    else:
        scheduler = None

    max_num_iterations = (max_num_steps - total_num_steps) // num_steps_per_iter

    # Main loop
    for iteration in range(max_num_iterations):
        if env_list is not None:
            env = env_list[iteration % len(env_list)]
        # Training rollout
        agent.collect_rollouts(
            env=env, buffer=train_buffer, num_steps=num_steps_per_iter, training=True
        )

        # Obtain (standardized) data for training
        data = train_buffer.get()

        # Train policy
        if multithread:
            set_barrier()

        train(
            agent=agent,
            current_system=env.unit_cells,
            optimizer=optimizer,
            scheduler=scheduler,
            data=data,
            max_num_steps=max_num_train_iters,
            clip_ratio=config["clip_ratio"],
            target_kl=config["target_kl"],
            vf_coef=config["vf_coef"],
            p_coef=config["p_coef"],
            entropy_coef=config["entropy_coef"],
            gradient_clip=config["gradient_clip"],
            multithread=multithread,
            num_atoms=env.num_atoms,
            device=device,
        )

        if multithread:
            set_barrier()

        # Update number of steps taken / trained
        total_num_steps += num_steps_per_iter

        # Evaluate policy
        if (iteration % save_freq == 0) or (iteration == max_num_iterations - 1):
            # Save model
            if mpi.get_proc_rank() == 0:
                model_handler.save(agent)


if __name__ == "__main__":
    # stream = open("/home/energy/jels/ArtiSAN/src/hyperparameter.yaml", "r")
    stream = open("/Users/jonaselsborg/Desktop/ArtiSAN/src/hyperparameter.yaml")

    config = yaml.safe_load(stream)

    project_name = config["project_name"]

    if config["niflheim"]:
        config["num_procs"] = mpi.get_num_procs()
        config["multithread"] = True
        model_dir = "/home/energy/jels/ArtiSAN/" + config["model_dir"]
    else:
        # Since we're running MPI on M1 right now
        config["num_procs"] = mpi.get_num_procs()
        config["multithread"] = True

        model_dir = config["model_dir"]

    tag = get_tag()
    model_handler = ModelIO(directory=model_dir, tag=tag)

    if config["multithread"]:
        rank = mpi.get_proc_rank()
        config["mace_device"] = (
            "cuda:{}".format(rank)
            if torch.cuda.is_available()
            else "cpu:{}".format(rank)
        )
        config["device"] = "cpu:{}".format(rank)
        seed = config["seed"] + rank
    else:
        seed = config["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    train_alloy = HEA(config, eval=False)

    wandb.init(config=config, project=project_name)

    if not config["load_model"]:
        agent = ArtiSAN(config=config)
    else:
        agent = model_handler.load()
        agent.device = config["device"]
        agent.mace_device = config["mace_device"]
        agent.to(agent.device)
        agent.mace.to(agent.mace_device)

    if config["multithread"]:
        mpi.sync_params(agent)
    train_alloy_objects = None
    if config["DiffSize_training_systems"]:
        train_alloy_objects = []
        cell_combinations = [
            [2, 2, 2],
            [2, 2, 3],
            [2, 3, 3],
            [3, 3, 3],
            [2, 2, 4],
            [2, 3, 4],
            [2, 2, 5],
            [2, 2, 6],
        ]

        for cell_comb in cell_combinations:
            ratio = 40 / 108
            horizon = int(np.ceil(4 * ratio * int(np.prod(cell_comb))))
            alloy_for_list = train_alloy.copy()
            alloy_for_list.horizon = horizon
            alloy_for_list.unit_cells = cell_comb
            train_alloy_objects.append(alloy_for_list)

    ppo(
        agent=agent,
        env=train_alloy,
        env_list=train_alloy_objects,
        config=config,
        model_handler=model_handler,
    )
