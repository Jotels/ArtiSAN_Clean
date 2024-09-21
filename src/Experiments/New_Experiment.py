import torch
import numpy as np
import os
import sys
import wandb
import yaml
import plotly.graph_objects as go
import plotly.subplots as sp
from src.tools.util import ModelIO, get_tag
from typing import List
from src.ArtiSAN import ArtiSAN
from src.tools.buffer import PPOBuffer
from src.reps.environment_hea import HEA
from src.tools.util import to_numpy
import time
from ase.io.trajectory import Trajectory
from ase.io import write
from itertools import combinations, product
import re

def get_all_formulas(config):
    elements = ["Al", "Ni", "Cu", "Pd", "Ag", "Pt", "Au"]
    element_dict = {"Al": 0, "Ni": 1, "Cu": 2, "Pd": 3, "Ag": 4, "Pt": 5, "Au": 6}

    reduced_formula_list = [config['new_experiment_formulas']]

    fraction_list_all = []

    for rf in reduced_formula_list:
        rf_dict = parse_molecule(rf)
        fraction_list = list(np.zeros(len(elements)))
        total_atoms = sum(rf_dict.values())
        for element in rf_dict:
            fraction_list[element_dict[element]] = rf_dict[element] / total_atoms
        fraction_list_all.append(fraction_list)

    return reduced_formula_list, fraction_list_all

def reduce_formula(formula):
    elements = []
    counts= []
    for i in range(0, len(formula), 3):
        element = formula[i:i+2]
        count = int(formula[i+2])
        elements.append(element)
        counts.append(count)
    gcd = np.gcd.reduce(counts)
    reduced_counts = [count // gcd for count in counts]
    for i in range(len(reduced_counts)):
        if reduced_counts[i] == 1:
            reduced_counts[i] = ''
    reduced_formula = ''.join(f"{element}{count}" for element, count in zip(elements, reduced_counts))
    return reduced_formula

def parse_molecule(molecule):
    # Parses a molecule into a dictionary of element counts
    elements = {}
    current_element = ""
    current_count = ""
    for char in molecule:
        if char.isupper():
            if current_element:
                elements[current_element] = int(current_count) if current_count else 1
            current_element = char
            current_count = ""
        elif char.islower():
            current_element += char
        elif char.isdigit():
            current_count += char
    if current_element:
        elements[current_element] = int(current_count) if current_count else 1
    return elements

def get_systems(config):
    # Set up evaluation alloys
    # Order is 'AlNiCuPdAgPtAu'
    compositions, fractions = get_all_formulas(config)
    eval_alloys = []
    cell_size = config['cell_size']
    num_switches = int(np.prod(cell_size) * 4 * 15)

    counter = 0

    for frac_list, comp in zip(fractions, compositions):
        if counter % 100 == 0:
           print("______________________________")
           print(f"Creating alloy no. {counter}")
           print("______________________________")
        eval_alloy = HEA(config, eval=True)
        eval_alloy.horizon = 1000000
        eval_alloy.unit_cells = cell_size
        eval_alloy = eval_alloy.reset(fixed_composition=comp, frac_list=frac_list)
        eval_alloys.append(eval_alloy)
        counter += 1
    return eval_alloys, num_switches, compositions

def check_create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"Directory '{dir}' created.")
    else:
        print(f"Directory '{dir}' already exists.")

def evaluate_model(agent: ArtiSAN,
                   eval_alloy: HEA,
                composition: str,
                   num_switches: int,
                   config: dict):
    energy_trajectory = np.zeros(num_switches)
    composition_type = re.sub(r'\d', '', composition)

    if len(composition_type) == 4:
        alloy_type = "Binary"
    elif len(composition_type) == 6:
        alloy_type = "Ternary"
    elif len(composition_type) == 8:
        alloy_type = "Quaternary"
    elif len(composition_type) == 10:
        alloy_type = "Quinary"

    starting_energy = eval_alloy.prior_energy
    eval_alloy_copy = eval_alloy.copy()

    energy_trajectory[0] = starting_energy

    path_to_traj_folder = config["Fig4_Traj_Dir"] + f"/{alloy_type}/{composition_type}"

    check_create_dir(path_to_traj_folder)

    wandb.init(config=config, project=config["Fig4_"], group=f"{alloy_type}")
    trajectory = Trajectory(path_to_traj_folder + '/'+str(composition)+ '_seed_' + str(config["seed"]) + '.traj', mode='a', master=True)
    trajectory.write(eval_alloy_copy.current_atoms)

    # Agent
    for j in range(1, num_switches):
        with torch.no_grad():
            pred = agent.step(obs=[eval_alloy_copy], training=True)

            action = to_numpy(pred['a'][0])  # Get action

            _, reward, _, post_swap_energy, _ = eval_alloy_copy.environment_step(focus_idx=action[0], swap_idx=action[1])
            print(
                f"AGENT SWITCH {eval_alloy_copy.num_swaps} |  {eval_alloy_copy.current_atoms.symbols[action[1]]} at {action[0]} <---> {eval_alloy_copy.current_atoms.symbols[action[0]]} at {action[1]} | Energy chany change: {-reward} eV")

            trajectory.write(eval_alloy_copy.current_atoms)
            energy_trajectory[j] = post_swap_energy

        final_energy = eval_alloy_copy.prior_energy

    for j in range(num_switches):
        wandb_traj_dict = {}
        wandb_traj_dict[f"{composition}_seed_{config['seed']} Energy"] = energy_trajectory[j]
        wandb.log(wandb_traj_dict)

    if __name__ == '__main__':
        stream = open("/home/energy/jels/ArtiSAN/src/hyperparameter.yaml", 'r')

    config = yaml.safe_load(stream)

    tag = get_tag()

    model_dir = "../ArtiSAN/" + config["model_dir"]

    model_handler = ModelIO(directory=model_dir, tag=tag)

    # Load model
    agent = model_handler.load()
    rank = 0
    config["mace_device"] = (
        "cuda:{}".format(rank)
        if torch.cuda.is_available()
        else "cpu:{}".format(rank)
    )
    config["device"] = "cpu:{}".format(rank)
    seed = config["seed"] + rank
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    agent.device = config["device"]
    agent.mace_device = config["mace_device"]
    agent.to(agent.device)
    agent.mace.to(agent.mace_device)
    eval_systems, num_switches, compositions = get_systems(config)

    for system, composition in zip(eval_systems, compositions):
        evaluate_model(agent=agent,
                       composition=composition,
                       eval_alloy=system,
                       num_switches=num_switches,
                       config=config)
    wandb.finish()
