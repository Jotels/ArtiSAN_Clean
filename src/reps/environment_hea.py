import numpy as np
import torch
import random

from ase import Atoms
import copy
from ase.io.trajectory import Trajectory
from ase.visualize import view
from ase.calculators.emt import EMT
#from asap3 import EMT
from ase.lattice.cubic import FaceCenteredCubic, BodyCenteredCubic, SimpleCubic
from ase.data import reference_states, atomic_numbers
from collections import Counter
from itertools import combinations
from src.reps.ase_graph import TransformAtomsObjectsToGraphXyz, collate_atomsdata
import time
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor

class HEA:
    def __init__(
        self,
        config,
        eval: bool,
    ):
        self.init_state_entropy = None
        self.prior_ent = None
        self.entropy = None
        self.eval = eval
        self.adj_mat = None
        self.simple_formula = None
        self.all_baseline_en = None
        self.collated_edges = None
        self.edge_dict = None
        self.edges = None
        self.edges_displacement = None
        self.e_tot = None
        self.horizon = None
        self.composition = None
        self.lattice_constant = None
        self.num_edges = None
        self.cg = None
        self.collated_graph = None
        self.graph_state = None
        self.transformer = None
        self.symbol_map = None
        self.cutoff = None
        self.init_state_energy = None
        self.prior_energy = None
        self.lowest_randomized_energy = None
        self.largest_randomized_energy = None
        self.num_atoms = None
        self.symbol_list = None
        self.symbol_list_copy = None
        self.baseline_energy = None
        self.hea_formula = config["formula"]
        self.num_unit_cells = config["num_unit_cells"]
        self.lattice_structure = config["lattice_structure"]
        self.terminal_only = config["terminal_only"]
        self.seed = config["seed"]
        self.device = config["device"]
        self.num_baseline_randomizations = config["num_baseline_randomizations"]
        self.niflheim = config["niflheim"]

        random.seed(self.seed)

        # Atoms holders stuff
        self.current_atoms = Atoms()
        self.init_state = Atoms()
        self.elements = [
            self.hea_formula[idx : idx + 2]
            for idx in range(0, len(self.hea_formula), 2)
        ]
        self.element_numbers = [atomic_numbers[element] for element in self.elements]

        if self.device == torch.device("cuda"):
            self.pin = True
        else:
            self.pin = False

        self.num_different_elements = len(self.elements)
        self.training = True
        self.cutoff_offset = config["cutoff_offset"]

        self.num_swaps = 0
        self.unit_cells = [3, 3, 3]
        self.horizon = config["horizon"]

    def create_structure(self, horizon: int):
        # Randomize new unit cell size:

        # Set the crystal structure:
        # FCC
        if self.lattice_structure == "fcc":
            unit_cells = self.unit_cells
            self.init_state = FaceCenteredCubic(
                "Au", latticeconstant=self.lattice_constant, size=unit_cells
            )
        # BCC
        elif self.lattice_structure == "bcc":
            unit_cells = self.unit_cells
            self.init_state = BodyCenteredCubic(
                "Au", latticeconstant=self.lattice_constant, size=unit_cells
            )
        # SimpleCubic
        elif self.lattice_structure == "sc":
            unit_cells = self.unit_cells
            self.init_state = SimpleCubic(
                "Au", latticeconstant=self.lattice_constant, size=unit_cells
            )

        self.num_atoms = self.init_state.get_global_number_of_atoms()

        if horizon is None:
            self.horizon = self.num_atoms
        else:
            self.horizon = horizon

        get_baseline = False

        num_atoms_per_element = [
            int(fraction * self.num_atoms) for fraction in self.composition
        ]

        symbol_list = []

        for number, element in zip(num_atoms_per_element, self.elements):
            symbol_list += number * [str(element)]

        if len(self.composition) != len(self.elements):
            raise ValueError(
                "Number of fractional occupancies must match number of elements!"
            )

        number_of_formula_atoms = len(symbol_list)
        if number_of_formula_atoms < self.num_atoms:
            difference = self.num_atoms - number_of_formula_atoms
            for i in range(difference):
                symbol_list += [random.choice(symbol_list)]

        print(
            "Rearranging {} alloy with {} atoms using {} switch(es):".format(
                self.lattice_structure, self.num_atoms, self.horizon
            )
        )
        symbol_count_series = pd.Series(symbol_list).value_counts()
        self.symbol_dict = dict()
        for element, count in zip(symbol_count_series.index, symbol_count_series):
            self.symbol_dict[element] = int(count)
        print(self.symbol_dict)
        c_atom, n_atom = list(pd.Series(symbol_list).value_counts().index), list(
            pd.Series(symbol_list).value_counts().values
        )
        simple_formula_list = [str(c_a) + str(n_a) for c_a, n_a in zip(c_atom, n_atom)]
        self.simple_formula = "".join(simple_formula_list)

        # Create symbol list, symbol map and a list containing the number of each constituent atom to be placed
        random.shuffle(symbol_list)
        self.symbol_list = symbol_list.copy()
        self.symbol_list_copy = symbol_list.copy()
        self.symbol_map = list(Counter(self.symbol_list).keys()).copy()
        self.init_state.symbols = self.symbol_list

        self.current_atoms = self.init_state.copy()

        self.current_atoms.symbols = self.symbol_list_copy

        # graph variables
        self.cutoff = self.lattice_constant + self.cutoff_offset
        self.transformer = TransformAtomsObjectsToGraphXyz(cutoff=self.cutoff)

        # get edges so neighborlist is only called once per alloy
        if np.any(self.current_atoms.get_pbc()):
            self.current_atoms.wrap()  # Make sure all atoms are inside unit cell
            (
                self.edges,
                self.edges_displacement,
            ) = self.transformer.get_edges_neighborlist(self.current_atoms)
        else:
            self.edges, self.edges_displacement = self.transformer.get_edges_simple(
                self.current_atoms
            )

        self.adj_mat = torch.zeros(
            self.num_atoms, self.num_atoms, dtype=bool, device=self.device
        )

        for edge in self.edges:
            i1, i2 = edge[0], edge[1]
            self.adj_mat[i1][i2] = True

        self.edge_dict = {
            "edges": torch.tensor(self.edges),
            "edges_displacement": torch.tensor(
                self.edges_displacement, dtype=torch.get_default_dtype()
            ),
            "num_edges": torch.tensor(self.edges.shape[0]),
        }

        self.graph_state = [{**self.transformer(self.current_atoms), **self.edge_dict}]

        self.collated_graph = collate_atomsdata(self.graph_state, pin_memory=self.pin)
        self.cg = {
            k: v.to(device=self.device, non_blocking=True)
            for (k, v) in self.collated_graph.items()
        }

        self.num_edges = self.collated_graph["num_edges"]

        # Calculate new baseline plus lowest and largest randomized energy
        if get_baseline:
            (
                self.baseline_energy,
                self.lowest_randomized_energy,
                self.largest_randomized_energy,
                self.all_baseline_en,
            ) = self.get_baseline(
                self.symbol_list_copy, self.num_baseline_randomizations
            )
        else:
            (
                self.baseline_energy,
                self.lowest_randomized_energy,
                self.largest_randomized_energy,
            ) = (0, 0, 0)

    def swap_atom(self, focus_index, swap_index):
        # NEW METHOD:
        (
            self.current_atoms.symbols[focus_index],
            self.current_atoms.symbols[swap_index],
        ) = (
            self.current_atoms.symbols[swap_index],
            self.current_atoms.symbols[focus_index],
        )

    def write_to_traj(
        self, path, mode
    ):  # Problems arise when changing this one, so don't
        traj = Trajectory(filename=path, mode=mode)
        traj.write(self.current_atoms)

    def reset(self, fixed_composition=None, frac_list=None):
        horizon = self.horizon
        if fixed_composition is None:
            # Choose between 0 and 5 zeroes to add, so we also sometimes test alloys of down to two different elements
            zero_array = np.zeros(np.random.randint(0, len(self.element_numbers)-1))

            composition_nonzero = np.ones(len(self.element_numbers)-len(zero_array))

            self.num_atoms = int(np.prod(self.unit_cells) * 4)

            min_atoms_per_species = int(np.prod(self.unit_cells) * (1/len(composition_nonzero)))
            composition_nonzero = composition_nonzero * min_atoms_per_species

            atoms_to_distribute = int(self.num_atoms - composition_nonzero.sum())

            for i in range(len(composition_nonzero)):
                if i == len(composition_nonzero)-1:
                    num_atoms_to_add = atoms_to_distribute
                else:
                    num_atoms_to_add = np.random.randint(0, atoms_to_distribute)
                composition_nonzero[i] += num_atoms_to_add
                atoms_to_distribute -= num_atoms_to_add

            composition_normalized = composition_nonzero / composition_nonzero.sum()

            self.composition = np.concatenate((composition_normalized, zero_array))
            self.composition = self.composition.tolist()

            # shuffle the array to distribute zeros
            np.random.shuffle(self.composition)

            self.lattice_constant = np.sum(
                [
                    reference_states[element_number]["a"] * fraction
                    for (element_number, fraction) in zip(
                        self.element_numbers, self.composition
                    )
                ]
            )
        else:
            self.composition = frac_list
            self.lattice_constant = np.sum(
                [
                    reference_states[element_number]["a"] * fraction
                    for (element_number, fraction) in zip(
                    self.element_numbers, self.composition
                )
                ]
            )

        self.create_structure(horizon=horizon)

        self.num_swaps = 0

        self.current_atoms.calc = EMT()
        self.prior_energy = self.current_atoms.get_potential_energy()
        self.current_atoms.calc = None
        self.init_state_energy = float(self.prior_energy)
        self.init_state_entropy = float(self.prior_ent)
        if self.eval:
            print("Initiating evaluation alloy at {} eV".format(self.init_state_energy))
        else:
            print("Initiating training alloy at {} eV".format(self.init_state_energy))
        print(
            "................................................................................................................................................................................................................................................"
        )
        self.e_tot = float(self.init_state_energy)

        return self

    def render(self):
        view(self.current_atoms)

    def get_reward(self):
        """
        Does the energy calculation using the set calculator
        """
        self.current_atoms.calc = EMT()
        self.e_tot = self.current_atoms.get_potential_energy()
        energy_reduction = self.prior_energy - self.e_tot
        reward = energy_reduction
        self.prior_energy = self.e_tot
        self.current_atoms.calc = None
        return reward

    def environment_step(self, focus_idx, swap_idx):
        self.num_swaps += 1

        terminal = self.num_swaps == self.horizon

        self.swap_atom(focus_index=focus_idx, swap_index=swap_idx)

        reward = self.get_reward()

        return self, reward, terminal, self.prior_energy, self.prior_ent

    def set_calc(self):
        self.current_atoms.calc = EMT()

    def copy(self):
        cp = copy.deepcopy(self)
        return cp
