# ArtiSAN: A deep reinforcement learning agent for solid structure search
This repository contains the code used in training and performing experiments with the Artificial Structure Arranging Net (ArtiSAN).


The ArtiSAN.model file in the Models folder contains the fully trained ArtiSAN model used for generating the figures in the manuscript (ArtiSAN_pretrained.model).
Note that ArtiSAN is trained on CUDA architecture, and thus requires a GPU to evaluate and train. 

One can also run the FORGE.py script to train a new ArtiSAN model that will also be placed in the Models folder.
Note that the scripts require Weights & Biases to visualize training curves (can be installed via pip install wandb). 

Whether running the evaluation scripts for generating the figures or training a new ArtiSAN model, one should replace the directory paths in the hyperparameter.yaml file.

## Installation
ArtiSAN was trained using Python 3.10. 
We recommend  setting up a conda environment and installing the packages outlined in the requirements.txt file.

## Usage

With the ArtiSAN, you can load the model and task it with rearranging any atomic formula in an FCC system.
Currently, the framework supports the elements Aluminum (Al), Nickel (Ni), Copper (Cu), Palladium (Pd), Silver (Ag), Platinum (Pt) and Gold (Au). 
To run an experiment on a list of formulas of choice, run the New_Experiment.py file and change the 'new_experiment_formulas' list in the hyperparameter.yaml file.
You can also change the cell_size parameter from [7, 7, 7] to any cell size desired, but keep in mind that the memory requirement on the GPU grows as a consequence of the MACE graph construction (some systems may not even be able to run the 7x7x7 cell). 

All other hyperparameters are specified in the config YAML file src/hyperparameter.yaml.