from typing import Optional

import numpy as np
import torch

from torch import nn
from src.tools.modules import masked_softmax, to_one_hot
from src.reps.environment_hea import HEA
from src.reps.ase_graph import TransformAtomsObjectsToGraphXyz, collate_atomsdata
from typing import List

from src.training.rollout import rollout
from src.tools.buffer import PPOBuffer
from MACE.macemodel import MACEModel


class ArtiSAN(nn.Module):
    def __init__(
            self,
            config: dict
    ):
        super().__init__()
        self.device = config['device']
        self.mace_device = config["mace_device"]
        self.num_unit_cells = config['num_unit_cells']
        self.formula = config['formula']
        self.composition = config['composition']
        self.horizon = config['horizon']

        # Multiply dimensions by 9 if using all tensor features (1+ 3 + 5)
        self.node_dim_multiplier = 9 if not config['scalar_pred'] else 1

        self.node_dim = config['node_dim'] * self.node_dim_multiplier

        if self.device == torch.device("cuda"):
            self.pin = True
        else:
            self.pin = False

        self.critic_width = int(config['pred_net_width_critic'])

        self.policy_width = int(config['pred_net_width_policy'])

        self.mace = MACEModel(in_dim=119,
                              emb_dim=config['node_dim'],
                              num_layers=config["conv_depth"],
                              scalar_pred=config["scalar_pred"])

        self.booster = nn.Sequential(
            nn.Linear(self.node_dim, self.critic_width, device=self.device),
            nn.ELU(),
            nn.Linear(self.critic_width, self.node_dim, device=self.device),
        )

        for layer in self.booster:
            classname = layer.__class__.__name__
            if classname.find('Linear') != -1:
                torch.nn.init.orthogonal_(layer.weight)

        self.critic = nn.Sequential(
            nn.Linear(3*self.node_dim, self.critic_width, device=self.device),
            nn.ELU(),
            nn.Linear(self.critic_width, self.critic_width, device=self.device),
            nn.Tanh(),
            nn.Linear(self.critic_width, 1, device=self.device),
        )

        for layer in self.critic:
            classname = layer.__class__.__name__
            if classname.find('Linear') != -1:
                torch.nn.init.orthogonal_(layer.weight)

        self.predict_focus_network = nn.Sequential(
            nn.Linear(self.node_dim*2, self.policy_width, device=self.device),
            nn.SELU(),
            nn.Linear(self.policy_width, 1, device=self.device),
        )

        for layer in self.predict_focus_network:
            classname = layer.__class__.__name__
            if classname.find('Linear') != -1:
                torch.nn.init.orthogonal_(layer.weight)

        self.predict_switch_network = nn.Sequential(
            nn.Linear(self.node_dim*2, self.policy_width, device=self.device),
            nn.SELU(),
            nn.Linear(self.policy_width, 1, device=self.device),
        )

        for layer in self.predict_switch_network:
            classname = layer.__class__.__name__
            if classname.find('Linear') != -1:
                torch.nn.init.orthogonal_(layer.weight)
        self.mace.to(self.mace_device)

    def make_atomic_tensors(self, observations: List[HEA],
                            num_atoms):

        atoms_torch_tensor = torch.zeros(size=(len(observations), num_atoms),
                                         dtype=torch.long,
                                         device=self.device)

        edges_torch = torch.tensor(observations[0].edges, device=self.device, dtype=torch.long)

        positions_tensor = torch.zeros(size=(len(observations), num_atoms, 3),
                                       dtype=torch.float32,
                                       device=self.device)

        for i, obs in enumerate(observations):
            # Get Atoms() object from observation
            atoms = obs.current_atoms
            transformer = TransformAtomsObjectsToGraphXyz(obs.cutoff)
            # Transform to graph dictionary
            graph_state = [transformer(atoms)]

            batch_host = collate_atomsdata(graph_state, pin_memory=self.pin)
            batch = {
                k: v.to(device=self.device, non_blocking=True)
                for (k, v) in batch_host.items()
            }
            x = batch['nodes_xyz'].view(batch['nodes_xyz'].shape[0] * batch['nodes_xyz'].shape[1], batch['nodes_xyz'].shape[2])

            positions_tensor[i, :, :] = x

            atoms_torch = torch.tensor(atoms.get_atomic_numbers(), device=self.device)
            atoms_torch = atoms_torch.unsqueeze(0)

            atoms_torch_tensor[i, :] = atoms_torch

            edges_torch = torch.tensor(obs.edges, device=self.device, dtype=torch.long)

        node_features = self.get_mace_features(atoms_tens=atoms_torch_tensor,
                                               num_atoms=num_atoms,
                                               edge_tens=edges_torch,
                                               pos_tens=positions_tensor)

        return node_features, atoms_torch_tensor

    def get_mace_features(self,
                          atoms_tens,
                          num_atoms,
                          edge_tens,
                          pos_tens):
        if num_atoms < 50:
            batch_size = 13
        elif 50 < num_atoms < 100:
            batch_size = 10
        else:
            batch_size = 5

        node_features = torch.zeros(size=(len(atoms_tens), num_atoms, self.node_dim),
                                    dtype=torch.float32,
                                    device=self.device)

        if len(atoms_tens) == 1:
            num_batches = 1
        else:
            num_batches = int(len(atoms_tens) / batch_size)

        proc_samples = 0
        edge_tens = edge_tens.to(self.mace_device)

        for i in range(num_batches):
            start_ind = i * batch_size
            end_ind = (i + 1) * batch_size
            atoms_tens_tmp = atoms_tens[start_ind:end_ind, :].to(self.mace_device)
            pos_tens_tmp = pos_tens[start_ind:end_ind, :].to(self.mace_device)

            node_features_tmp = self.mace(atom_types=atoms_tens_tmp, edges=edge_tens, positions=pos_tens_tmp)

            node_features_tmp.to(self.device)
            node_features[start_ind:end_ind, :, :] = node_features_tmp[:, :, :]

            proc_samples += i * batch_size
            torch.cuda.empty_cache()

        if proc_samples < len(atoms_tens):
            start_ind = proc_samples
            end_ind = len(atoms_tens)
            atoms_tens_tmp = atoms_tens[start_ind:end_ind, :].to(self.mace_device)
            pos_tens_tmp = pos_tens[start_ind:end_ind, :].to(self.mace_device)

            node_features_tmp = self.mace(atom_types=atoms_tens_tmp, edges=edge_tens, positions=pos_tens_tmp)

            node_features_tmp.to(self.device)
            node_features[start_ind:end_ind, :, :] = node_features_tmp[:, :, :]
            torch.cuda.empty_cache()

        return node_features

    def step(self, obs: [HEA],
             actions: Optional[np.ndarray] = None,
             training: bool = True):

        atomic_feats, atoms_torch = self.make_atomic_tensors(obs, num_atoms=obs[0].num_atoms)

        # Focus
        full_graph_rep = atomic_feats.mean(dim=1).unsqueeze(1) 
        full_graph_rep = full_graph_rep.expand_as(atomic_feats)
        focus_proposals = torch.cat([full_graph_rep, atomic_feats], dim=2)

        focus_logits = self.predict_focus_network(focus_proposals)  # n_obs x n_atoms x 1
        focus_logits = focus_logits.squeeze(-1)  # n_obs x n_atoms

        focus_p = masked_softmax(focus_logits, mask=None)  # n_obs x n_atoms
        focus_dist = torch.distributions.Categorical(probs=focus_p)

        # Cast action to Tensor
        if actions is not None:
            actions = torch.as_tensor(actions, device=self.device)

        # focus: n_obs x 1
        if actions is not None:
            focus = torch.round(actions[:, 0]).long().unsqueeze(-1)
        elif training:
            focus = focus_dist.sample().unsqueeze(-1)
        else:
            focus = torch.argmax(focus_p, dim=-1).unsqueeze(-1)

        focus_oh = to_one_hot(focus, num_classes=atomic_feats.shape[1], device=self.device)

        switch_masks = torch.zeros(size=(atomic_feats.shape[0], atomic_feats.shape[1]), dtype=torch.long, device=self.device)

        # Make this faster:
        for i, atom_type in enumerate(focus):
            switch_masks[i] = atoms_torch[i] != atoms_torch[i][focus[i]]

        focus_emb = (atomic_feats.transpose(1, 2) @ focus_oh[:, :, None]).squeeze(-1)  # n_obs x n_latent

        # Switch
        foc_emb_rep = focus_emb.unsqueeze(1)
        foc_emb_rep = foc_emb_rep.expand_as(atomic_feats)

        switch_proposals = torch.cat([foc_emb_rep, atomic_feats], dim=2)

        switch_logits = self.predict_switch_network(switch_proposals)  # n_obs x n_atoms x 1
        switch_logits = switch_logits.squeeze(-1)  # n_obs x n_atoms

        switch_p = masked_softmax(switch_logits, mask=switch_masks)  # n_obs x n_atoms
        switch_dist = torch.distributions.Categorical(probs=switch_p)

        # focus: n_obs x 1
        if actions is not None:
            switch = torch.round(actions[:, 1]).long().unsqueeze(-1)
        elif training:
            switch = switch_dist.sample().unsqueeze(-1)
        else:
            switch = torch.argmax(switch_p, dim=-1).unsqueeze(-1)

        switch_oh = to_one_hot(switch, num_classes=atomic_feats.shape[1], device=self.device)

        switch_emb = (atomic_feats.transpose(1, 2) @ switch_oh[:, :, None]).squeeze(-1)  # n_obs x n_latent

        if actions is None:
            actions = torch.cat(
                [focus, switch], dim=-1)

        # Critic
        full_mask = torch.abs(focus_oh + switch_oh - torch.ones(focus_oh.shape, device=self.device))

        nodes_boosted = self.booster(self.booster(atomic_feats * full_mask.unsqueeze(-1)))
        nodes_pooled = torch.sum(nodes_boosted, dim=1)  

        predicted_state_energy = self.critic((torch.cat([nodes_pooled, focus_emb, switch_emb], dim=1)))
        v = - predicted_state_energy

        # Log probabilities
        log_prob_list = [
            focus_dist.log_prob(focus.squeeze(-1)).unsqueeze(-1),
            switch_dist.log_prob(switch.squeeze(-1)).unsqueeze(-1),
        ]
        log_prob = torch.cat(log_prob_list, dim=-1)

        # Entropies
        entropy_list = [
            focus_dist.entropy().unsqueeze(-1),
            switch_dist.entropy().unsqueeze(-1),
        ]
        entropy = torch.cat(entropy_list, dim=-1)

        summary_dict = {
            'a': actions,  # n_obs x n_subactions
            'logp': log_prob.sum(dim=-1, keepdim=False),  # n_obs
            'ent': entropy.sum(dim=-1, keepdim=False),  # n_obs
            'v': v.squeeze(-1),  # n_obs
            'entropies': entropy,  # n_obs x n_entropies
        }
        return summary_dict

    def collect_rollouts(self,
                         env: HEA,
                         buffer: PPOBuffer,
                         training: bool,
                         num_steps: Optional[int] = None,
                         num_episodes: Optional[int] = None,
                         evaluation: bool = False,
                         simulate_traj: bool = False,
                         simulate_traj_path: str = None,
                         ):
        if training:
            rollout(agent=self,
                    env=env,
                    buffer=buffer,
                    num_steps=num_steps,
                    num_episodes=num_episodes,
                    evaluation=evaluation,
                    training=training)

        if evaluation and not simulate_traj:
            energy_trajectory = rollout(agent=self,
                                        env=env,
                                        buffer=buffer,
                                        num_steps=num_steps,
                                        num_episodes=num_episodes,
                                        evaluation=evaluation,
                                        training=training)
            return energy_trajectory

        if evaluation and simulate_traj:
            energy_trajectory, entropy_trajectory = rollout(agent=self,
                                        env=env,
                                        buffer=buffer,
                                        num_steps=num_steps,
                                        num_episodes=num_episodes,
                                        evaluation=evaluation,
                                        training=training,
                                        simulate_traj=simulate_traj,
                                        simulate_traj_path=simulate_traj_path,
                                        entropy_traj=True)
            return energy_trajectory, entropy_trajectory