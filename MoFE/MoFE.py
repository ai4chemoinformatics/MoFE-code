import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit.Chem import FunctionalGroups
import math

class MLPExpert(nn.Module):
    ##MLP based expert
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPExpert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class KANExpert(nn.Module):
    ##KAN based expert
    def __init__(self, input_dim, output_dim, grid_size=5, k=3):
        super(KANExpert, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.k = k  ##B-spline order
        
        ##Initialize B-spline basis function parameters
        self.spline_scaling = nn.Parameter(torch.ones(input_dim, output_dim))
        self.spline_coeff = nn.Parameter(torch.randn(input_dim, output_dim, grid_size + k))
        
        ##Initialize grid points
        self.register_buffer('grid', torch.linspace(-1, 1, grid_size + 1))
        
    def b_spline_basis(self, x, i, k):
        ##Compute B-spline basis function
        if k == 0:
            return ((self.grid[i] <= x) & (x < self.grid[i+1])).float()
        else:
            denom1 = self.grid[i+k] - self.grid[i]
            term1 = 0 if denom1 == 0 else (x - self.grid[i]) / denom1 * self.b_spline_basis(x, i, k-1)
            
            denom2 = self.grid[i+k+1] - self.grid[i+1]
            term2 = 0 if denom2 == 0 else (self.grid[i+k+1] - x) / denom2 * self.b_spline_basis(x, i+1, k-1)
            
            return term1 + term2
    
    def forward(self, x):
        batch_size = x.size(0)
        
        ##Apply learnable activation function for each input dimension
        outputs = []
        for j in range(self.output_dim):
            output_j = 0
            for i in range(self.input_dim):
                ##Compute B-spline basis function
                basis_values = []
                for g in range(self.grid_size + self.k):
                    basis_values.append(self.b_spline_basis(x[:, i], g, self.k))
                
                basis_matrix = torch.stack(basis_values, dim=1)  ##[batch_size, grid_size+k]
                
                ##Apply spline coefficients
                spline_contribution = torch.matmul(basis_matrix, self.spline_coeff[i, j])
                
                ##Apply scaling
                output_j += self.spline_scaling[i, j] * spline_contribution
            
            outputs.append(output_j)
        
        return torch.stack(outputs, dim=1)

class FunctionalGroupRouter(nn.Module):
    ##Functional Group based Routing layer
    def __init__(self, functional_group_smarts):
        super(FunctionalGroupRouter, self).__init__()

        self.num_experts = 8
        self.functional_group_smarts = functional_group_smarts

        self.other_expert_idx = 7
        
    def smiles_has_functional_group(self, smiles, smart_pattern):
        ##Check if SMILES has specific functional group
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            pattern = Chem.MolFromSmarts(smart_pattern)
            if pattern is None:
                return False
            return mol.HasSubstructMatch(pattern)
        except:
            return False
    
    def forward(self, smiles_list):
        ##Determine which experts to route to based on functional groups in SMILES strings
        
        batch_size = len(smiles_list)
        expert_mask = torch.zeros(batch_size, self.num_experts)
        
        for i, smiles in enumerate(smiles_list):
            ##Check each functional group
            active_experts = []
            for j, smart_pattern in enumerate(self.functional_group_smarts):
                if self.smiles_has_functional_group(smiles, smart_pattern):
                    active_experts.append(j)
            
            ##If no matching functional group, use "other" expert
            if not active_experts:
                active_experts = [self.other_expert_idx]
            
            ##Set expert mask
            for expert_idx in active_experts:
                expert_mask[i, expert_idx] = 1.0
                
        return expert_mask

class MoEByFunctionalGroup(nn.Module):
    ##Functional Group based MoE
    def __init__(self, input_dim, hidden_dim, output_dim, functional_group_smarts, expert_types):
        ##Parameters:
        ##input_dim: input dimension
        ##hidden_dim: hidden layer dimension for MLP experts
        ##output_dim: output dimension
        ##functional_group_smarts: list of functional group SMARTS patterns
        ##expert_types: list of expert types, length 8, each element is 'mlp' or 'kan'
        super(MoEByFunctionalGroup, self).__init__()
        ##7 specific functional group experts + 1 "other" expert
        self.num_experts = 8
        self.functional_group_smarts = functional_group_smarts
        self.expert_types = expert_types
        
        ##Create expert networks
        self.experts = nn.ModuleList()
        for i in range(self.num_experts):
            if expert_types[i] == 'mlp':
                self.experts.append(MLPExpert(input_dim, hidden_dim, output_dim))
            else:  ##'kan'
                self.experts.append(KANExpert(input_dim, output_dim))
        
        ##Routing layer
        self.router = FunctionalGroupRouter(functional_group_smarts)
        
        ##Gating network (optional, for weighted combination of expert outputs)
        self.gate = nn.Linear(input_dim, self.num_experts)
        
    def forward(self, x, smiles_list):
        ##Get routing mask
        expert_mask = self.router(smiles_list).to(x.device)
        
        ##Calculate gating weights
        gate_weights = F.softmax(self.gate(x), dim=1)
        
        ##Apply routing mask
        masked_gate_weights = gate_weights * expert_mask
        ##Normalize
        normalized_weights = masked_gate_weights / (masked_gate_weights.sum(dim=1, keepdim=True) + 1e-9)
        
        ##Collect expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)  ##[batch_size, 8, output_dim]
        
        ##Weighted combination of expert outputs
        output = torch.sum(expert_outputs * normalized_weights.unsqueeze(2), dim=1)
        
        return output, expert_mask, normalized_weights
    
    def get_expert_parameters(self, expert_idx):
        ##Get parameters of specific expert for fine-tuning
        return self.experts[expert_idx].parameters()
    
    def freeze_all_experts_except(self, expert_indices):
        ##Freeze all experts except specified ones
        for i, expert in enumerate(self.experts):
            if i not in expert_indices:
                for param in expert.parameters():
                    param.requires_grad = False
            else:
                for param in expert.parameters():
                    param.requires_grad = True

class ContrastiveLoss(nn.Module):
    ##Contrastive learning loss function
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features, expert_mask):
        ##Calculate contrastive loss
        
        ##Parameters:
        ##features: feature vectors [batch_size, feature_dim]
        ##expert_mask: expert mask [batch_size, 8]
        batch_size = features.size(0)
        
        ##Calculate similarity matrix
        features_norm = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features_norm, features_norm.T) / self.temperature
        
        ##Create label matrix: if two samples share at least one expert, they are positive pairs
        expert_assignments = expert_mask > 0.5  ##Boolean mask
        labels = torch.matmul(expert_assignments.float(), expert_assignments.float().T) > 0
        labels = labels.float()
        
        ##Create mask to exclude diagonal (self-comparison)
        mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        labels = labels[~mask].view(batch_size, -1)
        similarity_matrix = similarity_matrix[~mask].view(batch_size, -1)
        
        ##Calculate contrastive loss
        positives = similarity_matrix[labels.bool()]
        negatives = similarity_matrix[~labels.bool()]
        
        ##Calculate loss using softmax
        logits = torch.cat([positives.unsqueeze(1), negatives.view(batch_size, -1)], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=features.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss