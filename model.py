import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

class Attention(nn.Module):
    def __init__(self, feature_dim, intermediate_dim):
        super(Attention, self).__init__()
        self.feature_dim = feature_dim
        self.intermediate_dim = intermediate_dim
        self.attention_fc = nn.Sequential(
            nn.Linear(self.feature_dim, self.intermediate_dim),
            nn.Tanh(),
            nn.Linear(self.intermediate_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x shape: (batch_size, n_features, feature_dim)
        attention_weights = self.attention_fc(x)
        # Apply attention weights
        attended_features = x * attention_weights
        # print(attended_features.shape)
        return attended_features.sum(dim=1)  # Sum over features

class DynamicAttentionODE(nn.Module):
    def __init__(self, num_modalities, hidden_dim):
        super(DynamicAttentionODE, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_modalities, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=-1)  # Ensure output is a valid attention distribution
        )
    
    def forward(self, t, w):
        return self.fc(w)

class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.ode_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t, x):
        return self.ode_block(x)

def attention_ode_solver(attention_module, w0, t_span):
    """
    Solves the ODE defined by the attention module.
    w0: Initial attention weights, shape: [num_modalities]
    t_span: Tensor specifying the integration interval, shape: [2] (start and end)
    """
    solution = odeint(attention_module, w0, t_span)
    return solution[-1]  # Return solution at the final time

#MultimodalFusionODE is a novel implementation  inspired from Multimodal Transformer (MMT) architecture 
#proposed in the paper "Multimodal Transformer for Unaligned Multimodal Language Sequences" 
#by Tsai et al., Ruslan Salakhutdinov is an author (ACL 2019)

class MultimodalFusionODE(nn.Module):
    def __init__(self, input_dims, output_dim, num_modalities=3):
        super(MultimodalFusionODE, self).__init__()
        hidden_dim = output_dim
        self.linear_transforms = nn.ModuleList([nn.Linear(in_dim, hidden_dim) for in_dim in input_dims])
        self.attention_module = DynamicAttentionODE(num_modalities, hidden_dim)
        self.t_span = torch.tensor([0.0, 1.0], dtype=torch.float32)
        self.w0 = nn.Parameter(torch.ones(num_modalities) / num_modalities)
        
        # Use ODEFunc here
        self.ode_func = ODEFunc(hidden_dim)
        
        self.classifier = nn.Linear(hidden_dim * num_modalities, output_dim)

    def forward(self, modalities):
        # Ensure modalities is a list of tensors
        assert isinstance(modalities, list) and all(isinstance(m, torch.Tensor) for m in modalities)
        
        # Transform modalities to common dimensional space
        transformed = [transform(m) for m, transform in zip(modalities, self.linear_transforms)]
        
        # Solve for dynamic attention weights
        attention_weights = attention_ode_solver(self.attention_module, self.w0, self.t_span)
        
        # Apply attention weights and sum
        weighted = [w * t for w, t in zip(attention_weights, transformed)]
        
        # Integrate each modality's representation
        fused = [odeint(self.ode_func, rep, self.t_span, method='dopri5')[-1] for rep in weighted]
        
        # Concatenate fused representations
        fused_cat = torch.cat(fused, dim=-1)
        
        # Classification
        output = self.classifier(fused_cat)
        
        return output

# ------
class MultimodalNetwork(nn.Module):
    def __init__(self, tabular_data_size, n_classes=3):
        super(MultimodalNetwork, self).__init__()
        self.u = nn.Parameter(torch.ones(4))
        # Tabular data branch
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_data_size, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.tabular_classifier = nn.Linear(16, n_classes)
        
        # Genetic data branch, treating as flat input for simplicity
        self.genetic_branch = nn.Sequential(
            nn.Linear(500*6, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.genetic_classifier = nn.Linear(64, n_classes)
        
        
        # Image data branch (3D CNN for simplicity, adjust as needed)
        self.image_branch = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Flatten()
        )
        self.image_classifier = nn.Linear(32 * 32 * 32, n_classes)  # Adjust size accordingly
        
        # Attention layers for each modality
        self.tabular_attention = Attention(16, 8)  # Adjust dimensions as needed
        self.genetic_attention = Attention(64, 32)  # Adjust dimensions as needed
        flattened_image_size = 32 * 32 * 32  # Placeholder for the actual flattened size after image branch
        self.image_attention = Attention(flattened_image_size, flattened_image_size // 2)  # Adjust dimensions
        # Novel data fusion strategy
        self.fusion_gate = MultimodalFusionODE(input_dims=[16, 64, flattened_image_size], output_dim=512)

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Classification module
        self.classifier = nn.Linear(512, n_classes)
        # Final classifier that combines all branches
        
        # self.classifier = nn.Sequential(
        #     nn.Linear(16 + 64 + flattened_image_size, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.LeakyReLU(),
        #     nn.Linear(1024, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Linear(64, n_classes)  # Assuming 3 classes
        # )
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, tabular_data, genetic_data, image_data, labels):
        # print(tabular_data.shape)
        # print(genetic_data.shape)
        # print(image_data.shape)
        tabular_out = self.tabular_branch(tabular_data)
        genetic_out = self.genetic_branch(genetic_data.view(-1, 500*6))
        image_out = self.image_branch(image_data)
        tabular_cls = self.tabular_classifier(tabular_out)
        genetic_cls = self.genetic_classifier(genetic_out)
        image_cls = self.image_classifier(image_out)
        
        image_attn = self.image_attention(image_out.unsqueeze(1))
        tabular_attn = self.tabular_attention(tabular_out.unsqueeze(1))
        genetic_attn = self.genetic_attention(genetic_out.unsqueeze(1))

        # Fuse modalities using the novel fusion gate
        fused_representation = self.fusion_gate([tabular_attn, genetic_attn, image_attn])

        # Flatten and feed into transformer encoder
        fused_representation = fused_representation.view(fused_representation.size(0), -1)
        # print(fused_representation.shape)
        # Classification
        output = self.classifier(fused_representation)
        loss_t = self.criterion(tabular_cls, labels)
        loss_g = self.criterion(genetic_cls, labels)
        loss_i = self.criterion(image_cls, labels)
        
        #loss = self.criterion(output, torch.max(labels, 1)[1])
        loss = self.criterion(output, labels)
        weights = torch.softmax(self.u, dim=0)
        total_loss = weights[0]*loss_t + weights[1]*loss_g + weights[2]*loss_i + weights[3]*loss

        return weights, total_loss, output
