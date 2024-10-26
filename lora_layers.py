import torch
import torch.nn as nn
from functools import partial

# Define LoRA Layer
class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(input_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, output_dim))
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)

# LoRA integration for Linear layers
class LinearWithLoRA(nn.Module):
  def __init__(self, linear_layer, rank, alpha):
    super().__init__()
    self.base_layer = linear_layer  # Assign the input linear layer
    self.lora_layer = LoRALayer(linear_layer.in_features, linear_layer.out_features, rank, alpha)

  def forward(self, x):
    return self.base_layer(x) + self.lora_layer(x)
# class LinearWithLoRA(nn.Module):
#     def __init__(self, linear_layer, rank, alpha):
#         super().__init__()
#         self.base_layer = linear_layer
#         self.lora_layer = LoRALayer(linear_layer.in_features, linear_layer.out_features, rank, alpha)

#     def forward(self, x):
#         return self.base_layer(x) + self.lora_layer(x)


# Assign LoRA layers to model
def apply_lora_to_model(model, lora_r=8, lora_alpha=16):
    assign_lora_layer = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)

    for layer in model.distilbert.transformer.layer:
        # Apply LoRA to attention heads and feedforward layers
        layer.attention.q_lin = assign_lora_layer(layer.attention.q_lin)
        layer.attention.k_lin = assign_lora_layer(layer.attention.k_lin)
        layer.attention.v_lin = assign_lora_layer(layer.attention.v_lin)
        layer.attention.out_lin = assign_lora_layer(layer.attention.out_lin)
        layer.ffn.lin1 = assign_lora_layer(layer.ffn.lin1)
        layer.ffn.lin2 = assign_lora_layer(layer.ffn.lin2)

    # Apply LoRA to pre-classification and classification heads
    model.pre_classifier = assign_lora_layer(model.pre_classifier)
    model.classifier = assign_lora_layer(model.classifier)