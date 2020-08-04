# Weight Initialization
import math
import torch.nn as nn


def layer_init(layer):
    denominator = math.sqrt(layer.in_features)
    nn.init.uniform_(layer.weight, -1/denominator, 1/denominator)


def actor_last_layer_init(layer):
    nn.init.uniform_(layer.weight, -3e-3, 3e-3)


def critic_last_layer_init(layer):
    nn.init.uniform_(layer.weight, -3e-4, 3e-4)
