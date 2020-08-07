# Weight Initialization
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T


def layer_init(layer):
    denominator = math.sqrt(layer.in_features)
    nn.init.uniform_(layer.weight, -1 / denominator, 1 / denominator)


def actor_last_layer_init(layer):
    nn.init.uniform_(layer.weight, -3e-3, 3e-3)


def critic_last_layer_init(layer):
    nn.init.uniform_(layer.weight, -3e-4, 3e-4)


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.Grayscale(),
                    T.ToTensor()])


def get_screen(screen):
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen.transpose((2, 0, 1)))
    # resize and gray scale
    # return resize(screen).unsqueeze(0)
    return resize(screen).numpy().squeeze(0)


if __name__ == '__main__':
    import gym

    env = gym.make('Pendulum-v0')
    env.reset()
    state = env.render(mode='rgb_array')
    print(state.shape)
    plt.figure()
    # plt.imshow(get_screen(state).cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
    plt.imshow(np.array([get_screen(state).cpu().numpy() for _ in range(3)]).transpose((1, 2, 0)),
               interpolation='none')
    plt.title('Example extracted screen')
    plt.show()
    env.close()
