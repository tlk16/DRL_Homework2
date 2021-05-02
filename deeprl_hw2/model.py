import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):  # todo; no BN now
    def __init__(self, in_channels=4, n_actions=6):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.contiguous().view(x.shape[0], -1)))
        return self.head(x)

if __name__ == '__main__':
    model = Model(in_channels=4, n_actions=9)
    a = torch.randn(50, 4, 84, 84)
    b = model(a)
    print(b.shape)
    # todo: better model



