import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):  # todo: BN or not?
    def __init__(self, in_channels=4, n_actions=6):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc4(x.contiguous().view(x.shape[0], -1)))
        return self.head(x)


class LinearQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=6):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(LinearQN, self).__init__()
        self.fc = nn.Linear(84 * 84 * in_channels, n_actions)

    def forward(self, x):
        x = x.float() / 255
        return self.fc(x)


class DuelDQN(nn.Module):  # todo: BN or not?
    def __init__(self, in_channels=4, n_actions=6):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(DuelDQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.action_head = nn.Linear(512, n_actions)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.contiguous().view(x.shape[0], -1)
        x = F.relu(self.fc4(x))
        return self.action_head(x), self.value_head(x)



if __name__ == '__main__':
    model = DQN(in_channels=4, n_actions=9)
    a = torch.randn(50, 4, 84, 84)
    b = model(a)
    print(b.shape)
    # todo: better model



