import torch
from torch import nn

class tinyVGG(nn.Module):
    def __init__(self, input_shape, hidden_state, ouput_shape):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_state,
                        kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_state, out_channels=hidden_state,
                        kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_state, out_channels=hidden_state,
                        kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_state, out_channels=hidden_state,
                        kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_state*7*7, out_features=ouput_shape)
        )
    
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        return(self.classifier_layer(x))

