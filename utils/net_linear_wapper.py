import torch

class LinearWapper(nn.Module):
    def __init__(self, model):
        super(LinearWapper, self).__init__()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
            nn.init.constant_(m.bias, 0)

    def forward(self, input):
        return {'logits': F.linear(input, self.weight, self.bias)}
