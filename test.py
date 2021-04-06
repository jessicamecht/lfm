from architect import meta_learn
import torch
import utils
import torch.nn as nn
from config import SearchConfig
from models.visual_encoder import Resnet_Encoder

config = SearchConfig()
device = torch.device("cuda")


class EasyModel(nn.Module):
    def __init__(self, input_size):
        super(EasyModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1).to(device)
        x = self.activation(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


if __name__ == "__main__":
    input_size, input_channels, n_classes, train_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=False)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               num_workers=config.workers,
                                               pin_memory=True,
                                               drop_last=True)
    input, target = next(iter(train_loader))
    print(input.shape, target.shape)
    print(input.shape, target.shape)
    input_val, target_val = next(iter(train_loader))
    input, target = input.to(device), target.to(device)
    input_val, target_val = input_val.to(device), target_val.to(device)

    inputDim = next(iter(train_loader))[0].shape[0]
    coefficient_vector = torch.nn.Parameter(torch.ones(inputDim, 1, requires_grad=True).to(device))

    model = EasyModel(torch.flatten(input, start_dim=1).to(device).shape[1])
    model = model.to(device)
    with torch.no_grad():
        o = model(input)

    w_optim = torch.optim.SGD(model.parameters(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)

    visual_encoder = Resnet_Encoder(nn.CrossEntropyLoss())
    visual_encoder = visual_encoder.to(device)

    print('memory_allocated', torch.cuda.memory_allocated() / 1e9, 'memory_reserved',
          torch.cuda.memory_reserved() / 1e9)
    meta_learn(model, w_optim, input, target, input_val, target_val, coefficient_vector, visual_encoder)
    print('memory_allocated1', torch.cuda.memory_allocated() / 1e9, 'memory_reserved',
          torch.cuda.memory_reserved() / 1e9)