import torch
import gc
import utils
import torch.nn as nn
from config import SearchConfig
from weight_samples.sample_weights import calc_instance_weights

from models.visual_encoder import Resnet_Encoder
import higher
import torch.nn.functional as F
from mem_report import mem_report



config = SearchConfig()
device = torch.device("cuda")


class EasyModel(nn.Module):
    def __init__(self, input_size):
        super(EasyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1).to(device)
        x = self.fc1(x)
        return x

def meta_learn(model, optimizer, input, target, input_val, target_val, coefficient_vector, visual_encoder, visual_encoder_optimizer, coeff_vector_optimizer):
    with torch.no_grad():
        logits_val = model(input_val)
    visual_encoder_optimizer.zero_grad()
    coeff_vector_optimizer.zero_grad()
    with torch.backends.cudnn.flags(enabled=False):
        with higher.innerloop_ctx(model, optimizer, copy_initial_weights=True, track_higher_grads=True, device=device) as (fmodel, foptimizer):
            # functional version of model allows gradient propagation through parameters of a model
            logits = fmodel(input)

            weights = calc_instance_weights(input, target, input_val, target_val, logits_val, coefficient_vector, visual_encoder)
            loss = F.cross_entropy(logits, target, reduction='none')
            weighted_training_loss = torch.mean(weights * loss)
            foptimizer.step(weighted_training_loss)  # replaces gradients with respect to model weights -> w2

            logits_val = fmodel(input_val)
            meta_val_loss = F.cross_entropy(logits_val, target_val)
            meta_val_loss.backward()
            visual_encoder_optimizer.step()
            coeff_vector_optimizer.step()
            logits.detach()
            weighted_training_loss.detach()
        optimizer.zero_grad()

        for module in fmodel.modules():
            if isinstance(module, nn.Linear):
                del module.weight
        del logits, meta_val_loss, foptimizer, fmodel, weighted_training_loss, logits_val, weights,
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    input_size, input_channels, n_classes, train_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=False)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               num_workers=config.workers,
                                               pin_memory=True,
                                               drop_last=True)
    input, target = next(iter(train_loader))

    input_val, target_val = next(iter(train_loader))
    input, target = input.to(device), target.to(device)
    input_val, target_val = input_val.to(device), target_val.to(device)

    inputDim = next(iter(train_loader))[0].shape[0]
    coefficient_vector = torch.nn.Parameter(torch.ones(inputDim, 1, requires_grad=True).to(device))

    model = EasyModel(torch.flatten(input, start_dim=1).to(device).shape[1])
    model = model.to(device)

    w_optim = torch.optim.SGD(model.parameters(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)

    visual_encoder = Resnet_Encoder(nn.CrossEntropyLoss())
    visual_encoder = visual_encoder.to(device)

    visual_encoder_optimizer = torch.optim.Adam(visual_encoder.parameters(), betas=(0.5, 0.999),
                                                weight_decay=config.alpha_weight_decay)

    coeff_vector_optimizer = torch.optim.Adam([coefficient_vector], betas=(0.5, 0.999),
                                              weight_decay=config.alpha_weight_decay)


    for i in range(10):
        print('memory_allocated', torch.cuda.memory_allocated() / 1e9, 'memory_reserved',
          torch.cuda.memory_reserved() / 1e9)
        meta_learn(model, w_optim, input, target, input_val, target_val, coefficient_vector, visual_encoder, visual_encoder_optimizer, coeff_vector_optimizer)
        print('memory_allocated1', torch.cuda.memory_allocated() / 1e9, 'memory_reserved',
          torch.cuda.memory_reserved() / 1e9)
    del coeff_vector_optimizer, visual_encoder_optimizer, visual_encoder, w_optim, model, coefficient_vector, inputDim, input_val, target_val, device, input, target, input_size, input_channels, n_classes, train_data
    gc.collect()
    torch.cuda.empty_cache()
    mem_report()
