import os, sys
import torch
sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def measure_label_similarity(validation_targets, training_targets):
    '''checks for each training and test label if they are the same
    :param validation_targets torch of size (number val targets)
    :param training_targets torch of size (number train targets)
    :returns torch of size (number training targets, number val targets)'''

    val_targets = validation_targets.reshape(validation_targets.shape[0], 1)
    train_targets = training_targets.reshape(training_targets.shape[0], 1)

    train_targets = torch.repeat_interleave(train_targets, val_targets.shape[0], dim=1)
    val_targets = torch.repeat_interleave(val_targets, train_targets.shape[0], dim=1)

    val_train_diff = val_targets.T - train_targets
    assert(val_train_diff[0][0] == validation_targets[0] - training_targets.T[0] )
    assert(val_train_diff.shape[0] == training_targets.shape[0] and val_train_diff.shape[1] == validation_targets.shape[0])
    val_train_equality = (val_train_diff == 0).float()
    return val_train_equality

