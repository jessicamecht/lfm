import torch
import torch.nn as nn
from weight_samples.visual_similarity import visual_validation_similarity
from weight_samples.label_similarity import measure_label_similarity
import gc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_similarities(predictive_performance, visual_similarity_scores, label_similarity_scores):
    '''calculates the element wise multiplication of the three inputs
    multiplies all elements of each row element wise and
    :param predictive_performance torch of size (number val examples)
    :param visual_similarity_scores torch of size (number train examples, number val examples)
    :param label_similarity_scores torch of size (number train examples, number val examples)
    :return torch of size (number train examples, number val examples)'''
    predictive_performance = predictive_performance.reshape(1, predictive_performance.shape[0])
    repeated_pred_perf = predictive_performance.repeat_interleave(visual_similarity_scores.shape[0], dim=0)
    assert(visual_similarity_scores.shape == label_similarity_scores.shape == repeated_pred_perf.shape)

    return visual_similarity_scores * label_similarity_scores * repeated_pred_perf

def sample_weights(predictive_performance, visual_similarity_scores, label_similarity_scores, r):
    '''performs the multiplication with coefficient vector r and squishes everything using sigmoid
    :param predictive_performance torch of size (number val examples)
    :param visual_similarity_scores torch of size (number train examples, number val examples)
    :param label_similarity_scores torch of size (number train examples, number val examples)
    :param r coefficient torch tensor of size (number val examples, 1)
    :returns tensor of size (number train examples, 1)'''
    similiarities = calculate_similarities(predictive_performance, visual_similarity_scores, label_similarity_scores)
    r = r.reshape(r.shape[0], 1)
    dp = torch.mm(similiarities, r)
    a = torch.sigmoid(dp)
    assert(a.shape[0]== visual_similarity_scores.shape[0])
    return a

def calc_instance_weights(input_train, target_train, input_val, target_val, val_logits, coefficient, visual_encoder):
    '''calculates the weights for each training instance with respect to the validation instances to be used for weighted
    training loss
    Args:
        input_train: (number of training images, channels, height, width)
        target_train: (number training images, 1)
        input_val: (number of validation images, channels, height, width)
        target_val:(number validation images, 1)
        model: current architecture model to calculate predictive performance (forward pass)
        coefficient: current coefficient vector of size (number train examples, 1)
        '''
    crit = nn.CrossEntropyLoss(reduction='none')
    #preds, _ = torch.max(val_logits,1)
    predictive_performance = crit(val_logits, target_val)
    vis_similarity = visual_validation_similarity(visual_encoder, input_val, input_train)
    label_similarity = measure_label_similarity(target_val, target_train)
    weights = sample_weights(predictive_performance, vis_similarity, label_similarity, coefficient)
    del crit
    gc.collect()
    torch.cuda.empty_cache()
    return weights
