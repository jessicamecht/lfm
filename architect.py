""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch
import gc
from weight_samples.sample_weights import calc_instance_weights
import higher
import torch.nn.functional as F
import torch.nn as nn



class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

    def virtual_step(self, trn_X, trn_y, xi, w_optim):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        loss = self.net.loss(trn_X, trn_y) # L_trn(w)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.net.weights())

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w))

            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim, visual_encoder, coefficient_vector):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, xi, w_optim)

        # calc unrolled loss
        loss = self.v_net.loss(val_X, val_y) # L_val(w`)

        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        hessian = self.compute_hessian(dw, trn_X, trn_y)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad = da - xi*h


        model_backup = self.net.state_dict()
        w_optim_backup = w_optim.state_dict()

        #self.coefficient_vector, visual_encoder_parameters = meta_learn_new(self.net, trn_X, trn_y, val_X, val_y, coefficient_vector, visual_encoder, self.config)
        #self.visual_encoder.load_state_dict(visual_encoder_parameters)
        print('memory_allocated', torch.cuda.memory_allocated() / 1e9, 'memory_reserved',
              torch.cuda.memory_reserved() / 1e9)
        meta_learn(self.net, w_optim, trn_X, trn_y, val_X, val_y, coefficient_vector, visual_encoder)
        print('memory_allocated1', torch.cuda.memory_allocated() / 1e9, 'memory_reserved',
              torch.cuda.memory_reserved() / 1e9)


        self.net.load_state_dict(model_backup)
        w_optim.load_state_dict(w_optim_backup)

        #update_gradients(visual_encoder_gradients, coeff_vector_gradients, visual_encoder, coefficient_vector)

    def compute_hessian(self, dw, trn_X, trn_y):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_pos = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_neg = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian

def meta_learn(model, optimizer, input, target, input_val, target_val, coefficient_vector, visual_encoder):
    '''Method to meta learn the visual encoder weights and coefficient vector r, we use the higher library to be
     able to optimize through the validation loss because pytorch does not allow parameters to have grad_fn's

    Calculates the weighted training loss and performs a weight update, then calculates the validation loss and makes
    an update of the weights of the visual encoder and coefficient vector

    V' <- V - eps * d L_{Val}/dV
    r' <- r - gamma * d L_{Val}/dr

    Args:
        model: current network architecture model
        optimizer: weight optimizer for model
        input: training input of size (number of training images, channels, height, width)
        target: training target of size (number train examples, 1)
        input_val: validation input of size (number of validation images, channels, height, width)
        target_val: validation target of size (number val examples, 1)
        coefficient_vector: Tensor of size (number train examples, 1)
        visual_encoder: Visual encoder neural network to calculate instance weights
        eps: Float learning rate for visual encoder
        gamma: Float learning rate for coefficient vector
        '''
    with torch.no_grad():
        logits_val = model(input_val)

    with torch.backends.cudnn.flags(enabled=False):
        with higher.innerloop_ctx(model, optimizer, copy_initial_weights=False, track_higher_grads=True,
                                  device='cuda') as (fmodel, foptimizer):
            # functional version of model allows gradient propagation through parameters of a model
            logits = fmodel(input)

            weights = calc_instance_weights(input, target, input_val, target_val, logits_val, coefficient_vector, visual_encoder)
            loss = F.cross_entropy(logits, target, reduction='none')
            weighted_training_loss = torch.mean(weights * loss)
            foptimizer.step(weighted_training_loss)  # replaces gradients with respect to model weights -> w2

            logits_val = fmodel(input_val)
            meta_val_loss = F.cross_entropy(logits_val, target_val)
            meta_val_loss.backward()

            #coeff_vector_gradients = torch.autograd.grad(meta_val_loss, coefficient_vector, retain_graph=True)
            #coeff_vector_gradients = coeff_vector_gradients[0].detach()
            #visual_encoder_gradients = torch.autograd.grad(meta_val_loss, visual_encoder.parameters())
            #visual_encoder_gradients = (visual_encoder_gradients[0].detach(), visual_encoder_gradients[1].detach())# equivalent to backward for given parameters
            logits.detach()
            weighted_training_loss.detach()
        del logits, meta_val_loss, foptimizer, fmodel, weighted_training_loss, logits_val, weights,
        gc.collect()
        torch.cuda.empty_cache()
    #return visual_encoder_gradients, coeff_vector_gradients


def meta_learn_new(model, input, target, input_val, target_val, coefficient_vector, visual_encoder, config):
    device = 'cpu'
    with torch.no_grad():
        logits_val = model(input_val).to(device)



    model = model.to(device)
    input = input.to(device)
    target = target.to(device)
    input_val = input_val.to(device)
    target_val = target_val.to(device)
    coefficient_vector = torch.nn.Parameter(torch.tensor(coefficient_vector, requires_grad=True).to(device))
    visual_encoder = visual_encoder.to(device)

    optimizer = torch.optim.SGD(model.parameters(), config.w_lr, momentum=config.w_momentum,
                                weight_decay=config.w_weight_decay)

    visual_encoder_optimizer = torch.optim.Adam(visual_encoder.parameters(), betas=(0.5, 0.999),
                                                weight_decay=config.alpha_weight_decay)

    coeff_vector_optimizer = torch.optim.Adam([coefficient_vector], betas=(0.5, 0.999),
                                              weight_decay=config.alpha_weight_decay)

    visual_encoder_optimizer.zero_grad()
    coeff_vector_optimizer.zero_grad()
    with torch.backends.cudnn.flags(enabled=False):
        with higher.innerloop_ctx(model, optimizer, copy_initial_weights=True, track_higher_grads=True,
                                  device=device) as (fmodel, foptimizer):
            # functional version of model allows gradient propagation through parameters of a model
            logits = fmodel(input)

            weights = calc_instance_weights(input, target, input_val, target_val, logits_val, coefficient_vector,
                                            visual_encoder)
            loss = F.cross_entropy(logits, target, reduction='none')
            weighted_training_loss = torch.mean(weights * loss)
            foptimizer.step(weighted_training_loss)  # replaces gradients with respect to model weights -> w2

            logits_val = fmodel(input_val)
            meta_val_loss = F.cross_entropy(logits_val, target_val)
            meta_val_loss.backward()
            visual_encoder_optimizer.step()
            coeff_vector_optimizer.step()

            logits.detach()
            meta_val_loss.detach()
            loss.detach()
            weighted_training_loss.detach()

        del logits, meta_val_loss, foptimizer, fmodel, weighted_training_loss, logits_val, weights,
        gc.collect()
        torch.cuda.empty_cache()
        return coefficient_vector, visual_encoder.state_dict()


def update_gradients(visual_encoder_gradients, coeff_vector_gradients, visual_encoder, coefficient_vector):
    # Update the visual encoder weights
    with torch.no_grad():
        for p, grad in zip(visual_encoder.parameters(), visual_encoder_gradients):
            if p.grad is not None:
                p.grad.data += grad.detach()
            else:
                p.grad = grad.detach()
        # Update the coefficient vector
        for p, grad in zip(coefficient_vector, coeff_vector_gradients):
            if p.grad is not None:
                p.grad += grad.detach()
            else:
                p.grad = grad.detach()


