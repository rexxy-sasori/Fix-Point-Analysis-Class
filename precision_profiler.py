import numpy as np
import torch
from torch import nn
from tqdm import tqdm


class GradCollect:
    inputs_grad_collect = []
    weight_grad_collect = []
    bias_grad_collect = []

    weight_range_collect = []
    bias_range_collect = []
    activation_range_collect = []
    activation_range_collect_temp = []

    NUM_COMPUTE_LAYERS = 0
    @classmethod
    def retain_model_grad(cls, model):
        def retain_module_grad(nn_module):
            if isinstance(nn_module, nn.Conv2d) or isinstance(nn_module, nn.Linear):
                print('registering weight parameters of {} layer'.format(nn_module._get_name()))
                nn_module.weight.requires_grad_(True)
                weight_stats = np.power(2.0, np.ceil(np.log2(np.amax(np.absolute(nn_module.weight.data.cpu().numpy())))))
                GradCollect.weight_grad_collect.append((nn_module._get_name(), nn_module.weight))
                GradCollect.weight_range_collect.append((nn_module._get_name(), weight_stats))

        model.apply(retain_module_grad)
        GradCollect.NUM_COMPUTE_LAYERS = len(GradCollect.weight_grad_collect)
        GradCollect.activation_range_collect = np.zeros(GradCollect.NUM_COMPUTE_LAYERS)

    @classmethod
    def retain_inputs_grad(cls, model):
        def retain_nn_module_inputs(m):
            def retain_inputs(m, x):
                x = x[0]
                x = x.requires_grad_(True)
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    stats = np.power(2.0,np.ceil(np.log2(np.amax(np.absolute(x.detach().cpu().numpy())))))
                    GradCollect.inputs_grad_collect.append((m._get_name(), x))
                    GradCollect.activation_range_collect_temp.append((m._get_name(), stats))
                    x.retain_grad()

            m.register_forward_pre_hook(retain_inputs)

        model.apply(retain_nn_module_inputs)


def get_noise_gains(model, data_loader, device):
    weight_gains = [0] * GradCollect.NUM_COMPUTE_LAYERS
    activation_gains = [0] * GradCollect.NUM_COMPUTE_LAYERS
    data_size = len(data_loader)
    first_time = True
    for inputs, targets in tqdm(data_loader):
        inputs = inputs.to(device).requires_grad_(True)
        GradCollect.inputs_grad_collect = []
        GradCollect.activation_range_collect_temp = []
        inputs.retain_grad()
        outputs = model(inputs)
        outputs = outputs.sum(0)

        GradCollect.activation_range_collect = np.vstack([
            np.array([num for _, num in GradCollect.activation_range_collect_temp]),
            GradCollect.activation_range_collect]
        ).max(0)

        Z_fl, Y_fl = outputs.max(0)
        num_lbls = outputs.size(0)
        for i in range(num_lbls):
            if i != Y_fl:
                output_difference = Z_fl - outputs[i]
                output_difference.backward(retain_graph=True)
                with torch.no_grad():
                    denominator = 24 * (output_difference ** 2)
                    for idx in range(GradCollect.NUM_COMPUTE_LAYERS):
                        weight = GradCollect.weight_grad_collect[idx][1]
                        weight_grad = weight.grad
                        if first_time:
                            weight_gains[idx] = (weight_grad ** 2).sum() / denominator
                        else:
                            weight_gains[idx].add_((weight_grad ** 2).sum() / denominator)

                        weight.grad.zero_()

                    for idx, (module_name, activations) in enumerate(GradCollect.inputs_grad_collect):
                        grad = activations.grad
                        if first_time:
                            activation_gains[idx] = (grad ** 2).sum() / denominator
                        else:
                            activation_gains[idx].add_((grad ** 2).sum() / denominator)

                        activations.grad.zero_()

                    first_time = False

    for idx in range(GradCollect.NUM_COMPUTE_LAYERS):
        activation_gains[idx] = activation_gains[idx].cpu().numpy() / data_size
        weight_gains[idx] = weight_gains[idx].cpu().numpy() / data_size

    return weight_gains, activation_gains


def get_normalized_noise_gains(wg_coarse, ag_coarse):
    adjusted_wg_noise_gains = np.zeros(GradCollect.NUM_COMPUTE_LAYERS)
    adjusted_ag_noise_gains = np.zeros(GradCollect.NUM_COMPUTE_LAYERS)

    for l in range(GradCollect.NUM_COMPUTE_LAYERS):
        adjusted_wg_noise_gains[l] = wg_coarse[l] * np.square(GradCollect.weight_range_collect[l][1])
        adjusted_ag_noise_gains[l] = ag_coarse[l] * np.square(GradCollect.activation_range_collect[l])

    min_ag = adjusted_ag_noise_gains.min()
    min_wg = adjusted_wg_noise_gains.min()

    least_gain = min(min_ag,min_wg)
    return adjusted_wg_noise_gains, adjusted_ag_noise_gains, least_gain


def get_precision_offsets(wg, ag, least_gain):
    w_offsets = np.zeros(GradCollect.NUM_COMPUTE_LAYERS)
    a_offsets = np.zeros(GradCollect.NUM_COMPUTE_LAYERS)

    for l in range(GradCollect.NUM_COMPUTE_LAYERS):
        w_offsets[l] = np.round(0.5 * np.log2(wg[l] / least_gain))
        a_offsets[l] = np.round(0.5 * np.log2(ag[l] / least_gain))

    return w_offsets, a_offsets