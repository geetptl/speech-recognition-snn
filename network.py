import torch
import torch.nn as nn


class PseudoSpikeRect(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, vth, grad_win, grad_amp):
        ctx.save_for_backward(input)
        ctx.vth = vth
        ctx.grad_win = grad_win
        ctx.grad_amp = grad_amp

        output = torch.gt(input, ctx.vth).float()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        vth = ctx.vth
        grad_win = ctx.grad_win
        grad_amp = ctx.grad_amp
        grad_input = grad_output.clone()

        spike_pseudo_grad = torch.le(abs(input - vth), grad_win)

        grad = grad_amp * grad_input * spike_pseudo_grad.float()
        return grad, None, None, None


class LinearIFCell(nn.Module):

    def __init__(self, psp_func, pseudo_grad_ops, param):
        super(LinearIFCell, self).__init__()
        self.psp_func = psp_func
        self.pseudo_grad_ops = pseudo_grad_ops
        self.vdecay, self.vth, self.grad_win, self.grad_amp = param

    def forward(self, input_data, state):
        pre_spike, pre_volt = state

        volt = pre_volt * self.vdecay * (1 - pre_spike) + self.psp_func(input_data)

        output = self.pseudo_grad_ops(volt, self.vth, self.grad_win, self.grad_amp)

        return output, (output, volt)


class SingleHiddenLayerSNN(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, param_dict):
        super(SingleHiddenLayerSNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        pseudo_grad_ops = PseudoSpikeRect.apply

        self.hidden_cell = LinearIFCell(
            nn.Linear(self.input_dim, self.hidden_dim, bias=False), pseudo_grad_ops, param_dict["hid_layer"]
        )

        self.output_cell = LinearIFCell(
            nn.Linear(self.hidden_dim, self.output_dim, bias=False), pseudo_grad_ops, param_dict["out_layer"]
        )

    def forward(self, spike_data, init_states_dict, batch_size, spike_ts):
        hidden_state, out_state = init_states_dict["hid_layer"], init_states_dict["out_layer"]
        spike_data_flatten = spike_data.view(batch_size, self.input_dim, spike_ts)
        output_list = []
        for tt in range(spike_ts):
            s_input = spike_data_flatten[:, :, tt]

            hlayer_spikes, hidden_state = self.hidden_cell.forward(s_input, hidden_state)

            olayer_spikes, out_state = self.output_cell.forward(hlayer_spikes, out_state)

            output_list.append(olayer_spikes)

        output = torch.sum(torch.stack(output_list, dim=0), dim=0)

        return output


class WrapSNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, param_dict, device):
        super(WrapSNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.snn = SingleHiddenLayerSNN(input_dim, output_dim, hidden_dim, param_dict)

    def forward(self, spike_data):
        batch_size = spike_data.shape[0]
        spike_ts = spike_data.shape[-1]
        init_states_dict = {}
        # Hidden layer
        hidden_volt = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        hidden_spike = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        init_states_dict["hid_layer"] = (hidden_spike, hidden_volt)
        # Output layer
        out_volt = torch.zeros(batch_size, self.output_dim, device=self.device)
        out_spike = torch.zeros(batch_size, self.output_dim, device=self.device)
        init_states_dict["out_layer"] = (out_spike, out_volt)
        # SNN
        output = self.snn(spike_data, init_states_dict, batch_size, spike_ts)
        return output


def img_2_event_img(image, device, spike_ts):
    batch_size = image.shape[0]
    channel_size = image.shape[1]
    image_size_d1 = image.shape[2]
    image_size_d2 = image.shape[3]
    image = image.view(batch_size, channel_size, image_size_d1, image_size_d2, 1)

    random_image = torch.rand((batch_size, channel_size, image_size_d1, image_size_d2, spike_ts), device=device)
    event_image = torch.gt(image.to(device), random_image).float()

    return event_image
