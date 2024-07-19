import torch
import numpy as np


class Adapter_Model(torch.nn.Module):
    '''
    This is a class containing a simple adapter model.

    The model structure is an AdaptiveAvgPool1D for changing time-scale resolution followed by a linear + ReLU layer to project to a lower-dim space.

    Input: a sequence of hidden states of dimension (..., length, size)
    Output: a sequence of hidden states of dimension (..., length, size)

    Parameters:
    - input_sample_hz: the sample rate of the input features
    - output_sample_hz: the sample rate of the output features
    - input_size: the input feature size
    - output_size: the output feature size
    - intermediate_size: the size of the intermediate layer

    '''
    def __init__(self, input_sample_hz, output_sample_hz, input_size, output_size, intermediate_size: int = 0):
        super(Adapter_Model, self).__init__()
        self.input_sample_hz = input_sample_hz
        self.output_sample_hz = output_sample_hz
        self.input_size = input_size
        self.intermediate_size = intermediate_size
        if self.intermediate_size <= 0:
            self.intermediate_size = int(np.power(np.e, (0.5 * np.log(input_size * output_size))))
        self.output_size = output_size

        self.adapter = torch.nn.Sequential(torch.nn.Linear(self.input_size, self.intermediate_size),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(self.intermediate_size, self.output_size),)
        
    def to(self, device):
        self.adapter.to(device)
        return self
    
    def forward(self, input_tensor):
        n_frame = int(input_tensor.shape[-2] * float(self.output_sample_hz)/ self.input_sample_hz)
        input_tensor = input_tensor.swapaxes(-1, -2)
        input_tensor = torch.nn.AdaptiveAvgPool1d(n_frame)(input_tensor)
        input_tensor = input_tensor.swapaxes(-1, -2)
        return self.adapter(input_tensor)