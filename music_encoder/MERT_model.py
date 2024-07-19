import torchaudio
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
from adapter_model import *


class MERT_model(torch.nn.Module):
    '''
    This class contains a MERT feature extractor, with reprojection to a desired size.

    Input: input audio as a torch tensor of shape (..., length).
    Output: a sequence of hidden states of dimension (..., length, hidden_size = 128).

    Parameters:
    - hidden_size: the size of the hidden states
    - resample: whether to resample the input audio to the model's sample rate
    - device: the device to run the model on

    https://huggingface.co/m-a-p/MERT-v0

    '''
    def __init__(self, hidden_size=128, resample: bool = True, device: str = "cuda:0"):
        super(MERT_model, self).__init__()
        self.hidden_size = hidden_size
        self.model_string = 'm-a-p/MERT-v1-95M'
        self.device = device

        self.resampler = None

        self.model = AutoModel.from_pretrained(self.model_string, trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_string, trust_remote_code=True)
        self.adapter = Adapter_Model(75, 75, 768, self.hidden_size)

        self.resample_rate = self.processor.sampling_rate
        self.resample = resample

        # self.to(device)
    
    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        self.processor = self.processor.to(device)
        self.adapter = self.adapter.to(device)
        return self
    
    def forward(self, input_audio, sampling_rate: int):
        if not input_audio.device == self.device:
            input_audio = input_audio.to(self.device)
            print("Input audio tensor not on device")
        # Resample audio

        if self.resample_rate and sampling_rate is not None and self.resample_rate != sampling_rate:
            print(f'setting rate from {sampling_rate} to {self.resample_rate}')
            shape = input_audio.shape
            self.resampler = torchaudio.transforms.Resample(sampling_rate, self.resample_rate)
            self.resampler.to(self.device)
            input_audio = self.resampler(input_audio)
            input_audio = input_audio.reshape(*shape[:-1], -1)
            sampling_rate = self.resample_rate
        
        # Process input audio
        inputs = self.processor(input_audio, sampling_rate=sampling_rate, return_tensors="pt")
        # Get the hidden states from the MERT model
        hidden_states = self.model(**inputs, output_hidden_states=True).hidden_states
        hidden_states = torch.stack(list(hidden_states), dim=0).squeeze()[-1]
        print(hidden_states)
        print(hidden_states.shape)
        # Pass the hidden states through the linear layer
        output = self.adapter(hidden_states)
        print(output.shape)
        return output