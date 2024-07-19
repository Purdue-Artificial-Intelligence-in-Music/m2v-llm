import torchaudio
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
from adapter_model import *
from LoRA_funcs import *
import peft


class MERT_model(torch.nn.Module):
    '''
    This class contains a MERT feature extractor, with reprojection to a desired size.

    Input: input audio as a torch tensor of shape (..., length).
    Output: a sequence of hidden states of dimension (..., length, hidden_size = 128).

    Parameters:
    - hidden_size: the size of the hidden states
    - temporal_resolution: the temporal resolution of the model (in Hz)
    - resample: whether to resample the input audio to the model's sample rate
    - device: the device to run the model on

    https://huggingface.co/m-a-p/MERT-v1-95M

    '''
    def __init__(self, hidden_size=128, temporal_resolution: int = 75, resample: bool = True, device: str = "cuda:0"):
        super(MERT_model, self).__init__()
        self.hidden_size = hidden_size
        self.model_string = 'm-a-p/MERT-v1-95M'
        self.temporal_resolution = temporal_resolution
        self.device = device

        self.resampler = None

        self.model = AutoModel.from_pretrained(self.model_string, trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_string, trust_remote_code=True)
        self.adapter = Adapter_Model(75, self.temporal_resolution, 768, self.hidden_size)

        self.resample_rate = self.processor.sampling_rate
        self.resample = resample

        self.to(device)
    
    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        # self.processor = self.processor.to(device)
        self.adapter = self.adapter.to(device)
        return self
    
    def get_LoRA_model(self):
        '''
        Returns the LoRA-trainable version of the model.
        '''
        trainable_modules, adapter_modules = get_LoRA_trainable_modules(self, separate_out_adapters=True)
        config = LoraConfig(
            target_modules=trainable_modules,
            modules_to_save=adapter_modules,
            bias='lora_only',
            use_rslora=True,
        )
        return peft.get_peft_model(self, config)
    
    def forward_single_batch(self, input_audio, sampling_rate: int):
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
        inputs = self.processor(input_audio, sampling_rate=sampling_rate, return_tensors="pt").to(self.device)
        # Get the hidden states from the MERT model
        hidden_states = self.model(**inputs, output_hidden_states=True).hidden_states
        hidden_states = torch.stack(list(hidden_states), dim=0).squeeze()[-1]
        # Pass the hidden states through the linear layer
        output = self.adapter(hidden_states)
        return output
    
    def forward(self, input_audio, sampling_rate: int):
        if not input_audio.device == self.device:
            input_audio = input_audio.to(self.device)
            print("Input audio tensor not on device")
        if len(input_audio.shape) > 1:
            old_shape = input_audio.shape
            input_audio = input_audio.reshape(-1, input_audio.shape[-1])
            if input_audio.shape[0] == 1:
                input_audio = input_audio.squeeze()
                return self.forward_single_batch(input_audio, sampling_rate)
            first_batch = self.forward_single_batch(input_audio[0], sampling_rate)
            output_tensor = torch.empty((input_audio.shape[0], *first_batch.shape), device=first_batch.device, dtype=first_batch.dtype)
            output_tensor[0] = first_batch
            for i in range(1, input_audio.shape[0]):
                output_tensor[i] = self.forward_single_batch(input_audio[i], sampling_rate)
            return output_tensor.reshape(*old_shape[:-1], *output_tensor[0].shape)
        else:
            return self.forward_single_batch(input_audio, sampling_rate)