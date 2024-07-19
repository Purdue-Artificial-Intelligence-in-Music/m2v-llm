import os
import torch
from musicfm.model.musicfm_25hz import MusicFM25Hz
import subprocess
from adapter_model import *
from LoRA_funcs import *
import peft

class MusicFM_model(torch.nn.Module):
    '''
    This class contains a MusicFM feature extractor with reprojection using a linear + ReLU layer.

    Input: input audio as a torch tensor of shape (..., length).
    Output: a sequence of hidden states of dimension (..., length, hidden_size = 128).

    Parameters:
    - hidden_size: the size of the hidden states
    - temporal_resolution: the temporal resolution of the model (in Hz)
    - device: the device to run the model on
    
    https://github.com/minzwon/musicfm

    '''
    def __init__(self, hidden_size=128, model_str: str="MSD", temporal_resolution: int = 25, musicfm_path: str = "", device: str = "cuda:0"):
        super(MusicFM_model, self).__init__()
        self.hidden_size = hidden_size
        self.MUSICFM_PATH = musicfm_path
        if self.MUSICFM_PATH == "":
            self.MUSICFM_PATH = os.path.join(os.path.abspath(__file__).split("MusicFM_model.py")[0], "musicfm")
        self.device = device
        self.temporal_resolution = temporal_resolution

        # Download model weights and parameters
        self.model_str = model_str
        if not model_str == "MSD" and not model_str == "FMA":
            raise ValueError("model_str must be 'MSD' or 'FMA'")
        if model_str == "MSD":
            if not os.path.isfile(os.path.join(self.MUSICFM_PATH, "data/msd_stats.json")):
                subprocess.run(["wget", "-P", os.path.join(self.MUSICFM_PATH, "data"), "https://huggingface.co/minzwon/MusicFM/resolve/main/msd_stats.json"])
            if not os.path.isfile(os.path.join(self.MUSICFM_PATH, "data/pretrained_msd.pt")):
                subprocess.run(["wget", "-P", os.path.join(self.MUSICFM_PATH, "data"), "https://huggingface.co/minzwon/MusicFM/resolve/main/pretrained_msd.pt"])
            self.musicfm = MusicFM25Hz(
                is_flash=False,
                stat_path=os.path.join(self.MUSICFM_PATH, "data", "msd_stats.json"),
                model_path=os.path.join(self.MUSICFM_PATH, "data", "pretrained_msd.pt"),
            )
        elif model_str == "FMA":
            if not os.path.isfile(os.path.join(self.MUSICFM_PATH, "data/fma_stats.json")):
                subprocess.run(["wget", "-P", os.path.join(self.MUSICFM_PATH, "data"), "https://huggingface.co/minzwon/MusicFM/resolve/main/fma_stats.json"])
            if not os.path.isfile(os.path.join(self.MUSICFM_PATH, "data/pretrained_fma.pt")):
                subprocess.run(["wget", "-P", os.path.join(self.MUSICFM_PATH, "data"), "https://huggingface.co/minzwon/MusicFM/resolve/main/pretrained_fma.pt"])
            self.musicfm = MusicFM25Hz(
                is_flash=False,
                stat_path=os.path.join(self.MUSICFM_PATH, "data", "fma_stats.json"),
                model_path=os.path.join(self.MUSICFM_PATH, "data", "pretrained_fma.pt"),
            )
        self.adapter = Adapter_Model(25, self.temporal_resolution, 1024, self.hidden_size)

        self.to(device)
    
    def to(self, device):
        self.device = device
        self.musicfm = self.musicfm.to(device)
        self.adapter = self.adapter.to(device)
        return self
    
    def get_LoRA_model(self):
        trainable_modules, adapter_modules = get_LoRA_trainable_modules(self, separate_out_adapters=True)
        config = LoraConfig(
            target_modules=trainable_modules,
            modules_to_save=adapter_modules,
            bias='lora_only',
            use_rslora=True,
        )
        return peft.get_peft_model(self, config)
    
    def forward(self, input_audio, sampling_rate: int):
        if not input_audio.device == self.device:
            input_audio = input_audio.to(self.device)
            print("Input audio tensor not on device")
        emb = self.musicfm.get_latent(input_audio, layer_ix=7)
        output = self.adapter(emb)
        return output