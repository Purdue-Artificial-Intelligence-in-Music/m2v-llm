import torch
import transformers
import unittest
import numpy as np

from MERT_model import *
from MusicFM_model import *

SAMPLE_RATE = 16000
NUM_ITERS = 5

class model_sanity_checks(unittest.TestCase):
    def test_MERT_model(self):
        for i in range(NUM_ITERS):
            hidden_size = np.random.randint(50,400)
            temporal_resolution = np.random.randint(10,100)
            batch_size = np.random.randint(1,10)
            SAMPLE_RATE = 16000
            model = MERT_model(hidden_size=hidden_size, temporal_resolution=temporal_resolution)
            input_audio = torch.randn(batch_size, SAMPLE_RATE)
            output = model(input_audio, SAMPLE_RATE)
            assert output.shape[0] == batch_size
            assert output.shape[2] == hidden_size
            assert np.abs(output.shape[1] - SAMPLE_RATE / temporal_resolution) <= 2

    def test_MusicFM_model(self):
        for i in range(NUM_ITERS):
            hidden_size = np.random.randint(50,400)
            temporal_resolution = np.random.randint(10,100)
            batch_size = np.random.randint(1,10)
            SAMPLE_RATE = 16000
            model = MusicFM_model(hidden_size=hidden_size, temporal_resolution=temporal_resolution)
            input_audio = torch.randn(batch_size, SAMPLE_RATE)
            output = model(input_audio, SAMPLE_RATE)
            assert output.shape[0] == batch_size
            assert output.shape[2] == hidden_size
            assert np.abs(output.shape[1] - SAMPLE_RATE / temporal_resolution) <= 2