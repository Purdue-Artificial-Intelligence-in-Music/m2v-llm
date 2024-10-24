import threading
import llama
import time
import torchaudio
import torch
import numpy as np
import regex as re
import os
from history_list import HistoryList

class MusicAnalyzer(threading.Thread):
    def __init__(self, 
                 prompt_path = "prompt_list.txt",
                 output_path = "./music_hlst",
                 preamble_path = "",
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 audio_weight=1.5, 
                 seconds_used=20, 
                 seconds_jump=5,
                 cache_size=100,
                 cache_t=20,
                 cache_weight=0.0,
                 max_gen_len=512,
                 gen_t=0.6,
                 top_p=0.8,
                 truncate_music=False, 
                 debug_print=False):
        threading.Thread.__init__(self)
        self.device = device
        self.name = "Music Analyzer"

        with open(prompt_path, 'r') as f:
            self.prompt_list = f.readlines()

        if preamble_path is not None and preamble_path != "":
            with open(preamble_path, 'r') as f:
                self.preamble = f.read()
                self.preamble = self.preamble.replace("\n", "")
        else:
            self.preamble = ""

        self.processing_queue = []
        self.outputs = []
        self.output_path = output_path

        # self.stop_request = False
        self.sleep_delay = 1.0

        self.audio_weight = audio_weight
        self.seconds_used = seconds_used
        self.seconds_jump = seconds_jump

        self.SR = 24000
        self.SAMPLES_JUMP = int(max(self.seconds_jump * self.SR, 1))
        self.SAMPLES_USED = int(max(self.seconds_used * self.SR, 1))

        self.cache_size = cache_size
        self.cache_t = cache_t
        self.cache_weight = cache_weight
        self.max_gen_len = max_gen_len
        self.gen_t = gen_t
        self.top_p = top_p

        self.truncate_music = truncate_music
        self.debug_print = debug_print

        self.mullama_model = llama.load("./ckpts/checkpoint.pth", "./ckpts/LLaMA", mert_path="m-a-p/MERT-v1-330M", knn=True, knn_dir="./ckpts", llama_type="7B")
        self.mullama_model.eval()

        print("MusicAnalyzer initialized")


    def process_chunk(self, audio_chunk):
        inputs = {}
        inputs['Audio'] = [audio_chunk, self.audio_weight]  # Audio is [samples, weight]

        h_list = HistoryList()
        
        # Process audio with music encoder
        with torch.cuda.amp.autocast():
            audio_query = self.mullama_model.forward_audio(inputs, self.cache_size, self.cache_t, self.cache_weight)
            aq = audio_query.clone()
            prompts = [self.mullama_model.tokenizer.encode(x, bos=True, eos=False) for x in ["Listen to this music.\n"]]
            _ = self.mullama_model.generate_with_audio_query(aq, prompts, max_gen_len=self.max_gen_len, temperature=self.gen_t, top_p=self.top_p)

        # Music QA - store in h_list
        for prompt in self.prompt_list:
            if self.preamble != "":
                total_prompt = self.preamble + " "
            else:
                total_prompt = ""
            total_prompt += prompt

            prompts = [total_prompt]

            prompts = [self.mullama_model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

            aq = audio_query.clone()
            with torch.cuda.amp.autocast():
                result = self.mullama_model.generate_with_audio_query(aq, prompts, max_gen_len=self.max_gen_len, temperature=self.gen_t, top_p=self.top_p)

            if type(result) == list:
                result = result[0]

            result = re.sub(r'[^A-Za-z0-9.,!?;:()\[\]{}<>/@#&%*\-+=_\s]', '', result)
            result = result[:min(len(result), 200)]

            while result != "" and result[0] == " ":
                result = result[1:]

            if ":" in result:
                result = result.split(":")[1]

            if self.debug_print:
                print("Q:", total_prompt.replace("\n", ""))
                print("A:\x1B[3m", result.replace("\n", ""), "\x1B[0m")

            h_list.append(total_prompt.replace("\n", ""), result.replace("\n", ""))
        
        return h_list

    def preprocess_audio(self, audio_path):
        audio, sr = torchaudio.load(audio_path)
        if sr != self.SR:
            waveform = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.SR)
            sr = self.SR
        else:
            waveform = audio
        waveform = torch.mean(waveform, np.argmin([len(waveform), len(waveform[0])])) if len(waveform.shape) > 1 else waveform
        if self.truncate_music:
            waveform = waveform[:min(len(waveform), self.SAMPLES_USED+3*self.SAMPLES_JUMP+10)]
        waveform = torch.reshape(waveform, (1, -1))
        AUDIO_LEN = waveform.shape[1]
        audio = waveform
        if self.debug_print:
            print(f"Audio shape: {audio.shape}")
        
        return audio, AUDIO_LEN

    def process_file(self, audio_path, output_name):
        curr_sample = 0
        i = 1

        if os.path.isfile(os.path.join(self.output_path, f"{output_name}_ma_chunk_{i}.hlst")):
            return

        try:

            audio, AUDIO_LEN = self.preprocess_audio(audio_path)
            
            break_at_end = False
            
            while not break_at_end:
                # Calculate sample ranges and set up for processing
                end_sample = curr_sample + self.SAMPLES_USED
                if end_sample > AUDIO_LEN:
                    end_sample = AUDIO_LEN
                    break_at_end = True
                if self.debug_print:
                    print(f"----------------- Processing chunk {i}, samples [{curr_sample}, {end_sample}]\n")
                
                h_list = self.process_chunk(audio[:, curr_sample : end_sample])

                h_list.save(os.path.join(self.output_path, f"{output_name}_ma_chunk_{i}.hlst"))

                self.outputs.append((self.output_path, output_name, i))

                if self.debug_print:
                    print()
                    print()

                curr_sample += self.SAMPLES_JUMP
                i += 1
        except Exception as e:
            print("Error processing file", audio_path)
            print(e)

    def run(self):
        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)
        while not self.stop_request:
            if len(self.processing_queue) > 0:
                audio_path, name = self.processing_queue.pop(0)
                self.process_file(audio_path, name)
            time.sleep(self.sleep_delay)