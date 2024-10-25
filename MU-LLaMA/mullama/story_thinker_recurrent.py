import threading
import time
import torch
import regex as re
from history_list import HistoryList
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

class StoryThinker(threading.Thread):
    def __init__(self, 
                 output_dir=None,
                 summarize_path="summarize_v3.txt",
                 ecphory_path="ecphory.txt",
                 sd_summary_path="sd_summary.txt",
                 hf_token_path="hf_token.txt",
                 model_id = "meta-llama/Llama-2-7b-hf",
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 debug_print=False):
        threading.Thread.__init__(self)
        self.stop_request = False
        self.device = device
        self.name = "Story Thinker"
        # self.stop_request = False
        self.sleep_delay = 1.0
        self.debug_print = debug_print
        self.output_dir = output_dir

        self.processing_queue = []
        self.outputs = []
        
        if not os.path.isfile('hf_token.txt'):
            raise Exception("Make sure hf_token.txt exists on your system, and is a text file containing a HuggingFace token")
        with open(hf_token_path, 'r') as f:
            self.hf_token = f.readlines()[0]
        with open(summarize_path, 'r') as f:
            self.summarize_preamble = f.read()
            self.summarize_preamble = self.summarize_preamble.replace("\n", "")
        with open(ecphory_path, 'r') as f:
            self.ecphory = f.read()
            self.ecphory = self.ecphory.replace("\n", "")
        with open(sd_summary_path, 'r') as f:
            self.sd_summary = f.read()
            self.sd_summary = self.sd_summary.replace("\n", "")

        self.llama_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda:1", use_auth_token=self.hf_token)
        self.llama_model = self.llama_model.to(self.device)
        self.llama_tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=self.hf_token)
        self.llama_tokenizer.use_default_system_prompt = False

        print("StoryThinker initialized")
        

    def llama_inference(self,
                        prompt,
                        max_length=4096,
                        ):   
        if len(prompt) > max_length:
            print(f"Warning: input prompt for LLaMA inference is longer than {max_length} chars, truncating")
            print("The prompt is: ", prompt)
            prompt = prompt[:max_length]
        input_ids = self.llama_tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.llama_model.device)
        output = self.llama_model.generate(input_ids, max_length=max_length, num_beams=4, no_repeat_ngram_size=2, temperature=1)
        response = self.llama_tokenizer.decode(output[0], skip_special_tokens=True)

        response = re.sub(r'[^A-Za-z0-9.,!?;:()\'\"\[\]{}<>/@#&%*\-+=_\s]', '', response)

        return response


    def summarize_convo(self,
                        history_list,
                        summarize_preamble = "Summarize the following conversation. The conversation starts now. ",
                        explicit_labels = False,
                        debug_print = True,
                        combine_all = False,
                        ):
        # Formatting prompt - make sure to let LLaMA think at the end
        total_prompt = summarize_preamble
        if combine_all:
            total_prompt += "\""
        for history in history_list:
            if combine_all:
                total_prompt += history[0] + " " + history[1] + " "
            elif explicit_labels:
                total_prompt += "We asked: \"" + history[0] + "\" "
                if history[1] is not None:
                    total_prompt += "You replied: \"" + history[1] + "\" "
            else:
                total_prompt += "\"" + history[0] + "\" "
                if history[1] is not None:
                    total_prompt += "\"" + history[1] + "\" "
        if combine_all:
            total_prompt += "\" "

        total_prompt += "Your summary is: \""
        total_prompt = total_prompt.replace("\n", "")

        out = self.llama_inference(total_prompt)
        try:   
            out = out.split("Your summary is: \"")[1]
            out = out.split("\"")[0]

            if re.sub(r'[^A-Za-z0-9]', '', out) == "":
                print("Warning: LLaMA returned an empty summary, returning history_list[0][1] as a placeholder")
                out = history_list[0][1]    
        except Exception:
            print("Error: LLaMA failed to follow the prompt guidelines. Here is the response it gave: ", out)
            print("Returning history_list[0][1] as a placeholder")
            out = history_list[0][1]  

        out = out[:min(len(out), 500)]

        return out
    
    def ecphory_call(self, music_info):
        ecphory = self.llama_inference(self.ecphory + "\"" + music_info + "\" Your idea is: \"", max_length=2500)
        ecphory = ecphory.split("Your idea is: \"")[1]
        ecphory = ecphory.split("\"")[0]
        if self.debug_print:
            print("Metachory:\x1B[3m", ecphory, "\x1B[0m")
        return ecphory
    
    def sd_summarize_call(self, ecphory, music_info):
        prev_h_list = HistoryList()
        prev_h_list.load(os.path.join(prompts_folder, f"{prompts_name}_ma_chunk_{chunk}.hlst"))

        prompt = ""
        for item in h_list.get_list():
            if item[0] == "SD Prompt:":
                prompt = item[1]
                break
        
        if prompt == "":
            if self.debug_print:
                print("No SD prompt found in history list, skipping")
            return

        full_info = ecphory + " " + music_info
        sd_prompt = self.llama_inference(self.sd_summary + "\"" + full_info + "\" The previous prompt starts now. \"" + prompt + "\" Your prompt is: \"", max_length=2500)
        sd_prompt = sd_prompt.split("Your prompt is: \"")[1]
        sd_prompt = sd_prompt.split("\"")[0]
        sd_prompt = sd_prompt[:100]
        if re.sub(r'[^A-Za-z0-9]', '', sd_prompt) == "":
            print("Warning: LLaMA returned an empty Stable Diffusion prompt, returning full_info as a placeholder")
            sd_prompt = full_info
        sd_prompt = sd_prompt[:min(len(sd_prompt), 500)]
        if self.debug_print:
            print("SD Prompt:\x1B[3m", sd_prompt, "\x1B[0m")
        return sd_prompt

    def process_file(self, prompts_folder, prompts_name, chunk):
        h_list = HistoryList()
        h_list.load(os.path.join(prompts_folder, f"{prompts_name}_ma_chunk_{chunk}.hlst"))

        music_info = None
        ecphory = None
        sd_prompt = None
        try:
            music_info = self.summarize_convo(h_list.get_list(), self.summarize_preamble, self.debug_print, combine_all=True)

            # Do fun things and create synesthesia
            ecphory = self.ecphory_call(music_info)
            ecphory = ecphory[:min(len(ecphory), 500)]

            # Summarize into Stable Diffusion format
            sd_prompt = self.sd_summarize_call(ecphory, music_info)
        
        except Exception as e:
            print("LLaMA failed to process the prompt, recovering what we can")
            print("Exception:", e)
            if not music_info:
                music_info = "No music information available"
            if not ecphory:
                ecphory = "No ecphory available"
            if not sd_prompt:
                sd_prompt = "No SD prompt available"
        
        out_list = HistoryList()
        out_list.append("Music summary:", music_info)
        out_list.append("Ecphory:", ecphory)
        out_list.append("SD Prompt:", sd_prompt)
        if self.output_dir is not None:
            out_list.save(os.path.join(self.output_dir, f"{prompts_name}_st_chunk_recurrent_{chunk}.hlst"))
            self.outputs.append((self.output_dir, prompts_name, chunk))
        else:
            out_list.save(os.path.join(prompts_folder, f"{prompts_name}_st_chunk_recurrent_{chunk}.hlst"))
            self.outputs.append((prompts_folder, prompts_name, chunk))

    def run(self):
        while not self.stop_request:
            if len(self.processing_queue) > 0:
                prompts_folder, prompts_name, chunk = self.processing_queue.pop(0)
                self.process_file(prompts_folder, prompts_name, chunk)
            time.sleep(self.sleep_delay)