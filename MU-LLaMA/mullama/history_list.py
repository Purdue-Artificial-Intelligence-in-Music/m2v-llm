import pickle

class HistoryList:
    def __init__(self, name="", description=""):
        self.history = []
        self.history_index = 0
        self.name = name
        self.description = description
    
    def append(self, prompt, result):
        self.history.append([prompt, result])
        self.history_index += 1

    def undo(self):
        if self.history_index >= 0:
            self.history_index -= 1

    def get_list(self):
        return self.history

    def get(self, idx):
        return self.history[idx]
    
    def get_current(self):
        return self.history[-1]
    
    def get_length(self):
        return len(self.history)
    
    def write(self):
        with open("output_video_prompts.txt", 'w') as f:
            for value in self.history:
                assert type(value[0]) == str and type(value[1]) == str
                f.write("-------------------\nOur question:\n")
                f.write(value[0])
                f.write("\nMU-LLaMA's answer:\n")
                f.write(value[1])
                f.write("\n\n")

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.history, f)
    
    def load(self, filename):
        with open(filename, 'rb') as f:
            self.history = pickle.load(f)
            self.history_index = len(self.history) - 1