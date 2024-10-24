from music_analyzer import *
from story_thinker import *
from diffuser import *
from video_merger import *
import os

class RenderManager:
    def __init__(self):
        self.music_analyzer = MusicAnalyzer(device="cuda:0")
        self.story_thinker = StoryThinker(device="cuda:1")
        self.diffuser = Diffuser(device="cuda:0")
        self.video_merger = Video_Frame_Merger()

        self.threads = []   

    def start(self):
        self.threads.append(self.music_analyzer)
        self.threads.append(self.story_thinker)
        self.threads.append(self.diffuser)
        self.threads.append(self.video_merger)

        print("Init done")

        for thread in self.threads:
            thread.daemon = True
            thread.start()

        print("Threads started")
            
        try:
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')
                print("""
                Welcome to the Render Manager CLI!
                You can check on the status of the threads by typing one of the following numbers:""")
                for i, thread in enumerate(self.threads):
                    print(f"{i+1}. {thread.name}")
                thread_number = None
                while True:
                    try:
                        thread_number = int(input("Enter the number of the thread you want to check: "))
                        if not 1 <= thread_number <= len(self.threads):
                            print("Please enter a valid number")
                        else:
                            break
                    except ValueError:
                        print("Please enter a valid number")
                thread_ref = None
                if thread_number == 1:
                    thread_ref = self.music_analyzer
                elif thread_number == 2:
                    thread_ref = self.story_thinker
                elif thread_number == 3:
                    thread_ref = self.diffuser
                elif thread_number == 4:
                    thread_ref = self.video_merger
                print(f"{thread_ref.name} is alive: {thread_ref.is_alive()}")
                print(f"{thread_ref.name} processing queue: {thread_ref.processing_queue}")
                print(f"{thread_ref.name} outputs: {thread_ref.outputs}")
                print("Press any key to continue...")
                input()

        except KeyboardInterrupt:
            print("Keyboard interrupt")

if __name__ == "__main__":
    manager = RenderManager()
    manager.start()