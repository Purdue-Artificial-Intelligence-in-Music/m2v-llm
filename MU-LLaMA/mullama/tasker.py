import threading
import time

class Tasker(threading.Thread):
    def __init__(self, tasks):
        threading.Thread.__init__(self)
        self.tasks = tasks

    def run(self):
        while True:
            for i in range(len(self.tasks) - 1):
                first_task = self.tasks[i]
                second_task = self.tasks[i+1]
                while len(first_task.outputs) > 0:
                    output = first_task.outputs.pop(0)
                    print(f"Merging {first_task.name} output of {output} into {second_task.name}'s queue")
                    second_task.processing_queue.append(output)
            time.sleep(0.5)
            