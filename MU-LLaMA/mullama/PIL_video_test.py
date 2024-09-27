import cv2
from PIL import Image
import numpy as np

FPS = 30
SR = 24000

videodims = (512,512)

def interp_pipe(image1, image2, length, output_frame_rate = FPS):
    output_frames = []
    for i in range(int(output_frame_rate * length)):
        output_frames.append(image1)
    return output_frames

output_images = []

for i in range(150):
    output_images.append(Image.new('RGB', videodims, color = 'darkred'))

output_video_frames = []
for i in range(len(output_images) - 1):
    output_video_frames.extend(interp_pipe(output_images[i], output_images[i + 1], 1/29.97))

videodims = output_video_frames[0].size
fourcc = cv2.VideoWriter_fourcc(*"mp4v")    
video = cv2.VideoWriter("test.mp4", fourcc, FPS, videodims)
for frame in output_video_frames:
    imtemp = frame.copy()
    video.write(cv2.cvtColor(np.array(imtemp), cv2.COLOR_RGB2BGR))
video.release()