HEIGHT = 512
WIDTH = 512
FPS = 10
NUM_FRAMES = 10

from diffusers.utils import make_image_grid, load_image
import cv2
from torch.nn import functional as F
from PIL import Image, ImageFilter
# from train_log.RIFE_HDv3 import Model
import torch
import numpy as np
from diffusers import I2VGenXLPipeline

# def rife(img0,img1,n,model):
#   img0, img1,h,w = process(img0,img1)
#   img_list = [img0, img1]
#   for i in range(n):
#     tmp = []
#     for j in range(len(img_list) - 1):
#       mid = model.inference(img_list[j], img_list[j + 1])
#       tmp.append(img_list[j])
#       tmp.append(mid)
#     tmp.append(img1)
#     img_list = tmp
#   final=[]
#   for i in range(len(img_list)-1):
#     k= (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
#     a= Image.fromarray(k)
#     final.append(a)
#   return final

# def process(img0,img1):
#   img0 = np.array(img0)
#   img1 = np.array(img1)
#   img0 = cv2.resize(img0, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#   img1 = cv2.resize(img1, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#   img0 = (torch.tensor(img0.transpose(2, 0, 1)).to("cuda") / 255.).unsqueeze(0)
#   img1 = (torch.tensor(img1.transpose(2, 0, 1)).to("cuda") / 255.).unsqueeze(0)
#   n, c, h, w = img0.shape
#   ph = ((h - 1) // 32 + 1) * 32
#   pw = ((w - 1) // 32 + 1) * 32
#   padding = (0, pw - w, 0, ph - h)
#   img0 = F.pad(img0, padding)
#   img1 = F.pad(img1, padding)

#   return img0, img1,h,w

# def zoom(image, zoom_factor):
#     original_width, original_height = image.size
#     new_width = int(original_width * zoom_factor)
#     new_height = int(original_height * zoom_factor)
#     zoomed_image = image.resize((new_width, new_height))
#     left = max((new_width - original_width) // 2, 0)
#     top = max((new_height - original_height) // 2, 0)
#     right = min((new_width + original_width) // 2, new_width)
#     bottom = min((new_height + original_height) // 2, new_height)
#     cropped_image = zoomed_image.crop((left, top, right, bottom))
#     return cropped_image

# def add_noise(img,k):
#     img_gray = np.array(img)
#     noise = np.random.normal(0, k, img_gray.shape)
#     img_noised = img_gray + noise
#     img_noised = np.clip(img_noised, 0, 255).astype(np.uint8)
#     i = Image.fromarray(img_noised)
#     return Image.fromarray(img_noised)


# def intialize_rife():
#   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#   torch.set_grad_enabled(False)
#   if torch.cuda.is_available():
#       print("availale")
#       torch.backends.cudnn.enabled = True
#       torch.backends.cudnn.benchmark = True

#   model = Model()
#   model.device()
#   model.load_model('train_log')
#   model.eval()
#   return model

# def warmup(model):
#   prompt = "Classical art, A curious fox explores a golden-hued forest bathed in the warm light of autumn, feeling playful and mischievous, 4k, high quality"
#   img1 = np.zeros((512, 512, 3) , dtype=np.uint8)
#   img1 = Image.fromarray(img1)
#   img2 = img1
#   img_list = rife(img1,img2,2,model)

def initialize_i2vgen():
  '''
  This function intializes pipeline for i2vgen
  
  Input: 
  Output: Pipeline of I2VGen
  
  Parameters:

  '''
      
  repo_id = "ali-vilab/i2vgen-xl" 
  pipeline = I2VGenXLPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, variant="fp16").to("cuda")
  return pipeline

def generate_vid(pipe,image,prompt,height,width,fps,num_frames):
  '''
  This function intializes pipeline for i2vgen
  
  Input: 
  Output: List of images
  
  Parameters:
  - pipe: I2VGen pipeline used for video generation
  - image: initial image input for video
  - height: height of video
  - width: width of video
  - fps: frames per second of video
  - num_frames: number of frames being generated
  '''
  generator = torch.manual_seed(8888)
  frames = pipe(
      prompt=prompt,
      image=image,
      height=height,
      width=width,
      generator=generator,
      target_fps = fps,
      num_frames = num_frames
  ).frames[0]
  return frames

def main():
#   model = intialize_rife()
#   warmup(model)
  pipe = initialize_i2vgen()
  image_url = "https://raw.githubusercontent.com/ali-vilab/VGen/main/data/test_images/img_0009.png"
  image = load_image(image_url).convert("RGB")
  prompt = "Papers were floating in the air on a table in the library"
  result = generate_vid(pipe,image,prompt,HEIGHT,WIDTH,FPS,NUM_FRAMES)
