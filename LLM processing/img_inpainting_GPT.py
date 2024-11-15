import os
import re
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline
from openai import OpenAI
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import supervision as sv
from typing import List

# Configuration paths
HOME = "/path/to/home"  # Replace with the actual path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "config", "grounding_dino.yaml")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "grounding_dino.pth")

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# Load models
grounding_dino_model = Model(
    model_config_path=GROUNDING_DINO_CONFIG_PATH,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH
)

SAM_ENCODER_VERSION = "vit_h"
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe.to(DEVICE)

# Enhance class names for GroundingDINO
def enhance_class_name(class_names: List[str]) -> List[str]:
    return [f"all {class_name}s" for class_name in class_names]

# Use GPT to generate dynamic CLASS and PROMPT
def gpt_decision_making(prev_music_caption, current_music_caption, prev_key_frame):
    client = OpenAI()
    prompt = f"""
    You are an AI assistant tasked with analyzing the previous music caption "{prev_music_caption}" and the current music caption "{current_music_caption}". Based on the change in the music captions, decide which object in the previous key frame "{prev_key_frame}" should be modified to better match the current music caption. 

    Provide the following information in your response:
    1. The name of the object (class) that should be modified (e.g., "tree", "person", etc.)
    2. A detailed prompt that can be used with Stable Diffusion Inpaint Pipeline to generate the modified key frame (e.g., "Big green tree, high resolution, spring")

    Respond in the following format:
    Object: <object_name>
    Prompt: <prompt>
    """
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    gpt_output = completion.choices[0].message.content.strip()
    return gpt_output

# Extract object name from GPT output
def extract_object_name(gpt_output):
    match = re.search(r"Object:\s*(\w+)", gpt_output)
    return match.group(1) if match else None

# Extract prompt from GPT output
def extract_prompt(gpt_output):
    match = re.search(r"Prompt:\s*(.*)", gpt_output)
    return match.group(1) if match else None

# Perform segmentation using SAM
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

# Core processing function
def process_image(prev_music_caption, current_music_caption, source_image_path):
    # Load input image
    image = cv2.imread(source_image_path)

    # Generate dynamic CLASS and PROMPT using GPT
    gpt_output = gpt_decision_making(prev_music_caption, current_music_caption, source_image_path)
    dynamic_class = extract_object_name(gpt_output)
    dynamic_prompt = extract_prompt(gpt_output)

    # Detect objects using GroundingDINO
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=[dynamic_class]),
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # Segment objects using SAM
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    # Save the segmentation mask
    mask_path = "mask.png"
    plt.imshow(detections.mask[0], cmap='gray')
    plt.axis('off')
    plt.savefig(mask_path, bbox_inches='tight', pad_inches=0)
    mask_image = Image.open(mask_path).convert("L")

    # Perform inpainting using Stable Diffusion
    result_image = pipe(prompt=dynamic_prompt, image=Image.open(source_image_path), mask_image=mask_image).images[0]

    # Save the output image
    output_path = "output.png"
    result_image.save(output_path)

    return output_path

# Main function for testing
if __name__ == "__main__":
    prev_caption = "Calm forest with a light breeze"
    current_caption = "Energetic forest with bright sunlight"
    source_image = f"{HOME}/data/test.jpg"

    output = process_image(prev_caption, current_caption, source_image)
    print(f"Processed image saved to: {output}")
