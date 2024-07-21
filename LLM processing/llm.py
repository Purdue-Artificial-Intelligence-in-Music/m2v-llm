from openai import OpenAI
import os
import re
import torch
from diffusers import StableDiffusion3Pipeline

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", 'your_api_key_here'))

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe.to("cuda")

def img_generation(prompt, save_dir='temp'):
    '''
    This function generates an image based on the prompt and saves it to the specified directory
    
    Input: input prompt for image generation
    Output: path to the saved image
    
    Parameters:
    - prompt: input prompt for image generation
    - save_dir: directory to save the generated image
    '''
    global pipe
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image = pipe(prompt).sample()
    save_path = os.path.join(save_dir, f"generated_image_{len(os.listdir(save_dir)) + 1}.png")

    image.save(save_path)
    print(f"Image saved to: {save_path}")
    return save_path

def recurrentStoryboard(current_clip, music_summary, previous_summary, previous_frame, previous_suggestions):
    '''
    This function generates a storyboard for a piece of music to express the music based on the input parameters
    
    Input: input parameters for generating the storyboard
    Output: generated storyboard
    
    Parameters:
    - current_clip: description of the current music clip
    - music_summary: summary of the overall music
    - previous_summary: summary of the previous story
    - previous_frame: description of the frame for each iteration and it will be used as input for the next iteration
    - previous_suggestions: optional suggestions for the previous frame
    '''
    template = """
You are an expert in story writing who can fully understand music and realize synesthesia. Please design a storyboard for a piece of music to express the music. You only need to design the story and the description of the frame instead of designing the real frame.

In the design of each frame described in this storyboard, you will receive the following input：
input 1. A description of 10-second clip of the current music
input 2. The summary of the entire music
input 3. The summary of the previous story
input 4. Optional suggestions when designed the previous frame

During the design process of current frame, make sure you always keep the following guidelines in mind:
    1. Coherence between frames: The current frame is coherent with the previous frame
    2. Alignment of music clip and frame: The current frame can realie synesthesia with the input music
    3. Consistency of music and story: story as a whole fits the overall and theme of the music
    4. Integrity of a single frame description: The whole story is coherent, but while maintaining continuity between frames, each frame itself is also complete

The following is the input you receive, if only the description of the current music clip is left blank, it means that it is currently the first frame of the entire storyboard. For the first frame, use the summary of the current music clip and the summary of the current frame directly as a summary of the entire music and story to start the iteration.
Input 1 - Description for current music clip: {current_clip}
Input 2 - Summary of the overall music: {music_summary}
Input 3 - Summary of the previous story: {previous_story}
Input 4 - Description the previous frame: {previous_frame}
Input 5 - Optional suggestions for the previous frame: {previous_suggestions}

Follow the template below to output the result, and replace the placeholder with your response:
***Template Begins***

## Analysis for current clip ##
[Think step by step. Based on the input of a 10-second clip of the current music, use a short paragraph to describe how you should design a picture to achieve synesthesia with music]

## Summary of Overall Music ##
[Based on the current music clip and the overall summary of the current music, update the summary of the current entire music and the theme]

## Description of Current Frame ##
[Based on the analysis of the current music clip, as well as the description of the previous frame and the summary of the previous story and optional suggestions, generate a description of the new frame of the storyboard]

## Suggestions for the Next Frame ##
[Based on the current frame you describe, make a suggestion for the design of the next frame in order to ensure that the next frame is coherent with the current frame and that the story as a whole fits the overall and theme of the music]

## Updated Story Theme Summary ##
[Based on the description of current frame and the summary of the previous story, update the summary of the overall direction and theme of the story]

***Template Ends***“”“

    """
    try:
        # Generate completion from OpenAI's chat model
        completion = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            timeout=100,
            messages=[
                {"role": "user", "content": template.format(current_clip=current_clip, music_summary=music_summary, previous_story=previous_summary, previous_frame=previous_frame, previous_suggestions=previous_suggestions)},
            ]
        )
        return completion.choices[0].message.content  # Return the predicted result
    except Exception as e:
        return "Error: " + str(e)
    
def extract_info(response):
    '''
    This function extracts the relevant information from the response generated by the recurrentStoryboard function
    
    Input: response generated by the recurrentStoryboard function
    Output: extracted information from the response
    
    Parameters:
    - response: response generated by the recurrentStoryboard function
    '''
    music_summary = re.search(r"## Summary of Overall Music ##\n(.*?)\n", response, re.DOTALL).group(1)
    previous_summary = re.search(r"## Updated Story Theme Summary ##\n(.*?)\n", response, re.DOTALL).group(1)
    previous_frame = re.search(r"## Description of Current Frame ##\n(.*?)\n", response, re.DOTALL).group(1)
    previous_suggestions = re.search(r"## Suggestions for the Next Frame ##\n(.*?)\n", response, re.DOTALL).group(1)
    return music_summary, previous_summary, previous_frame, previous_suggestions

#TODO: Implement get_current_clip with Music Processing
def get_current_clip():
    current_clip =  "A description of 10-second clip of the current music"
    return current_clip

def main():
    current_clip = get_current_clip()
    music_summary = current_clip
    previous_summary = ""
    previous_frame = ""
    previous_suggestions = ""
    while current_clip:
        response = recurrentStoryboard(current_clip, music_summary, previous_summary, previous_frame, previous_suggestions)
        print(response)
        music_summary, previous_summary, previous_frame, previous_suggestions = extract_info(response)
        img_generation(previous_frame)
        current_clip = get_current_clip()
    
