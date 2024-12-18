from openai import OpenAI
import os
import re
import torch
from diffusers import StableDiffusion3Pipeline

llm_prompt_path = "llm_prompt.txt"

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

def recurrentStoryboard(current_clip, music_summary, previous_summary, previous_end_frame, previous_suggestions):
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
    prompt_file = open(llm_prompt_path, "r")
    template = prompt_file.read()
    try:
        # Generate completion from OpenAI's chat model
        completion = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            timeout=100,
            messages=[
                {"role": "user", "content": template.format(current_clip=current_clip, music_summary=music_summary, previous_story=previous_summary, previous_end_frame=previous_end_frame, previous_suggestions=previous_suggestions)},
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
    music_summary = re.search(r"## Summary of Overall Music ##\n(.*?)\n\n", response, re.DOTALL).group(1).strip()
    previous_summary = re.search(r"## Updated Story Theme Summary ##\n(.*?)\n\n", response, re.DOTALL).group(1).strip()
    start_frame = re.search(r"## Start Frame Description ##\n(.*?)\n\n", response, re.DOTALL).group(1).strip()
    end_frame = re.search(r"## End Frame Description ##\n(.*?)\n\n", response, re.DOTALL).group(1).strip()
    next_segment_suggestions = re.search(r"## Suggestions for the Next Segment ##\n(.*?)$", response, re.DOTALL).group(1).strip()
    return music_summary, previous_summary, start_frame, end_frame, next_segment_suggestions
    

def get_captions():
    '''
    This captions list is from the MuLLaMA.
    '''
    return [
        "This music sounds like a mysterious and atmospheric soundscape. It features a haunting melody played on a string instrument, accompanied by subtle, ambient sounds. The tempo is slow and deliberate, creating a sense of suspense and intrigue. The overall mood is dark and evocative, with a hint of melancholy.",
        "This music is a beautiful and ethereal soundscape. It features a haunting melody played on a string instrument, accompanied by gentle, ambient sounds. The tempo is slow and meditative, creating a sense of peace and tranquility. The overall mood is serene and dreamy, with a hint of sadness.",
        "This music is a blend of mystery and ethereal beauty. It starts with a haunting melody played on a string instrument, creating a sense of suspense and intrigue. As the music progresses, it becomes more atmospheric, with the addition of ambient sounds and a delicate piano melody. The tempo is slow and deliberate, creating a sense of peace and tranquility. The overall mood is dark and evocative, with a hint of melancholy",
        "This music is a captivating blend of haunting melodies and ambient sounds. It begins with a slow, deliberate tempo, creating a sense of mystery and intrigue. A string instrument plays a haunting melody, while subtle ambient sounds add depth and texture to the soundscape. The music gradually builds in intensity, with the string instrument becoming more prominent and the ambient sounds becoming more layered. The overall mood is dark and evocative, with a hint of melancholy."
    ]

def main():
    captions = get_captions()  
    music_summary = ""
    previous_summary = ""
    previous_end_frame = ""
    previous_suggestions = ""

    for i, current_clip in enumerate(captions):
        print(f"\nProcessing caption {i+1}/{len(captions)}")
        response = recurrentStoryboard(current_clip, music_summary, previous_summary, previous_end_frame, previous_suggestions)
        print(response)
        music_summary, previous_summary, start_frame, end_frame, next_segment_suggestions = extract_info(response)
        img_generation(start_frame)
        img_generation(end_frame)
        previous_end_frame = end_frame  

    
