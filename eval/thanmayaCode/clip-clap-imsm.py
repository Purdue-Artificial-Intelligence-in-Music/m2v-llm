from imsm_model import compute_imsm_melfusion
from preprocess_text import read_file_into_list
import os


# Example file paths for images, texts, and audios
def get_image_files_from_directory(directory):
    """
    Get a list of all PNG image files in the specified directory.
    """
    image_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.png')]
    return image_files

# Define the directory containing the screenshots
screenshot_directory = 'DemoOutput/1918'  

# Get the list of all PNG images in the screenshot directory
image_files = get_image_files_from_directory(screenshot_directory)
texts = read_file_into_list('DemoOutput/Demo outputs/1918_prompts.txt')
audio_files = ['DemoOutput/Demo outputs/1918.wav']  # Replace with your audios

# Call the IMSM computation function
compute_imsm_melfusion(image_files,audio_files, texts)
