# your_code/your_module.py

# Import necessary libraries
import mediapy as media
import random
import sys
import torch
from diffusers import AutoPipelineForText2Image, __version__

# Display diffusers version to check for update
print(f"Using diffusers version: {__version__}")

# Define the function to generate images
def generate_image(prompt):
    # Load the Text2Image Diffuser model without specifying the variant
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )

    # Generate a random seed
    seed = random.randint(0, sys.maxsize)

    # Set the number of inference steps
    num_inference_steps = 4

    # Generate images using the Text2Image Diffuser
    images = pipe(
        prompt=prompt,
        guidance_scale=0.0,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator().manual_seed(seed),
    ).images

    # Print prompt and seed information
    print(f"Prompt:\t{prompt}\nSeed:\t{seed}")

    # Display the generated images
    media.show_images(images)

    # Save the first generated image
    images[0].save("output.jpg")

# Example usage
generate_image("a photo of Pikachu fine dining with a view to the Eiffel Tower")
