import gradio as gr
from transformers import pipeline
from diffusers import StableDiffusionPipeline
text_generator = pipeline("text-generation", model="gpt2",pad_token_id=50256)
model_id ="CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

def generate_future(prompt):
    future_prompt = f"How will {prompt} look in the future ?"
    result= text_generator(future_prompt,do_sample=True)
    return result

def generate_image(prompt):
    if isinstance(prompt,list):
        prompt = prompt[0]
    image=pipe(str(prompt)[:77]).images[0]
    return image

def predict_future(user_input):
    future_description = generate_future(user_input)
    image = generate_image(future_description)
    return  future_description, image

def create_interface():
    interface = gr.Interface(
    fn=predict_future,
    inputs="text",
    outputs=["text","image"],
    title="Future Mİrror",
    description="Type something and see how it may look in the future!"
    )
    return interface

interface = create_interface()
interface.launch()

