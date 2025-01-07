from flask import Flask, request, render_template, jsonify
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import io

app = Flask(__name__)

# Load the Qwen2-VL-7B-Instruct model and processor only once
print("Loading model and processor...")
model_name = "Qwen/Qwen2-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
print("Model and processor loaded.")

def vqa_model(image, question):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    # Create the text prompt using the chat template
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Prepare inputs for the model
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # Generate the model's response
    output_ids = model.generate(**inputs, max_new_tokens=1024)

    # Decode the generated output
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return output_text[0] if output_text else "No response generated."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Retrieve the image and question from the form
    image_file = request.files.get('image')
    question = request.form.get('question')

    if not image_file or not question:
        return jsonify({'error': 'Please provide both an image and a question.'}), 400

    # Convert the image file to a PIL image
    image = Image.open(io.BytesIO(image_file.read()))

    # Get the response from the VQA model
    response = vqa_model(image, question)

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
