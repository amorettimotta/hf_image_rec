import gradio as gr
from PIL import Image
import numpy as np
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import BlipProcessor, BlipForConditionalGeneration
import warnings
warnings.filterwarnings('ignore')

# Load models
image_classifier = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to analyze images
def analyze_image(image):
    try:
        # Ensure image is in RGB
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Image Classification
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = image_classifier(**inputs)
        probs = outputs.logits.softmax(1)
        
        # Get top 5 predictions
        top5_prob, top5_indices = torch.topk(probs, 5)
        classifications = {
            image_classifier.config.id2label[idx.item()]: prob.item()
            for idx, prob in zip(top5_indices[0], top5_prob[0])
        }
        
        # Generate Image Caption
        inputs = caption_processor(image, return_tensors="pt")
        out = caption_model.generate(**inputs)
        caption = caption_processor.decode(out[0], skip_special_tokens=True)
        
        # Additional Analysis
        img_array = np.array(image)
        average_color = img_array.mean(axis=(0,1))
        brightness = img_array.mean()
        
        description = f"""Image Caption: {caption}
        
Analysis:
- Brightness: {brightness:.1f}/255
- Average Color: R:{average_color[0]:.1f}, G:{average_color[1]:.1f}, B:{average_color[2]:.1f}
- Resolution: {image.size[0]}x{image.size[1]}
"""
        return classifications, description

    except Exception as e:
        return {"error": str(e)}, f"An error occurred: {str(e)}"

# Custom CSS for better appearance
css = """
body {
    background-color: white !important;
    color: black !important;
}
h1 {
    font-size: 32px !important;
    font-weight: bold !important;
    text-align: center !important;
    color: black !important;
}
label {
    font-size: 18px !important;
    color: black !important;
}
.gradio-container {
    font-size: 16px !important;
    color: black !important;
    background-color: white !important;
}
"""

# Create the Gradio interface
with gr.Blocks(css=css) as demo:
    gr.Markdown("# Image Recognition System")
    gr.Markdown("Upload an image to analyze its content and get detailed information.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("Analyze Image", variant="primary")
        
        with gr.Column():
            output_labels = gr.Label(label="Classifications")
            output_text = gr.Textbox(label="Analysis Details", lines=10)
    
    submit_btn.click(
        analyze_image,
        inputs=[input_image],
        outputs=[output_labels, output_text]
    )

if __name__ == "__main__":
    demo.launch()