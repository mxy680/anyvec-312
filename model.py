import transformers
import torch
from transformers import pipeline

#Ensures everything is already downloaded
print(f"Transformers version: {transformers.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")


from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Fetch an image from the internet
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Define text labels to compare the image against
text_labels = ["a photo of a cat", "a photo of a dog"]

# Process inputs
inputs = processor(text=text_labels, images=image, return_tensors="pt", padding=True)

# Get model output
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # Image-text similarity score
probs = logits_per_image.softmax(dim=1)  # Convert scores to probabilities

# Print the probabilities for each label
for label, prob in zip(text_labels, probs.squeeze().tolist()):
    print(f"{label}: {prob:.4f}")

# Print the most likely label
predicted_label = text_labels[torch.argmax(probs)]
print(f"Predicted Label: {predicted_label}")


