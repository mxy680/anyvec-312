import torch
from transformers import CLIPProcessor, CLIPModel
from torch.nn.functional import cosine_similarity
from PIL import Image
import io


class Model:
    def __init__(self):
        self.model_name =  "openai/clip-vit-large-patch14"
# Load CLIP Model and Processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def process_elements(self, data: tuple):
        #going to have a text string
        #going to have a list of bytes
        text = data[0]  # Text string
        byte_list = data[1]  # List of bytes (images/files)

        if len(text) == 0 and len(byte_list) == 0:
            return None

        # Process text
        text_inputs = self.processor(text=[text], return_tensors="pt", padding=True)

        # Convert text to vector
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)

        print(f"Text Vector Shape: {text_features.shape}")

        # Process bytes (assuming images)
        image_vectors = []
        
        for byte_data in byte_list:
            image = Image.open(io.BytesIO(byte_data))
            image_inputs = self.processor(images=image, return_tensors="pt", padding=True)

            # Convert image to vector
            with torch.no_grad():
                image_features = self.model.get_image_features(**image_inputs)

            image_vectors.append(image_features)
            print(f"Image Vector Shape: {image_features.shape}")

        return (text_features, image_vectors)