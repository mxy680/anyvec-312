import torch
from transformers import CLIPProcessor, CLIPModel
from torch.nn.functional import cosine_similarity
from PIL import Image
import io

# Load CLIP Model and Processor
model_name = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

def process_elements(data: tuple):
    #going to have a text string
    #going to have a list of bytes
    text = data[0]  # Text string
    byte_list = data[1]  # List of bytes (images/files)

    if len(text) == 0:
        return None

    # Process text
    text_inputs = processor(text=[text], return_tensors="pt", padding=True)

     # Convert text to vector
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)

    print(f"Text Vector Shape: {text_features.shape}")

    # Process bytes (assuming images)
    image_vectors = []
    for byte_data in byte_list:
        image = Image.open(io.BytesIO(byte_data))
        image_inputs = processor(images=image, return_tensors="pt", padding=True)

        # Convert image to vector
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)

        image_vectors.append(image_features)
        print(f"Image Vector Shape: {image_features.shape}")

    return text_features, image_vectors

# ------------------------
# 🚀 Test the Function Here
# ------------------------

# 1️⃣ Sample text
sample_text = "Picturesque scene of heaven"

# 2️⃣ Sample byte data (create a simple red image in memory)
image = Image.new('RGB', (64, 64), color='black')  # Simple dummy image
img_byte_arr = io.BytesIO()
image.save(img_byte_arr, format='JPEG')
byte_data = img_byte_arr.getvalue()

# 3️⃣ Tuple input: (text, [list of image bytes])
data = (sample_text, [byte_data])

# 4️⃣ Run the function
text_vec, img_vecs = process_elements(data)

# 5️⃣ Output the result
print("Text Vector:", text_vec)
print("First Image Vector:", img_vecs[0])

# Assuming text_vec and img_vecs[0] are your tensors
similarity = cosine_similarity(text_vec, img_vecs[0])
print(f"Cosine Similarity: {similarity.item()}")
