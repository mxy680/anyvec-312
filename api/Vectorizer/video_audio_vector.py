import torch
import torchaudio
import torchvision
from transformers import CLIPProcessor, CLIPModel
from torch.nn.functional import cosine_similarity
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchaudio.transforms import MelSpectrogram
from PIL import Image
import io

class AVModel:
    def __init__(self):
        self.model_name = "openai/clip-vit-large-patch14"

        # Load CLIP Model and Processor
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

        # Audio processor (Convert audio bytes into spectrogram)
        self.audio_transform = MelSpectrogram(sample_rate=16000, n_mels=128)

        # Video frame transformations (Resize, Normalize)
        self.video_transform = Compose([
            Resize((224, 224)),  # Resize frames to CLIP input size
            ToTensor(),  # Convert to tensor
            Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])  # CLIP normalization
        ])

    def process_video(self, video_bytes):
        """ Extracts frames from video bytes and processes them using CLIP. """
        video_tensor, _, _ = torchvision.io.read_video(io.BytesIO(video_bytes), pts_unit="sec")

        if video_tensor.shape[0] == 0:
            return None  # No frames extracted

        # Select a subset of frames (every 10th frame for efficiency)
        sampled_frames = video_tensor[::10]

        # Apply transformations to each frame
        processed_frames = torch.stack(
            [self.video_transform(Image.fromarray(frame.numpy())) for frame in sampled_frames])

        # Get CLIP image features
        with torch.no_grad():
            video_features = self.model.get_image_features(pixel_values=processed_frames.unsqueeze(0))

        return video_features

    def process_audio(self, audio_bytes):
        """ Converts raw audio bytes into a spectrogram and extracts features. """
        audio_tensor, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))

        # Convert to mono if stereo
        if audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

        # Convert to Mel spectrogram
        spectrogram = self.audio_transform(audio_tensor)

        # Normalize spectrogram
        spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()

        # Resize spectrogram for CLIP processing (simulate an image)
        spectrogram = torch.nn.functional.interpolate(spectrogram.unsqueeze(0), size=(224, 224), mode="bilinear")

        # Convert to 3-channel image format expected by CLIP
        spectrogram = spectrogram.repeat(1, 3, 1, 1)

        # Get CLIP image features
        with torch.no_grad():
            audio_features = self.model.get_image_features(pixel_values=spectrogram)

        return audio_features

    def process_elements(self, data: tuple):
        """
        Processes video and audio input.
        data[0] -> video bytes
        data[1] -> audio bytes
        """
        video_bytes, audio_bytes = data

        video_features = self.process_video(video_bytes) if video_bytes else None
        audio_features = self.process_audio(audio_bytes) if audio_bytes else None

        return video_features, audio_features