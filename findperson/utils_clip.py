from appPublic.jsonConfig import getConfig
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

class CLIPEmbedder:
    def __init__(self):
		# model_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
		self.config = getConfig()
		model_path = self.config.clip_model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_path)

    def embed_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
        return embedding[0].cpu().numpy()

    def embed_text(self, text):
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embedding = self.model.get_text_features(**inputs)
        return embedding[0].cpu().numpy()

