from transformers import AutoProcessor, AutoModelForZeroShotImageClassification, AutoTokenizer
import torch
import faiss
import numpy as np

class ImageFind:
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

        #Load CLIP model, processor and tokenizer
        self.processor = AutoProcessor.from_pretrained("facebook/metaclip-b16-fullcc2.5b")
        self.model = AutoModelForZeroShotImageClassification.from_pretrained("facebook/metaclip-b16-fullcc2.5b",  torch_dtype=torch.float16).to(device)
        self.model = torch.compile(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/metaclip-b16-fullcc2.5b")
        self.device = device
        self.faiss_index = faiss.IndexFlatL2(1024)

    def save_vector(self, vector):
        self.faiss_index.add(vector)

    def similarity(self, image):
        v = self.image2vector(image)
        distances, indices = self.faiss_index.search(v, 5)



    def image2vector(self, image):
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            vector = image_features.detach().numpy()
            vector = np.float32(vector)
            faiss.normalize_L2(vector)
            return vector
