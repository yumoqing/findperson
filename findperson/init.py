from ahserver.serverenv import ServerEnv
from appPublic.worker import awaitify
from .imageface import ImageFaces
from .utils_clip import CLIPEmbedder

def load_findperson():
	g = ServerEnv()
	ifs = ImageFaces()
	ce = CLIPEmbedder()
	g.find_face_in_image = awaitify(ifs.find_face_in_image)
	g.save_faces = awaitify(ifs.save_faces)
	g.embed_image = ce.embed_image
	g.embed_text = ce.embed_text
