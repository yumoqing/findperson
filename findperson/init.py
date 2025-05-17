from ahserver.serverenv import ServerEnv
from appPublic.worker import awaitify
from .imageface import ImageFaces

def load_findperson():
	g = ServerEnv()
	ifs = ImageFaces()
	g.find_face_in_image = awaitify(ifs.find_face_in_image)
	g.save_faces = awaitify(ifs.save_faces)

