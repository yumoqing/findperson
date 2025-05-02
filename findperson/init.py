from ahserver.serverenv import ServerEnv
from appPublic.worker import awaitify
from .imageface import ImageFaces

def load_findperson():
    g = ServerEnv()
    g.imagefaces = ImageFaces()
    g.save_faces = awaitify(g.imagefaces.save_faces)
    g.find_face_in_image = awaitify(g.imagefaces.find_face_in_image)

