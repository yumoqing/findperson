from ahserver.webapp import webapp
from ahserver.serverenv import ServerEnv
from appPublic.worker import awaitify
from imageface import ImageFace

def get_module_dbname(name):
    return 'imagefind'

def init():
    g = ServerEnv()
    g.get_module_dbname = get_module_dbname
    g.imageface = ImageFace()
    g.save_faces = awaitify(g.imageface.save_faces)
    g.find_face_in_image = awaitify(g.imageface.find_face_in_image)

if __name__ == '__main__':
    webapp(init)
