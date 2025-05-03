import json
import face_recognition
from PIL import Image
from appPublic.jsonConfig import getConfig

from .image_imbedding import ImageImbedding
from .vectordb import MilvusVectorDB

class ImageFaces:
    """
    implements image to faces and to vector and save the face's vector to vector database
    and search similarities faces in vector database with faces in argument image_path
    """
    def __init__(self, userid):
		self.vdb = MilvusVectorDB(userid)
        self.ii = ImageImbedding()

    def save_faces(self, image_path, imgid=None):
        """
        find all  the faces in image identified by image_path, 
        and save the face's info to vector database
        """
        v = self.ii.image2faces(image_path, imgid=imgid)
        for face in v['faces']:
            self.vdb.add('faces', face)
        return v

    def find_face_in_image(self, image_path):
        """
        similarities search for all the faces in image identified by image_path
        return faces info in image attached similarities face's info
        """
        self.vdb.create_vector_index('faces')
        v = self.ii.image2faces(image_path)
        for face in v['faces']:
            ret = self.vdb.search_by_vector('faces', face['vector'])
            ret = [d for d in ret.pop()]
            face['similarities'] = ret
        return v

if __name__ == '__main__':
    import sys
    from appPublic.jsonConfig import getConfig
    config = getConfig('./', NS={'workdir':'.'})
    if len(sys.argv) < 3:
        print(f'{sys.argv[0]} COMMAND imgfile ...')
        sys.exit(1)
    act = sys.argv[1]
    i_f = ImageFaces('oooo1')
    if act == 'add':
        for f in sys.argv[2:]:
            v = i_f.save_faces(f)
    else:
        fif = sys.argv[2]
        v = i_f.find_face_in_image(fif)
        print(v['id'], v['image'])
        for face in v['faces']:
            t = face['similarities']
            for d in t:
                print(f'{d=}')
            print(f"{face['id']=},{face['imgid']=}")

        
