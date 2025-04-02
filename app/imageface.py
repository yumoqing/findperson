import face_recognition
from PIL import Image
from appPublic.jsonConfig import getConfig

from image_imbedding import ImageImbedding
from vectordb import MilvusVectorDB

class ImageFaces:
    """
    implements image to faces and to vector and save the face's vector to vector database
    and search similarities faces in vector database with faces in argument image_path
    """
    def __init__(self):
        self.ii = ImageImbedding()

    def get_vectordb(self, orgid):
        """
        get or create a vector database for orgid's organization
        """
        config = getConfig()
        dbname = f'{config.vectordb_path}/{orgid}'
        db = MilvusVectorDB(dbname, 'faces')
        return db

    def save_faces(self, orgid, userid, image_path, imgid=None):
        """
        find all  the faces in image identified by image_path, 
        and save the face's info to orgid's vector database
        """
        db = MilvusVectorDB(orgid)
        v = self.ii.image2faces(image_path, imgid=imgid)
        for face in v['faces']:
            face['userid'] = userid
            print(f'{face=}')
            db.add('faces', face)
        return v

    def find_face_in_image(self, orgid, userid, image_path):
        """
        similarities search for all the faces in image identified by image_path
        return faces info in image attached similarities face's info
        """
        db = MilvusVectorDB(orgid)
        v = self.ii.image2faces(image_path)
        for face in v['faces']:
            ret = db.search_by_vector('faces', face['vector'])
            face['similarities'] = ret
        return v

if __name__ == '__main__':
    import sys
    from appPublic.jsonConfig import getConfig
    config = getConfig('./', NS={'workdir':'.'})
    if len(sys.argv) < 2:
        print(f'{sys.argv[0]} imgfile ...')
        sys.exit(1)
    i_f = ImageFaces()
    fif = sys.argv[1]
    # for f in sys.argv[1:]:
    #    v = i_f.save_faces('oooo1', 'testuser', f)
    v = i_f.find_face_in_image('oooo1', 'testuser', fif)
    print(v)
