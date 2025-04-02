import face_recognition
import numpy as np
from sklearn.decomposition import PCA
from appPublic.uniqueID import getID

def expand_to_768_zero_padding(embedding):
    return np.pad(embedding, (0, 768 - 128), mode="constant")

class ImageImbedding:
    def image2faces(self, image_path, imgid=None):
        image = face_recognition.load_image_file(image_path)

        # 检测人脸并提取嵌入向量
        face_locations = face_recognition.face_locations(image)  # 找到所有人脸
        face_encodings = face_recognition.face_encodings(image, face_locations)  # 计算 128 维特征
        if imgid is None:
            imgid = getID()
        faces = []
        ret = {
            "id": imgid,
            "image": image_path,
            "faces": faces
        }
        for i, v in enumerate(face_encodings):
            # print('vector v has', v.shape, 'shape', v)
            v_768 = expand_to_768_zero_padding(v)
            # print(v.shape, v_768.shape)
            top, right, bottom, left = face_locations[i]
            face = {
                "id": getID(),
                "imgid": imgid,
                "vector": v_768,
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom
            }
            faces.append(face)
        return ret


if __name__ == '__main__':
    ii = ImageImbedding()
    v = ii.image2vector('test.png')
