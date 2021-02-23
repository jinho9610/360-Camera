import cv2
import numpy as np
from keras.models import load_model
from facenet_pytorch import MTCNN, InceptionResnetV1

IMG_SIZE = (34, 26) # 입 이미지의 가로, 세로 사이즈

mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40) # keep_all=True
resnet = InceptionResnetV1(pretrained='vggface2').eval()

model = load_model('models/2021_02_12_16_38_25.h5')

def crop_mouse(ori, box, face):
    nose_x, nose_y = int(face[2][0]), int(face[2][1])
    lm_x, lm_y = int(face[3][0]), int(face[3][1])
    rm_x, rm_y = int(face[4][0]), int(face[4][1])
    
    img = ori[nose_y + 6 : int(box[3]), lm_x : rm_x]

    rect = [(lm_x, nose_y + 6), (rm_x, int(box[3]))]

    img = cv2.resize(img, dsize=(34,26))

    return img, rect


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()

        if not ret:
            print('end of video')
            break

        img_cropped_list, prob_list = mtcnn(img, return_prob=True)

        if img_cropped_list is not None:
            boxes, _, faces = mtcnn.detect(img, landmarks=True)
            
            for i in range(len(faces)):
                face = faces[i]
                box = boxes[i]

                mouse, mouse_rect = crop_mouse(img, box, face)
                img = cv2.rectangle(img, mouse_rect[0], mouse_rect[1], (255, 0, 0), 1)
                
                mouse = cv2.cvtColor(mouse, cv2.COLOR_BGR2GRAY) # 입 사진 gray로 변경
                mouse_input = mouse.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255
                pred = model.predict(mouse_input)
                state = 'O %.1f' if pred > 0.1 else '- %.1f'
                state = state % pred

                cv2.putText(img, state, (mouse_rect[0][0] + int((mouse_rect[1][0] - mouse_rect[0][0]) / 4), mouse_rect[0][1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
                for k in range(len(face)):
                    img = cv2.circle(img, (int(face[k][0]), int(face[k][1])), 1, (0, 0, 255), -1)

        cv2.imshow("IMG", img)
        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()


        
        