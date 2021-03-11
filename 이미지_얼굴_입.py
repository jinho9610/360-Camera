'''
torch와 torchvision 버전을 잘 맞춰줘야함
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
'''

from facenet_pytorch import MTCNN, InceptionResnetV1
from keras.models import load_model
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import mtcnn
import cv2
import time
import os
from distortion import *
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


IMG_SIZE = (34, 26)  # 입 이미지의 가로, 세로 사이즈

mtcnn = MTCNN(image_size=240, margin=0, keep_all=True,
              min_face_size=40)  # keep_all=True
# resnet = InceptionResnetV1(pretrained='vggface2').eval()
#resnet = torch.load('new_res.pt').eval()
#resnet = torch.load('4xxxfinetuned_IRV1.pt').eval()
resnet = torch.load('new_res2.pt').eval()
model = load_model('models/2021_03_05_13_51_54.h5')


def check_mouse(ori, box, face):
    nose_x, nose_y = int(face[2][0]), int(face[2][1])
    lm_x, lm_y = int(face[3][0]), int(face[3][1])
    rm_x, rm_y = int(face[4][0]), int(face[4][1])

    mouse = ori[nose_y + 6: int(box[3]), lm_x: rm_x]

    mouse_rect = [(lm_x, nose_y + 6), (rm_x, int(box[3]))]

    mouse = cv2.resize(mouse, dsize=(34, 26))
    ori = cv2.rectangle(
        ori, mouse_rect[0], mouse_rect[1], (255, 0, 0), 1)  # 입주변 박스 그리기

    mouse = cv2.cvtColor(mouse, cv2.COLOR_BGR2GRAY)  # 입 사진 gray로 변경
    mouse_input = mouse.copy().reshape(
        (1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255
    pred = model.predict(mouse_input)
    state = 'O %.1f' if pred > 0.1 else '- %.1f'
    state = state % pred

    cv2.putText(ori, state, (mouse_rect[0][0] + int((mouse_rect[1][0] - mouse_rect[0][0]) / 4),
                             mouse_rect[0][1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return ori


def img_face_mouse_rec(load_data, img_path):
    img = cv2.imread(img_path)
    img0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(type(img))
    # img0 = cv2.resize(img, dsize=(512, 512))  # 이걸 왜 쳐하고 있었던거지
    img_cropped_list, prob_list = mtcnn(img0, return_prob=True)
    print(img_cropped_list)
    print(1)

    if img_cropped_list is not None:
        print(2)
        boxes, _, faces = mtcnn.detect(img0, landmarks=True)

        for i, prob in enumerate(prob_list):
            print(3)
            if prob > 0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()

                dist_list = []  # list of matched distances, minimum distance is used to identify the person
                print(dist_list)

                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)

                min_dist = min(dist_list)  # get minumum dist value
                min_dist_idx = dist_list.index(
                    min_dist)  # get minumum dist index
                # get name corrosponding to minimum dist
                name = name_list[min_dist_idx]

                box = boxes[i]

                # original_frame = frame.copy() # storing copy of frame before drawing on it

                if min_dist < 0.90:
                    img = cv2.putText(img, name+' '+str(round(min_dist, 3)), (int(box[0]), int(
                        box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    img = cv2.putText(img, 'UNKNOWN', (int(box[0]), int(
                        box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

                img = cv2.rectangle(img, (int(box[0]), int(
                    box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

                face = faces[i]
                for k in range(len(face)):
                    img = cv2.circle(
                        img, (int(face[k][0]), int(face[k][1])), 1, (0, 0, 255), -1)

    cv2.imshow('img', img)

    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    load_data = torch.load('new_data8.pt')
    embedding_list = load_data[0]
    name_list = load_data[1]

    # img_face_mouse_rec(load_data, 'photos/hyeontae/HT.jpg')
    img_face_mouse_rec(
        load_data, 'not_face/3men.jpg')
    # img_face_mouse_rec(
    #     load_data, 'trans_learn_dataset/hyeontae/KakaoTalk_20210225_162557523_01.jpg')
