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
from random import *

IMG_SIZE = (34, 26)  # 입 이미지의 가로, 세로 사이즈

mtcnn = MTCNN(image_size=240, margin=0, keep_all=True,
              min_face_size=40)  # keep_all=True
resnet = InceptionResnetV1(pretrained='vggface2').eval()
model = load_model('models/2021_02_23_00_15_13.h5')


def img_face_mouse_rec(load_data, img):
    flag = False  # 플래그 초기화
    # 현재 img는 ndarray
    # img0 = Image.fromarray(img)  # PIL 형태로 변형
    img0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_cropped_list, prob_list = mtcnn(img0, return_prob=True)

    if img_cropped_list is not None:
        # 얼굴 하나라도 찾으면
        flag = True
        boxes, _, faces = mtcnn.detect(img0, landmarks=True)

        for i, prob in enumerate(prob_list):
            if prob > 0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()

                dist_list = []  # list of matched distances, minimum distance is used to identify the person

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

    return img, flag


if __name__ == '__main__':
    load_data = torch.load('data.pt')
    embedding_list = load_data[0]
    name_list = load_data[1]

    #img_face_mouse_rec(load_data, 'photos/hyeontae/HT.jpg')
    cam = cv2.VideoCapture(0)
    while True:
        r1 = randint(1, 1000)
        r2 = randint(1, 1000)

        ret, frame = cam.read()

        if not ret:
            print("fail to grab frame, try again")
            break

        frame, flag = img_face_mouse_rec(load_data, frame)
        print(flag)

        if not flag:
            cv2.imwrite('not_face/' + str(r1) + str(r2) + '.jpg', frame)

        cv2.imshow('frames', frame)
        if cv2.waitKey(1000) == 27:
            break
