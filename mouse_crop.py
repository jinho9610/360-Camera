from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
from random import *
import mtcnn
import cv2
import time
import os
import sys

mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False,
               min_face_size=40)  # keep_all=False
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True,
              min_face_size=40)  # keep_all=True
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def crop_mouse(img_path):
    img = cv2.imread(img_path)
    #img = Image.fromarray(img)
    img_cropped_list, prob_list = mtcnn(img, return_prob=True)

    if img_cropped_list is not None:
        boxes, _, faces = mtcnn.detect(img, landmarks=True)

        box = boxes[0]
        nose_x, nose_y = int(faces[0][2][0]), int(faces[0][2][1])
        lm_x, lm_y = int(faces[0][3][0]), int(faces[0][3][1])
        rm_x, rm_y = int(faces[0][4][0]), int(faces[0][4][1])
        #print(nose_x, nose_y)
        #print(rm_x, rm_y, lm_x, lm_y)
        # img = cv2.circle(img, (rm_x, rm_y), 1, (0, 0, 255), -1)
        #print(nose_x, int(box[2]), lm_y, rm_y)
        img = img[nose_y + 6: int(box[3]), lm_x: rm_x]

        img = cv2.resize(img, dsize=(34, 26))

        if 'opened' in img_path:
            cv2.imwrite('mouse/opened_mouse/' + img_path.split('\\')[-1], img)
        else:
            cv2.imwrite('mouse/closed_mouse/' + img_path.split('\\')[-1], img)
        # print(img_path.split('\\')[-1])


def get_open_and_closed_mouse(target_path):
    target_file_list = os.listdir(target_path)

    for img_path in target_file_list:
        img_path = os.path.join(target_path, img_path)
        crop_mouse(img_path)


if __name__ == '__main__':
    get_open_and_closed_mouse('mouse/closed_face')
    get_open_and_closed_mouse('mouse/opened_face')
