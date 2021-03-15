'''
torch와 torchvision version info
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
'''

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import mtcnn
import cv2
import time
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import numpy as np

mtcnn = MTCNN(image_size=240, margin=0, keep_all=True,
              min_face_size=40)  # keep_all=True
resnet = InceptionResnetV1(pretrained='vggface2').eval()

dest_dir = 'data'


def collate_fn(x):
    return x[0]


def detect_face(mode):
    if mode == 'train':
        print("making dataset for TRAIN")
        dataset = datasets.ImageFolder('photos/train')

    elif mode == 'val':
        print("making dataset for VAL")
        dataset = datasets.ImageFolder('photos/val')

    # accessing names of peoples from folder names
    idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}

    loader = DataLoader(dataset, collate_fn=collate_fn)

    i = 1
    prev_name = ''
    for img, idx in loader:
        cur_name = idx_to_class[idx]

        if cur_name != prev_name:
            i = 1
        else:
            i += 1

        img_cropped_list, prob_list = mtcnn(img, return_prob=True)

        if img_cropped_list is not None:
            boxes, _, faces = mtcnn.detect(
                img, landmarks=True)  # detect는 PIL 상태에서 진행
            box = boxes[0]
            img = np.array(img)  # 이미지 다루기 편하게 opencv로 변환
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img[int(box[1]): int(box[3]), int(
                box[0]): int(box[2])]  # 얼굴 부분만 자르기
            img = cv2.resize(img, dsize=(160, 160))  # 160x160으로 resize
            flipped_img = cv2.flip(img, 1)

            if mode == 'train':
                new_name = 't_' + cur_name
                print(dest_dir + '/train/' +
                      cur_name + '/' + new_name + str(i) + '.jpg')
                cv2.imwrite(
                    dest_dir + '/train/' + cur_name + '/' + new_name + str(i) + '.jpg', img)
            elif mode == 'val':
                new_name = 'v_' + cur_name
                print(dest_dir + '/val/' +
                      cur_name + '/' + new_name + str(i) + '.jpg')
                cv2.imwrite(
                    dest_dir + '/val/' + cur_name + '/' + new_name + str(i) + '.jpg', img)

        prev_name = cur_name


if __name__ == "__main__":
    detect_face('train')
    detect_face('val')
