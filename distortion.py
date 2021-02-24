# 배럴 왜곡, 핀쿠션 왜곡 (reamp_barrel.py)

import cv2
import numpy as np
import os


def barrel_distortion(img_path):
    # 왜곡 계수 설정 ---①
    k1, k2, k3 = 0.5, 0.2, 0.0  # 배럴 왜곡
    # k1, k2, k3 = -0.3, 0, 0    # 핀큐션 왜곡

    img = cv2.imread(img_path)
    rows, cols = img.shape[:2]

    # 매핑 배열 생성 ---②
    mapy, mapx = np.indices((rows, cols), dtype=np.float32)

    # 중앙점 좌표로 -1~1 정규화 및 극좌표 변환 ---③
    mapx = 2*mapx/(cols-1)-1
    mapy = 2*mapy/(rows-1)-1
    r, theta = cv2.cartToPolar(mapx, mapy)

    # 방사 왜곡 변영 연산 ---④
    ru = r*(1+k1*(r**2) + k2*(r**4) + k3*(r**6))

    # 직교좌표 및 좌상단 기준으로 복원 ---⑤
    mapx, mapy = cv2.polarToCart(ru, theta)
    mapx = ((mapx + 1)*cols-1)/2
    mapy = ((mapy + 1)*rows-1)/2
    # 리매핑 ---⑥
    distorted = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    return distorted


def make_distorted_photos():
    dir_list = os.listdir('photos')

    for dir_name in dir_list:
        dir_path = os.path.join('photos', dir_name)

        imgs = os.listdir(dir_path)
        for img in imgs:

            if img[:2] == 'd_':
                continue

            img_path = os.path.join(dir_path, img)

            tmp = barrel_distortion(img_path)
            distored_img_path = dir_path + '\\d_' + img
            cv2.imwrite(distored_img_path, tmp)


def make_distorted_mouses():
    dir_list = os.listdir('mouse')

    for dir_name in dir_list:
        if dir_name == 'closed_face' or dir_name == 'opened_face':
            dir_path = os.path.join('mouse', dir_name)
            print(dir_name)

            imgs = os.listdir(dir_path)
            cnt = len(imgs)
            i = 1
            for img in imgs:
                print(f'{i}/{cnt}')
                if img[:2] == 'd_':
                    continue

                img_path = os.path.join(dir_path, img)

                tmp = barrel_distortion(img_path)
                distored_img_path = dir_path + '\\d_' + img
                cv2.imwrite(distored_img_path, tmp)
                i += 1
