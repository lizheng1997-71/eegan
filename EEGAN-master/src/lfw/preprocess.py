import os

import glob

import shutil

import cv2

import numpy as np

from tqdm import tqdm

import requests

import tarfile
from skimage import io

def flatten(input_list):
    output_list = []
    while True:
        if input_list == []:
            break
        for index, i in enumerate(input_list):

            if type(i)== list:
                input_list = i + input_list[index+1:]
                break
            else:
                output_list.append(i)
                input_list.pop(index)
                break

    return output_list


def preprocess():

    print('... loading data')
    os.mkdir('E:/data/raw')
    os.mkdir('E:/data/raw/train')
    os.mkdir('E:/data/raw/test')
    os.mkdir('E:/data/npy')
    # if os.path.exists(r'E:/data/raw'):
    #     os.rmdir('E:/data/raw')
    # else:
    #     os.mkdir('E:/data/raw')
    # if os.path.exists(r'E:/data/raw/train'):
    #     os.rmdir(r'E:/data/raw/train')
    # else:
    #     os.mkdir('E:/data/raw/train')
    # if os.path.exists(r'E:/data/raw/test'):
    #     os.rmdir(r'E:/data/raw/test')
    # else:
    #     os.mkdir('E:/data/raw/test')
    # #if os.path.exists(r'E:/data/npy'):
    #  #   os.rmdir('E:/data/npy')
    # else:
    #     os.mkdir('E:/data/npy')''''''

    imagepath1 = []
    imagepath2 = []

    persons = glob.glob('E:\\遥感数据集\\UCMerced_LandUse\\images\\*')

    for person in persons:
        for imagepath in [glob.glob(os.path.join(person,'*'))]:
            imagepath1.append(imagepath)
    imagepath2.extend(imagepath1)

    #path1 = np.array([e for x in [glob.glob(os.path.join(person, '*'))for person in persons] for e in x])
    #paths = np.array([c for t in [glob.glob(os.path.join(killer, '*')) for killer in path1] for c in t])
    imagepaths = flatten(imagepath2)
    np.random.shuffle(imagepaths)



    r = int(len(imagepaths) * 0.99)

    train_paths = imagepaths[:r]

    test_paths = imagepaths[r:]



    x_train = []

    pbar = tqdm(total=(len(train_paths)))

    for i, d in enumerate(train_paths):

        pbar.update(1)

        img = io.imread(d)

        face = img

        face = cv2.resize(face, (96, 96))

        if face is None:

            continue

        x_train.append(face)

        name = "{}.png".format("{0:05d}".format(i))

        imgpath9 = os.path.join('E:/data/raw/train', name)

        cv2.imwrite(imgpath9, face)

    pbar.close()



    x_test = []

    pbar = tqdm(total=(len(test_paths)))

    for i, d in enumerate(test_paths):

        pbar.update(1)

        img = io.imread(d)

        face = img

        face = cv2.resize(face, (96, 96))

        if face is None:

            continue

        x_test.append(face)

        name = "{}.png".format("{0:05d}".format(i))

        imgpath8 = os.path.join('E:/data/raw/test', name)

        cv2.imwrite(imgpath8, face)

    pbar.close()



    x_train = np.array(x_train, dtype=np.uint8)

    x_test = np.array(x_test, dtype=np.uint8)

    np.save('E:/data/npy/x_train.npy', x_train)

    np.save('E:/data/npy/x_test.npy', x_test)





def main():

    preprocess()





if __name__ == '__main__':

    main()