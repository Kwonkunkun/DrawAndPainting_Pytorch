env_name = "../DrawAndPainting_VR/Build/DrawAndPainting_VR"
# env_name = "C:/Users/user/Documents/GitHub/IMG_Transport_Sample/Build/IMG_Transport_Sample"  # Name of the Unity environment binary to launch
train_mode = False  # Whether to run the environment in training or inference mode

import os
import cv2
import sys
import json
import argparse
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from Model.nets import resnet34
from torchvision import transforms

# dataset
from DataUtils.load_data import QD_Dataset
from mlagents.envs.environment import UnityEnvironment

import torch
import numpy as np
import matplotlib.pyplot as plt
import testUnityCommunication

# model - resnet34
from Model.nets import resnet34
#from Model.nets import convnet


if __name__ == '__main__':
    def loadCnnModel():
        #load cnn model
        net = resnet34(20)
        PATH = "./Checkpoints/model.pytorch"
        net.load_state_dict(torch.load(PATH))
        return net

    def PredictSingleImage(net, data):
        net.eval()
        loss_avg = 0.0
        correct = 0

        tensorData = torch.from_numpy(data).type(torch.FloatTensor)
        tensorData = tensorData.view(-1, 1, 28, 28)
        tensorData /= 255.0

        # forward
        output = net(tensorData)
        print(output)

        #predict
        _,output_index = torch.max(output, 1)
        print(output_index)

        result = int(output_index[0])

        return result

    # connect Image
    env = testUnityCommunication.ConnectToUnity()

    # 학습정보 리셋
    env_info = testUnityCommunication.EnvReset(env)
    # 파이썬 구문 반복 Bool
    isPyLoop = testUnityCommunication.PyLoopBoolFromUnity(env_info)

    # 브레인 가져오기
    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]

    while isPyLoop:
        # 이미지 체크할지 bool
        isImgCheck = testUnityCommunication.ImgCheckBoolFromUnity(env_info)

        #print (isImgCheck)
        # 브레인 엑션 사이즈
        action_size = brain.vector_action_space_size

        if isImgCheck:  # 이미지 인식을 할경우 (해당 이미지 번호를 유니티에 반환 1~20)
            net = loadCnnModel()

            # getImage 1 : 이미지를 불러오기
            # singleImagePATH = "./mydraw/glasses4.png"
            # gray = cv2.imread(singleImagePATH, cv2.IMREAD_GRAYSCALE)
            # resizeImage = cv2.resize(gray, (28, 28))

            # get Image 2 :
            resizeImage = testUnityCommunication.GetImgFromUnity(env_info)

            # convert numpy
            transform = np.array(resizeImage)
            # 색 반전
            data = np.where(transform < 25, 255, np.where(transform > 25, 0, transform))

            # image show
            plt.imshow(data, cmap="gray")
            #plt.show()

            # 인식한 애 보여주기
            #print(data)

            # predict
            result = PredictSingleImage(net, data)+1
            env_info = env.step(result)[default_brain]
            #print("env.step 실핼 IF")

        else :  # 이미지 인식을 안할경우 (0을 반환 )
            env_info = env.step(0)[default_brain]
            # print("env.step 실핼 else")

        # 학습정보 리셋
        env_info = testUnityCommunication.EnvReset(env)

        # 파이썬 구문 반복 Bool
        isPyLoop = testUnityCommunication.PyLoopBoolFromUnity(env_info)

    testUnityCommunication.DisconnectToUnity(env)