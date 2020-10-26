env_name = "../DrawAndPainting_VR/Build/DrawAndPainting_VR"
# env_name = "C:/Users/user/Documents/GitHub/IMG_Transport_Sample/Build/IMG_Transport_Sample"  # Name of the Unity environment binary to launch
train_mode = False  # Whether to run the environment in training or inference mode

import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2
import datetime

from PIL import Image
from matplotlib import cm

from mlagents.envs.environment import UnityEnvironment

def ConnectToUnity():
    print("ConnectToUnity")

    # 파이썬 버전 검사
    print("Python version:")
    print(sys.version)
    if (sys.version_info[0] < 3):
        raise Exception("ERROR: ML-Agents Toolkit (v0.3 onwards) requires Python 3")

    # 파일의 경로를 가져와서 유니티를 킨다.
    # print("파일의 경로를 가져와서 유니티를 킨다.")
    env = UnityEnvironment(file_name=env_name)

    return env

def DisconnectToUnity(env):
    print("DisconnectToUnity")
    env.close()

def GetImgFromUnity(env_info):
    print("GetImgFromUnity")

    img = np.uint8(255 *np.array(env_info.visual_observations[0]))

    #img reshape
    img = np.reshape(img, (28, 28))
    return img


# 학습정보 리셋
def EnvReset(env):
    # 첫번째 브레인을 가저온다.
    #print("첫번째 브레인을 가저온다.")
    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]

    # 학습 정보 리셋
    env_info = env.reset(train_mode=train_mode)[default_brain]

    return env_info

# 유니티에서 while 문에대한 bool 값을 정해줌
def PyLoopBoolFromUnity(env_info):
    #print(env_info.vector_observations[0][0])
    return env_info.vector_observations[0][0]


# 유니티에서 이미지 체크할지 bool 값을 정해줌
def ImgCheckBoolFromUnity(env_info):
    #print(env_info.vector_observations[0][1])
    return env_info.vector_observations[0][1]
