import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
import matplotlib.pyplot as plt



#sourceImage = Image.open('CameraScreenShot.png')
#resizeImage = sourceImage.resize((28, 28))

#load image and convert grayscale
gray = cv2.imread('airplane3.png',cv2.IMREAD_GRAYSCALE)

plt.imshow(gray, cmap="gray")
plt.show()

resizeImage = cv2.resize(gray, (28, 28), Image.BILINEAR)

#show image
cv2.imshow('Gray', resizeImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

#convert numpy and resize
data = np.array(resizeImage)
print(data.shape)
tensorData = torch.from_numpy(data).type(torch.LongTensor)
print(tensorData)

plt.imshow(resizeImage, cmap="gray")
plt.show()

