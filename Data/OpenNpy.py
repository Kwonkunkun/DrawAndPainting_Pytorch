import numpy as np
import matplotlib.pyplot as plt

img_array = np.load('car.npy')
rimg = np.reshape(img_array[0], (28, 28))

plt.imshow(rimg, cmap="gray")
plt.show()

print(type(rimg))
print(rimg)

