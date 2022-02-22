import os
import SimpleITK as sitk
os.chdir('E:\Dropbox (UCL)\PPFE\Exported data\GLE689_top_seg')
import numpy as np
# load and display an image with Matplotlib
# from matplotlib import image
from matplotlib import pyplot as plt
from PIL import Image


# load image as pixel array

GLE_labels_vol = np.zeros((4096, 4096, 512))

for i in range(2):
    #string = "GLE698_top_segmentation000" + str(i) + ".tiff"
    image = Image.open("GLE698_top_segmentation000" + str(i) + ".tiff")
    image = np.array(image).squeeze()
    GLE_labels_vol[:, :, i] = image

    #image = image.imread('GLE698_top_segmentation0001'.tiff')
    # GLE_labels = np.zeros((4096, 4096))

    # GLE_labels[:image.shape[0], :image.shape[1]] = image
    # GLE_labels = np.stack((GLE_labels,GLE_labels)).shape
    # np.split(GLE_labels_vol, 8)


# summarize shape of the pixel array
print(GLE_labels_vol.dtype)
print(GLE_labels_vol.shape)
# display the array of pixels as an image
plt.imshow(GLE_labels_vol[:, :, 0])
plt.show()