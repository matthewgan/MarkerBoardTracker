# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# image = mpimg.imread("./seal.jpg")
# image = image[600:2600, 900:2600]

# plt.imshow(image)

# # b = image.copy()
# # # set green and red channels to 0
# # b[:, :, 1] = 0
# # b[:, :, 2] = 0


# # g = image.copy()
# # # set blue and red channels to 0
# # g[:, :, 0] = 0
# # g[:, :, 2] = 0

# r = image.copy()
# # set blue and green channels to 0
# r[:, :, 0] = 0
# r[:, :, 1] = 0
# r[:, :, 2] = 255 - r[:, :, 2]

# r_threshold = cv2.inRange(r, (0,0,140), (0,0,255))

# # RGB - Blue
# # cv2.imshow('B-RGB', b)

# # RGB - Green
# # cv2.imshow('G-RGB', g)

# # RGB - Red
# cv2.imshow('R-RGB', r_threshold)

# cv2.waitKey(0)

# import cv2
# import numpy as np

# #read image
# src = cv2.imread("./seal.jpg")
# clone = src.copy()

# clone = clone[850:2600, 700:2500]
# cv2.imshow("origin", clone)
# print(clone.shape)

# # extract red channel
# red_channel = clone[:,:,2]

# # create empty image with same shape as that of src image
# red_img = np.zeros(clone.shape)

# #assign the red channel of src to empty image
# red_img[:,:,2] = red_channel
# output_img = cv2.inRange(red_img, (0,0,90), (0,0,200))
# # cv2.imshow("red", red_img)
# # cv2.waitKey(0)

# #save image
# cv2.imwrite("./red.png",output_img) 


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('seal.jpg',0)
img = img[850:2600, 700:2500]
# global thresholding
ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# Otsu's thresholding
ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img,(5,5),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
for i in xrange(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()

print(th1.shape)
sp = th1.shape
red_color = (0, 0, 255, 255)

res = np.zeros((sp[0],sp[1],4))

for i in range(sp[0]):
    for j in range(sp[1]):
        if th1[i,j] < 1:
            res[i,j,:] = red_color

# #save image
cv.imwrite("./red.png",res) 

