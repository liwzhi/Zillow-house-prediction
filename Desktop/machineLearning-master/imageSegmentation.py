# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 08:03:24 2015

@author: weizhi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 01 10:33:28 2015

@author: Algorithm 001
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as nd

from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
import cv2
path2 = 'D:/2nd day/1stCut_1C/Images/2015-06-11_10.45.40/Image_I0000_F1_420_20_2015-06-11_10.46.12.066.tiff'
path1 = 'F:/MSI/DataPath/Pig1/Pig1_1B___Burn_1/Image_I0000_F1_Filter 1_1A_health_2014-05-20_11.08.19.146.tiff'
path3 = 'F:/MSI/DataPath/Pig1/Pig1_1B___Burn_1/Image_I0000_F3_Filter 3_2014-05-20_11.08.19.194.tiff'
path4 = 'F:/MSI/DataPath/Pig1/Pig1_1B___Burn_1/Image_I0000_F8_Filter 8_2014-05-20_11.08.19.238.tiff'

path1 = path2
path3 = path2
path4 = path2
img1 = cv2.imread(path1,-1)
img1 = img1[:,:1392]
img1 = cv2.resize(img1,(256,256))
img1 = img1/(2**(4))


img2 = cv2.imread(path2,-1)
img2 = img2[:,:1392]
img2 = cv2.resize(img2,(256,256))
img2 = img2/(2**(4))

img3 = cv2.imread(path3,-1)
img3 = img3[:,:1392]
img3 = cv2.resize(img3,(256,256))
img3 = img3/(2**(4))


img4 = cv2.imread(path4,-1)
img4 = img4[:,:1392]
img4 = cv2.resize(img4,(256,256))
img4 = img4/(2**(4))
#img = cv2.resize(img,(256,256))
#img = img/(2**(4))
#img = np.uint8(img)



shrink = (slice(0, None, 3), slice(0, None, 3))
brick = img_as_float(data.load('brick.png'))[shrink]
grass = img_as_float(data.load('grass.png'))[shrink]
wall = img_as_float(data.load('rough-wall.png'))[shrink]
image_names = ('F1', 'F2', 'F3','F4')
images = (img1,img2,img3,img4)

# prepare reference features


def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(nd.convolve(image, np.real(kernel), mode='wrap')**2 +
                   nd.convolve(image, np.imag(kernel), mode='wrap')**2)
    
#    return nd.convolve(image, np.real(kernel), mode='wrap')**2

# Plot a selection of the filter bank kernels and their responses.
results = []
results1 = []
kernel_params = []
#img = cv2.resize(img,(256,256))
for theta in (np.arange(0,8,0.5)):
    theta = theta / 4. * np.pi
    for frequency in (0.1, 0.2):
        kernel = gabor_kernel(frequency, theta=theta)
        params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
        kernel_params.append(params)
        # Save kernel and the power image for each image
        results1.append(power(img4, kernel))

#%% output        
imgResult = results1[16:]
imgMax = np.zeros((256,256,16))
for i in range(16):
    imgMax[:,:,i] = imgResult[i]
imgmax = np.reshape(imgMax,(256*256,16))
imgmax = np.reshape(np.amax(imgmax,axis=1),(256,256))


plt.figure;plt.imshow(imgmax,cmap=plt.cm.gray, interpolation='nearest')
plt.figure;plt.imshow(img4,cmap=plt.cm.gray, interpolation='nearest')


import cv2

edges = cv2.Canny(np.uint8(img[:,:,0]),100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()


#%%
fig, axes = plt.subplots(nrows=10, ncols=6, figsize=(8, 9))
plt.gray()

fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)

axes[0][0].axis('off')

# Plot original images
for label, img, ax in zip(image_names, images, axes[0][1:]):
    ax.imshow(img)
    ax.set_title(label, fontsize=9)
    ax.axis('off')

for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
    # Plot Gabor kernel
    ax = ax_row[0]
    ax.imshow(np.real(kernel), cmap=plt.cm.jet, interpolation='nearest')
    ax.set_ylabel(label, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot Gabor responses with the contrast normalized for each filter
    vmin = np.min(powers)
    vmax = np.max(powers)
    for patch, ax in zip(powers, ax_row[1:]):
        ax.imshow(patch, vmin=vmin, vmax=vmax)
        ax.axis('off')

plt.show()



#%%
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

from skimage import data
from skimage import color
from skimage.util.shape import view_as_windows
from skimage.util.montage import montage2d

np.random.seed(42)

patch_shape = 8, 8
n_filters = 49


def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(nd.convolve(image, np.real(kernel), mode='wrap')**2 +
                   nd.convolve(image, np.imag(kernel), mode='wrap')**2)



results = []
kernel_params = []
for theta in (0, 1,2,3,4,5,6,7):
    theta = theta / 4. * np.pi
    for frequency in (0.1, 0.4):
        kernel = gabor_kernel(frequency, theta=theta)
        params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
        kernel_params.append(params)
        # Save kernel and the power image for each image
        results.append((kernel, [power(image_gray, kernel)]))

output = power(image_gray,kernel)
#astro = color.rgb2gray(data.astronaut())
astro = output

# -- filterbank1 on original image
patches3 = view_as_windows(astro, patch_shape)
patches1 = patches3.reshape(-1, patch_shape[0] * patch_shape[1])[::8]
fb3, _ = kmeans2(patches1, n_filters, minit='points')
fb1 = fb3.reshape((-1,) + patch_shape)
fb1_montage = montage2d(fb1, rescale_intensity=True)

# -- filterbank2 LGN-like image
astro_dog = ndi.gaussian_filter(astro, .5) - ndi.gaussian_filter(astro, 1)
patches2 = view_as_windows(astro_dog, patch_shape)
patches2 = patches2.reshape(-1, patch_shape[0] * patch_shape[1])[::8]
fb2, _ = kmeans2(patches2, n_filters, minit='points')
fb2 = fb2.reshape((-1,) + patch_shape)
fb2_montage = montage2d(fb2, rescale_intensity=True)

# --
fig, axes = plt.subplots(2, 2, figsize=(7, 6))
ax0, ax1, ax2, ax3 = axes.ravel()

ax0.imshow(astro, cmap=plt.cm.gray)
ax0.set_title("Image (original)")

ax1.imshow(fb1_montage, cmap=plt.cm.gray, interpolation='nearest')
ax1.set_title("K-means filterbank (codebook)\non original image")

ax2.imshow(astro_dog, cmap=plt.cm.gray)
ax2.set_title("Image (LGN-like DoG)")

ax3.imshow(fb2_montage, cmap=plt.cm.gray, interpolation='nearest')
ax3.set_title("K-means filterbank (codebook)\non LGN-like DoG image")

for ax in axes.ravel():
    ax.axis('off')

fig.subplots_adjust(hspace=0.3)
plt.show()


#%% GLCM teature features extractoin


import matplotlib.pyplot as plt

from skimage.feature import greycomatrix, greycoprops
from skimage import data


PATCH_SIZE = 21

# open the camera image
image = np.uint8(image_gray).T


plt.figure()
plt.imshow(img1,cmap=plt.cm.gray)

# select some patches from grassy areas of the image
grass_locations = [(132, 142), (130, 152), (154, 135), (164, 139)]
grass_patches = []
for loc in grass_locations:
    grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])

# select some patches from sky areas of the image
sky_locations = [(45, 106), (75, 76), (48, 187), (73, 128)]
sky_patches = []
for loc in sky_locations:
    sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                             loc[1]:loc[1] + PATCH_SIZE])

# compute some GLCM properties each patch
xs = []
ys = []
for i, patch in enumerate(grass_patches + sky_patches):
    glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
    
#skimage.feature.greycomatrix(image, distances, angles, levels=256, symmetric=False, normed=False)
    xs.append(greycoprops(glcm, 'energy')[0, 0])
    ys.append(greycoprops(glcm, 'correlation')[0, 0])

# create the figure
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)
ax.imshow(image,  interpolation='nearest',
          vmin=0, vmax=255)
for (y, x) in grass_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
for (y, x) in sky_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
        label='Grass')
ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
        label='Sky')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLVM Correlation')
ax.legend()

# display the image patches
for i, patch in enumerate(grass_patches):
    ax = fig.add_subplot(3, len(grass_patches), len(grass_patches)*1 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
              vmin=0, vmax=255)
    ax.set_xlabel('Grass %d' % (i + 1))

for i, patch in enumerate(sky_patches):
    ax = fig.add_subplot(3, len(sky_patches), len(sky_patches)*2 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
              vmin=0, vmax=255)
    ax.set_xlabel('Sky %d' % (i + 1))


# display the patches and plot
fig.suptitle('Grey level co-occurrence matrix features', fontsize=14)
plt.show()

#%% bob detection

from matplotlib import pyplot as plt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.color import rgb2gray

#image = data.hubble_deep_field()[0:500, 0:500]
image = cv2.resize(img1,(256,256))
image_gray = rgb2gray(image)

blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

blobs_list = [blobs_log, blobs_dog, blobs_doh]
colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          'Determinant of Hessian']
sequence = zip(blobs_list, colors, titles)

for blobs, color, title in sequence:
    fig, ax = plt.subplots(1, 1)
    ax.set_title(title)
    ax.imshow(image, interpolation='nearest')
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax.add_patch(c)

plt.show()