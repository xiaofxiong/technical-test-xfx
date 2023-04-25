# basic libraies
import os 
import imageio.v2 as imageio
import numpy as np

# libraies for image processing and watershed segmentation
from skimage.filters import threshold_otsu, rank, gaussian, unsharp_mask
from skimage.segmentation import watershed
from skimage.measure import label
from skimage.morphology import disk, binary_closing
from skimage.color import label2rgb
from skimage.util import img_as_ubyte
from skimage.exposure import equalize_adapthist, rescale_intensity
from scipy.ndimage import binary_fill_holes

# library for plotting
# import cv2
import matplotlib.pyplot as plt


# define imput/output paths
img_folder = 'Images'
out_folder = 'Segmentation'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

ID = '13570'

actin = imageio.imread(os.path.join(img_folder, ID + '-actin.tif'))
dna = imageio.imread(os.path.join(img_folder, ID + '-DNA.tif'))

# normalize and enhance the contrast
actin = equalize_adapthist(rescale_intensity(actin))
dna = equalize_adapthist(rescale_intensity(dna))

# segment nucleus to be seeds for seeded watershed segmentation
T_dna = threshold_otsu(dna)
dna_bin = dna > T_dna
seeds = label(dna_bin)


# using morphological opreations to define the background in the actin image
global_thresh = threshold_otsu(actin)
actin_bin = actin > global_thresh
actin_bin = binary_closing(gaussian(actin_bin, 1), disk(1))
actin_bin = binary_fill_holes(actin_bin)
background = np.max(seeds) + 1
seeds[actin_bin == 0] = background

# implement the watershed on the gradient image and remove the background
# markers = np.zeros(actin.shape)
# for l in range(np.max(seeds)):
#     labelmap = seeds * (seeds == l)
#     intensity_img = actin * (seeds == l)
#     local_min = np.min(actin[seeds == l])
#     labelmap[intensity_img > local_min] = 0
#     markers += labelmap.reshape(actin.shape)

# sharpen the image to better find local maximum in gradient
actin_sharpened = unsharp_mask(actin, radius=5, amount=2)
gradient = rank.gradient(img_as_ubyte(actin_sharpened), disk(3))
seg = watershed(gradient, seeds)
seg[seg == background] = 0

image_label_overlay = label2rgb(seg, image=actin, alpha = 0.2, bg_label=0)
seg_rgb = label2rgb(seg, image = None)

# results visualization
fig, axes = plt.subplots(ncols=4, figsize=(18,5))
ax = axes.ravel()
ax[0] = plt.subplot(1, 4, 1)
ax[1] = plt.subplot(1, 4, 2)
ax[2] = plt.subplot(1, 4, 3)
ax[3] = plt.subplot(1, 4, 4)

ax[0].imshow(actin_sharpened, cmap=plt.cm.gray)
ax[0].set_title('Original')

ax[1].imshow(seeds, cmap=plt.cm.gray)   
ax[1].set_title('seeds')

ax[2].imshow(image_label_overlay, cmap=plt.cm.jet)
ax[2].set_title('segmentation')

ax[3].imshow(seg_rgb)
ax[3].set_title('segmentation')

plt.show()


### for visualization only
#for f in os.listdir(img_folder):
#    imageio.imwrite(os.path.join('Visualization', f.replace('tif', 'png')), cv2.normalize(imageio.imread(os.path.join(img_folder, f)), None, 0 ,255 ,cv2.NORM_MINMAX, cv2.CV_8UC3))