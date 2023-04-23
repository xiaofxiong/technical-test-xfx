# basic libraies
import os 
import imageio
import numpy as np

# libraies for image processing and watershed segmentation
from skimage.filters import threshold_otsu, rank, gaussian
from skimage.segmentation import watershed
from skimage.measure import label
from skimage.morphology import disk, binary_opening, binary_closing
from skimage.color import label2rgb
from skimage.util import img_as_ubyte
from skimage import exposure
from scipy.ndimage import binary_fill_holes

# library for plotting
import matplotlib.pyplot as plt

'''
def seeded_watershed(actin, dna):
    # define seeds for the seeded watershed using nucleus
    T_dna = threshold_otsu(dna)
    dna_bin = dna > T_dna
    seeds = label(binary_opening(dna_bin, disk(3)))

    # using morphological opreations to define the background in the actin image
    global_thresh = threshold_otsu(actin)
    actin_bin = actin > global_thresh
    actin_bin = binary_closing(gaussian(actin_bin, 1), disk(5))
    actin_bin = binary_fill_holes(actin_bin)
    background = np.max(seeds) + 1
    seeds[actin_bin == 0] = background

    # implement the watershed on the gradient image and remove the background
    gradient = rank.gradient(actin, disk(5))
    seg = watershed(gradient, seeds)
    seg[seg == background] = 0

    image_label_overlay = label2rgb(seg, image=exposure.rescale_intensity(actin), alpha = 0.1, bg_label=0)
    seg_rgb = label2rgb(seg, image = None)

    return image_label_overlay, seg_rgb
'''

img_folder = 'Images'
out_folder = 'Segmentation'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

ID = '00735'

actin = imageio.imread(os.path.join(img_folder, ID + '-actin.tif'))
dna = imageio.imread(os.path.join(img_folder, ID + '-DNA.tif'))
ph3 = imageio.imread(os.path.join(img_folder, ID + '-pH3.tif'))

# define seeds for the seeded watershed using nucleus
T_dna = threshold_otsu(dna)
dna_bin = dna > T_dna
seeds = label(binary_opening(dna_bin, disk(3)))

# using morphological opreations to define the background in the actin image
global_thresh = threshold_otsu(actin)
actin_bin = actin > global_thresh
actin_bin = binary_closing(gaussian(actin_bin, 1), disk(5))
actin_bin = binary_fill_holes(actin_bin)
background = np.max(seeds) + 1
seeds[actin_bin == 0] = background

# implement the watershed on the gradient image and remove the background
gradient = rank.gradient(actin, disk(5))
seg = watershed(gradient, seeds)
seg[seg == background] = 0

image_label_overlay = label2rgb(seg, image=exposure.rescale_intensity(actin), alpha = 0.1, bg_label=0)
seg_rgb = label2rgb(seg, image = None)

#image_label_overlay, seg_rgb = seeded_watershed(actin, dna)

fig, axes = plt.subplots(ncols=4, figsize=(18,5))
ax = axes.ravel()
ax[0] = plt.subplot(1, 4, 1)
ax[1] = plt.subplot(1, 4, 2)
ax[2] = plt.subplot(1, 4, 3)
ax[3] = plt.subplot(1, 4, 4)

ax[0].imshow(actin, cmap=plt.cm.gray)
ax[0].set_title('Original')

ax[1].imshow(seeds, cmap=plt.cm.gray)   
ax[1].set_title('seeds')

ax[2].imshow(seg_rgb, cmap=plt.cm.jet)
ax[2].set_title('segmentation')

ax[3].imshow(image_label_overlay)
ax[3].set_title('segmentation')

plt.show()

imageio.imwrite(os.path.join(out_folder, ID + '_overlay.tif'), img_as_ubyte(image_label_overlay))
imageio.imwrite(os.path.join(out_folder, ID + '_seg.tif'), img_as_ubyte(seg_rgb))


'''
# extract unique sample IDs
IDs = [f.split('-')[0] for f in os.listdir(img_folder)]
IDs = list(set(IDs))

for ID in IDs:
    # three corresponding images for a given ID
    actin = imageio.imread(os.path.join(img_folder, ID + '-actin.tif'))
    dna = imageio.imread(os.path.join(img_folder, ID + '-DNA.tif'))
    ph3 = imageio.imread(os.path.join(img_folder, ID + '-pH3.tif'))
'''




### for visualization only
#for f in os.listdir(img_folder):
#    imageio.imwrite(os.path.join('Visualization', f.replace('tif', 'png')), cv2.normalize(imageio.imread(os.path.join(img_folder, f)), None, 0 ,255 ,cv2.NORM_MINMAX, cv2.CV_8UC3))

