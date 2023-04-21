import os 
import imageio

#import cv2
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.measure import label

import matplotlib.pyplot as plt

img_folder = 'Images'
out_folder = 'Segmentation'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

# extract unique sample IDs
IDs = [f.split('-')[0] for f in os.listdir(img_folder)]
IDs = list(set(IDs))

ID = '00733'

actin = imageio.imread(os.path.join(img_folder, ID + '-actin.tif'))
dna = imageio.imread(os.path.join(img_folder, ID + '-DNA.tif'))
ph3 = imageio.imread(os.path.join(img_folder, ID + '-pH3.tif'))

T = threshold_otsu(dna)
dna_T = dna > T
connected_components = label(dna_T)

fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
ax = axes.ravel()
ax[0] = plt.subplot(1, 3, 1)
ax[1] = plt.subplot(1, 3, 2)
ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

ax[0].imshow(dna, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].imshow(dna_T, cmap=plt.cm.gray)
ax[1].set_title('Otsu')
ax[0].axis('off')

ax[2].imshow(connected_components, cmap=plt.cm.jet)
ax[2].set_title('connected components')
ax[2].axis('off')

plt.show()
'''
for ID in IDs:
    actin = imageio.imread(os.path.join(img_folder, ID + '-actin.tif'))
    dna = imageio.imread(os.path.join(img_folder, ID + '-DNA.tif'))
    ph3 = imageio.imread(os.path.join(img_folder, ID + '-pH3.tif'))

    actin_T = threshold_otsu(actin)
    connected_components = label(actin_T)
'''




### for visualization only
#for f in os.listdir(img_folder):
#    imageio.imwrite(os.path.join('Visualization', f.replace('tif', 'png')), cv2.normalize(imageio.imread(os.path.join(img_folder, f)), None, 0 ,255 ,cv2.NORM_MINMAX, cv2.CV_8UC3))
