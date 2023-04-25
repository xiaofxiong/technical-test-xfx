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


def seeded_watershed(actin, dna):
    '''
    function to implement the seeded watershed segmentation given a pair of actin and DNA images.
    '''

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
    actin_sharpened = unsharp_mask(actin, radius=5, amount=2) # sharpen the image to better find local maximum in gradient
    gradient = rank.gradient(img_as_ubyte(actin_sharpened), disk(3))
    seg = watershed(gradient, seeds)
    seg[seg == background] = 0

    image_label_overlay = label2rgb(seg, image=actin, alpha = 0.2, bg_label=0)
    seg_rgb = label2rgb(seg, image = None)

    return image_label_overlay, seg_rgb


if __name__ == "__main__":

    # define imput/output paths
    img_folder = 'Images'
    out_folder = 'Segmentation'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # extract unique sample IDs
    IDs = [f.split('-')[0] for f in os.listdir(img_folder)]
    IDs = list(set(IDs))

    for ID in IDs:
        # read actin and DNA images for a given ID
        print(f'segmenting (ID): {ID}')
        actin = imageio.imread(os.path.join(img_folder, ID + '-actin.tif'))
        dna = imageio.imread(os.path.join(img_folder, ID + '-DNA.tif'))

        # normalize and enhance the contrast
        actin = equalize_adapthist(rescale_intensity(actin))
        dna = equalize_adapthist(rescale_intensity(dna))    

        # apply the segmentation function
        image_label_overlay, seg_rgb = seeded_watershed(actin, dna)

        # save output images
        imageio.imwrite(os.path.join(out_folder, ID + '_overlay.tif'), img_as_ubyte(image_label_overlay))
        imageio.imwrite(os.path.join(out_folder, ID + '_seg.tif'), img_as_ubyte(seg_rgb))