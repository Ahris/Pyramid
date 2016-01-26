#!/usr/bin/env python

""" pyramid.py: Image Alignment with Pyramids

Computational Photography
Feburary 3, 2016
Alice Wang

About:
    http://www.cs.wustl.edu/~pless/classes/555/proj1/
    Automatic color aligning and compositing of the Prokudin-Gorskii photo
    collection
    
Usage:
    ./pyramid.py image_path.jpg

"""

import sys
from PIL import Image
from scipy import signal
from math import *
import numpy as np
from skimage import data
import matplotlib.pyplot as plt

__author__  = "Alice Wang"
__email__   = "awang26@wustl.edu"


def crop_thirds(img):
    """Crops an image into three verticle parts.
    
    Args: 
        img: Single channel image

    Returns:
        Tuple of cropped images 
    """
    height      = len(img)
    width       = len(img[0])
    height_crop = height / 3
    
    # print("height: ", height)
    # print("width: ", width)
    
    top = img[0:height_crop]                # blue
    mid = img[height_crop:height_crop*2]    # green
    bot = img[height_crop*2:height-1]       # red
    
    return (top, mid, bot)


def single_scale_align(cropped_img, offset):
    """Aligns the second and third images 
    
    Exhaustively search over a window of possible displacements (e.g.
    [-15,15] pixels), score each one using some image matching metric,
    and take the displacement with the best score.
    The image matching metric is using sum of squared differences.
    
    Args: 
        cropped_img: Tuple of three grayscale images
        offset: size of offset to search over

    Returns:
        Tuple of aligned images
    """
    
    top           = cropped_img[0]
    mid           = cropped_img[1]
    bot           = cropped_img[2]
    mid_offset    = (0,0)
    bot_offset    = (0,0)
    mid_aligned   = []
    bot_aligned   = []
    min_mid_score = float("inf")
    min_bot_score = float("inf")
    
    top_norm = top/np.linalg.norm(top)
    
    for x in range(-offset, offset):
        for y in range(-offset, offset):
             mid_roll = np.roll(np.roll(cropped_img[1], y, axis=0), x, axis=1)
             bot_roll = np.roll(np.roll(cropped_img[2], y, axis=0), x, axis=1)
             
             # mid_score = np.dot(top/np.norm(top), mid_roll/np.norm(mid_roll))
             # bot_score = np.dot(top/np.norm(top), bot_roll/np.norm(bot_roll))
             
             mid_score = np.sum(np.power(top - mid_roll, 2))
             bot_score = np.sum(np.power(top - bot_roll, 2))
             
             min_mid_score = min(min_mid_score, mid_score)
             min_bot_score = min(min_bot_score, bot_score)
             
             if min_mid_score == mid_score:
                 mid_offset  = (x,y)
                 mid_aligned = mid_roll
            
             if min_bot_score == bot_score:
                 bot_offset  = (x,y)
                 bot_aligned = bot_roll
    
    return (top, mid_aligned, bot_aligned)


def concat_images(imga, imgb):
    """
    Combines two single channel image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width), dtype=np.uint8)
    new_img[:ha,:wa] = imga
    new_img[:hb,wa:wa+wb] = imgb
    return new_img


def concat_n_images(image_list):
    """
    Combines N color images from a list of images.
    """
    output = None
    for i, img in enumerate(image_list):
        if i == 0:
            output = img
        else:
            output = concat_images(output, img)
            
    return output


def overlay_images(imgs):
    """ Combine three image channels into one color image
    
    Args:
        img: tuple of three channels - Blue, Green, then Red
    
    Return:
        Single full color image
    """

    height   = len(imgs[0])
    width    = len(imgs[0][0])
    rgbArray = np.zeros((height, width, 3), "uint8")
    rgbArray[..., 0] = imgs[2]   # red
    rgbArray[..., 1] = imgs[1]   # green
    rgbArray[..., 2] = imgs[0]   # blue
    return Image.fromarray(rgbArray)
    

def main(argv = sys.argv):
    img = plt.imread("prk2000000199.jpg") #argv[1]
    # plt.imshow(img, plt.get_cmap('gray'), vmin = 0, vmax = 255)
    # plt.show()
    # 
    # arr = [[1, 2, 3], [4, 5, 6]]
    # print(np.roll(np.roll(arr, 1, axis=0), 1, axis=1))

    cropped_img = crop_thirds(img)
    aligned_img = single_scale_align(cropped_img, 15)
    final_img   = overlay_images(aligned_img)
    
    #concat = concat_n_images(cropped_img)
    plt.imshow(final_img, plt.get_cmap('gray'), vmin = 0, vmax = 255)
    plt.show()
    
    # input("Press ENTER to exit.")
    return 0    


if __name__ == "__main__":
    # Exit status is the return value from main
    # sys.exit(main())
    main()