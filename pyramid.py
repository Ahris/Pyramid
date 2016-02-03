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

"""
NOTES

    # Convert array to PIL image
    PIL = Image.fromarray(np.uint8(img_bw)*255)
"""

import sys
import scipy
import copy
import numpy as np
from math import *
from PIL import Image, ImageChops
from scipy import signal
from skimage import data, filter
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
    height_crop = int(height / 3)
    
    top = img[0:height_crop]                # blue
    mid = img[height_crop:height_crop*2]    # green
    bot = img[height_crop*2:height-1]       # red
    
    return (top, mid, bot)


def multi_scale_align(imgs):
    """
    """
    pass


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
    
    top = cropped_img[0]
    mid = cropped_img[1]
    bot = cropped_img[2]
    min_mid_score = float("inf")
    min_bot_score = float("inf")
    mid_aligned   = []
    bot_aligned   = []
    
    for x in range(-offset, offset):
        for y in range(-offset, offset):
             mid_roll = np.roll(np.roll(mid, y, axis=0), x, axis=1)
             bot_roll = np.roll(np.roll(bot, y, axis=0), x, axis=1)
             
             mid_score = np.sum(np.power(np.array(top)-np.array(mid_roll), 2))
             bot_score = np.sum(np.power(np.array(top)-np.array(bot_roll), 2))
             
             if min_mid_score >= mid_score:
                 min_mid_score = mid_score
                 mid_aligned   = mid_roll
            
             if min_bot_score >= bot_score:
                 min_bot_score = bot_score
                 bot_aligned   = bot_roll
    
    return (top, mid_aligned, bot_aligned)
    

def single_scale_align_edge(cropped_img, offset):
    """Aligns the second and third images 
    
    Exhaustively search over a window of possible displacements (e.g.
    [-15,15] pixels), score each one using some image matching metric,
    and take the displacement with the best score.
    The image matching metric is using sum of squared differences.
    
    Apply sobel's edge detection.
    
    Args: 
        cropped_img: Tuple of three grayscale images
        offset: size of offset to search over

    Returns:
        Tuple of aligned images
    """
    
    top = cropped_img[0]
    mid = cropped_img[1]
    bot = cropped_img[2]
    min_mid_score = float("inf")
    min_bot_score = float("inf")
    mid_offset    = []
    bot_offset    = []

    # Edge detection
    top_edge_sobel = filter.sobel(top)
    mid_edge_sobel = filter.sobel(mid)
    bot_edge_sobel = filter.sobel(bot)
    
    top_array = np.array(top_edge_sobel)
    
    for x in range(-offset, offset):
        for y in range(-offset, offset):
             mid_roll = np.roll(np.roll(mid_edge_sobel, y, axis=0), x, axis=1)
             bot_roll = np.roll(np.roll(bot_edge_sobel, y, axis=0), x, axis=1)
             
             mid_score = np.sum(np.power(top_array - np.array(mid_roll), 2))
             bot_score = np.sum(np.power(top_array - np.array(bot_roll), 2))
             
             if min_mid_score >= mid_score:
                 min_mid_score = mid_score
                 mid_offset   = (x,y)
            
             if min_bot_score >= bot_score:
                 min_bot_score = bot_score
                 bot_offset   = (x,y)
    
    mid_aligned = np.roll(np.roll(mid, bot_offset[1], axis=0), bot_offset[0], 1)
    bot_aligned = np.roll(np.roll(bot, bot_offset[1], axis=0), bot_offset[0], 1)
    return (top, mid_aligned, bot_aligned)


def trim_border(im):
    """ 
    It gets the border colour from the top left pixel, using getpixel, 
    so you don't need to pass the colour.
    Subtracts a scalar from the differenced image, this is a quick way of
    saturating all values under 100, 100, 100 (in my example) to zero. So
    is a neat way to remove any 'wobble' resulting from compression.
    
    https://stackoverflow.com/questions/10615901/trim-whitespace-using-pil
    """
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    
    if bbox:
        return im.crop(bbox)

    
def trim_left_right(img):
    """ Trims black border off the left and right of a single image
    
    Increase contrast 
    Convert image to only black and white
    Samples the center of the image for border width
    Crop image 
    
    Assumes border is within 10% of image
    Assumes white specks within 5% of the image are ok
    
    Args:
        Single image to crop
    
    Returns:
        Cropped single image
    """    
    height         = len(img)
    width          = len(img[0])
    white_ok_width = int(width * .02)
    max_crop_width = int(width * .10)
    threshold      = height * .3
    
    img_bw  = np.around(img/150)
    col_sum = np.sum(img_bw, axis=0)

    crop_left = 0;    
    for i in range(0, max_crop_width):
        if col_sum[i] > threshold and i > white_ok_width:
            crop_left = i
            break

    crop_right = 0;
    for i in range(width-1, width-max_crop_width, -1):
        if col_sum[i] > threshold and i < width-white_ok_width:
            crop_right = width-i
            break
    
    # Crop columns
    to_delete = list(range(width-crop_right*2, width)) 
    img_crop = np.delete(img, list(range(crop_left)), axis=1)
    img_crop = np.delete(img_crop, to_delete, axis=1) 
    return img_crop


def trim_top_bot(imgs):
    """ Trims black border off the top and bot of three images
    
    Increase contrast 
    Convert image to only black and white
    Samples the vertical center of the image for border width
    Crop image 
    
    Assumes border is within 10% of image
    Assumes white specks within 5% of the image are ok
    
    Problem: need to trim each image equally
    
    Args:
        3 images to crop
    
    Returns:
        3 trimmed images
    """   
    height          = len(imgs[0])
    width           = len(imgs[0][0])
    white_ok_height = int(height * .02)
    max_crop_height = int(height * .10)
    threshold       = width * .3
    
    # Contrast and b/w
    img_bw_b = np.around(imgs[0]/150)
    img_bw_g = np.around(imgs[1]/150)
    img_bw_r = np.around(imgs[2]/150)
    
    # White count of each row
    col_sum_b = np.sum(img_bw_b, axis=1)
    col_sum_g = np.sum(img_bw_g, axis=1)
    col_sum_r = np.sum(img_bw_r, axis=1)
    
    crop_top = 0
    done = [False, False, False]
    
    for i in range(0, max_crop_height):
        if i > white_ok_height:
            if col_sum_b[i] > threshold and not done[0]:
                crop_top = i
                done[0] = True
            if col_sum_g[i] > threshold and not done[1]:
                crop_top = i
                done[1] = True
            if col_sum_r[i] > threshold and not done[2]:
                crop_top = i
                done[2] = True
    
    crop_bot = 0
    done = [False, False, False]
    
    for i in range(height-1, height-max_crop_height, -1):
        if i < height-white_ok_height:
            if col_sum_b[i] > threshold and not done[0]:
                crop_bot = height-i;
                done[0] = True
            if col_sum_g[i] > threshold and not done[1]:
                crop_bot = height-i;
                done[1] = True
            if col_sum_r[i] > threshold and not done[2]:
                crop_bot = height-i;
                done[2] = True

    # Crop columns
    final_imgs = [None]*3
    for i in range(0,3):
        img_crop = np.delete(imgs[i], list(range(crop_top)), 0)
        img_crop = np.delete(img_crop, list(range(height-crop_bot, height)), 0)
        final_imgs[i] = img_crop
        
    # Make sure all the images are the same height - chop from the bottom!
    min_height = min(len(final_imgs[0]), len(final_imgs[1]), len(final_imgs[2]))
    for i in range(0,3):
        final_imgs[i] = final_imgs[i][0:min_height]
    
    return final_imgs
    
    
def imshow(img):    
    """ 
    Show an image with range 0 to 255
    """
    plt.imshow(img, plt.get_cmap('gray'), vmin = 0, vmax=255)
    plt.show()
    

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
    Also does manual color balancing
    
    Args:
        img: tuple of three channels - Blue, Green, then Red
    
    Return:
        Single full color image
    """
    height   = len(imgs[0])
    width    = len(imgs[0][0])
    rgbArray = np.zeros((height, width, 3), "uint8")
    rgbArray[..., 0] = imgs[2]*.9   # red
    rgbArray[..., 1] = imgs[1]      # green
    rgbArray[..., 2] = imgs[0]*.9   # blue
    return Image.fromarray(rgbArray)
    

def main(argv = sys.argv):
    #img = plt.imread("prk2000000780.jpg") #argv[1]
    
    PIL_img       = Image.open("images/prk2000000780.jpg")
    trimmed_img   = trim_border(PIL_img) # remove white border
    ndarray       = np.asarray(trimmed_img, dtype=np.uint8)
    trimmed_img   = trim_left_right(ndarray)
    cropped_img   = crop_thirds(trimmed_img)
    retrimmed_img = trim_top_bot(cropped_img)
    aligned_img   = single_scale_align_edge(retrimmed_img, 15)    
    final_img     = overlay_images(aligned_img)
    
    imshow(final_img)
    return 0    


if __name__ == "__main__":
    # Exit status is the return value from main
    # sys.exit(main())
    main()