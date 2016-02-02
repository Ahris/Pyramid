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
    
    top = cropped_img[0]
    mid = cropped_img[1]
    bot = cropped_img[2]
    min_mid_score = float("inf")
    min_bot_score = float("inf")
    mid_aligned   = []
    bot_aligned   = []
    
    # top_norm = top/np.linalg.norm(top)
    
    for x in range(-offset, offset):
        for y in range(-offset, offset):
             mid_roll = np.roll(np.roll(mid, y, axis=0), x, axis=1)
             bot_roll = np.roll(np.roll(bot, y, axis=0), x, axis=1)
             
             # mid_score = np.dot(top/np.linalg.norm(top, ord="fro"), mid_roll/np.linalg.norm(mid_roll, ord="fro"))
             # bot_score = np.dot(top/np.linalg.norm(top, ord="fro"), bot_roll/np.linalg.norm(bot_roll, ord="fro"))
             
             mid_score = np.sum(np.power(np.array(top) - np.array(mid_roll), 2))
             bot_score = np.sum(np.power(np.array(top) - np.array(bot_roll), 2))
             
             if min_mid_score >= mid_score:
                 min_mid_score = mid_score
                 mid_aligned   = mid_roll
            
             if min_bot_score >= bot_score:
                 min_bot_score = bot_score
                 bot_aligned   = bot_roll
    
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
        

def trim_messy_borders(imgs):
    """ Trim the unequal black border from all sides of each image
    
    Args:
        Three images
        
    Returns:
        Three cropped images
    """
    
    top = trim_messy_border(imgs[0])
    mid = trim_messy_border(imgs[1])
    bot = trim_messy_border(imgs[2])
    
    return (top, mid, bot)
    
    
def trim_left_right(img):
    """ Trims black border off the left and right of a single image
    
    Increase contrast 
    Convert image to only black and white
    Blur Image to remove white border
    Samples the center of the image for border width
    Crop image 
    
    Assumes border is within 10% of image
    Assumes white specks within 5% of the image is ok
    
    Args:
        Single image to crop
    
    Returns:
        Cropped single image
    """    
    height      = len(img)
    width       = len(img[0])
    
    # area where white in border is ok
    white_ok_width = width * .05
    white_ok_height = height * .02
    
    max_crop_percent = .10
    max_crop_height = int(height * max_crop_percent)
    max_crop_width = int(width * max_crop_percent)
    
    # Increase contrast and turn image in black/white
    img_bw = np.around(img / 150)
    
    # plt.imshow(img_bw, plt.get_cmap('gray'), vmin = 0, vmax=1)
    # plt.show()
    
    crop_left = 0;
    crop_right = 0;
    for y in range(int(height/2)-10, int(height/2)+10):
        # Check left
        for x in range(0, max_crop_width):
            if img_bw[y][x] == 1 and x > white_ok_width:
                # Found a white pixel, reached edge of black border
                # Set a new crop width
                crop_left = max(x, crop_left)
                break
        
        # Check right
        for x in range(width-1, width-max_crop_width, -1):
            if img_bw[y][x] == 1 and x < width-white_ok_width:
                # Found a white pixel, reached edge of black border
                # Set a new crop width
                crop_right = max(width-x, crop_right)
                break
    
    print(max_crop_width)
    print(crop_left)
    print(crop_right)
    
    # Crop columns
    img_crop = np.delete(img, list(range(crop_left)), axis=1)
    #imshow(img)
    
    print("width from right:",width-crop_right, width)
    print(list(range(width-crop_right, width)))
    img_crop = np.delete(img_crop, list(range(width-crop_right*2, width)), axis=1) # UMMMM cropright *2???? whyy??? 
    imshow(img_crop)
    
    #img_blur = scipy.ndimage.filters.gaussian_filter(img_bw, sigma=7)
    
    
def imshow(img):    
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
    #img = plt.imread("prk2000000780.jpg") #argv[1]
    img = Image.open("images/prk2000000780.jpg")
    trimmed_img = trim_border(img)
    
    # Convert Pillow image to ndarray and transpose
    trimmed_img = np.asarray(trimmed_img, dtype=np.uint8)
    
    # Testing!!!
    trim_left_right(trimmed_img)
    
    # useful
    # cropped_img = crop_thirds(trimmed_img)    
    # aligned_img = single_scale_align(cropped_img, 15)    
    # final_img   = overlay_images(aligned_img)
    
    # concat = concat_n_images(cropped_img)
    # imshow(final_img)
    # input("Press ENTER to exit.")
    return 0    


if __name__ == "__main__":
    # Exit status is the return value from main
    # sys.exit(main())
    main()