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
    Edit the image name directly at the beginning of the main file
    Large images need to use the multi-scale alignment 

"""

import sys
import copy
import scipy
import numpy as np
from math import *
from PIL import Image, ImageChops
from scipy import signal, ndimage
from skimage import data, filter
import matplotlib.pyplot as plt
from matplotlib import image as im

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
    height_crop = int(height/3)
    
    top = img[0:height_crop]                # blue
    mid = img[height_crop:height_crop*2]    # green
    bot = img[height_crop*2:height-1]       # red
    
    return (top, mid, bot)


def multi_scale_align(imgs):
    """ Uses image pyramids to align 3 channels of a large image
    
    Align the red and green channels to the blue channel, which will
    later be combined into a single image. The image pyramid is
    made of progressively smaller images. Useful for the alignment 
    search so we don't have to search over as large of an area.
    
    Args:
        img: 3 images. Blue, green, then red
    
    Returns: 
        Green and red images aligned to the blue image
    """
    orig      = copy.deepcopy(imgs)
    offsets   = multi_helper(imgs)
    aligned_g = np.roll(np.roll(orig[1], offsets[0][1], 0), offsets[0][0], 1)
    aligned_r = np.roll(np.roll(orig[2], offsets[1][1], 0), offsets[1][0], 1)
    
    return (orig[0], aligned_g, aligned_r)

def multi_helper(imgs):
    """ Uses image pyramids to align 3 channels of a large image
    
    Helper function to take care of the recursive calls. One the image
    is small enough, perform a normal single scale alignment. At the
    larger scale, check a 5x5 range around the new offset using single
    scale alignment to see if there is a better offset. Return the offset.
    
    Args:
        img: 3 images. Blue, green, then red
    
    Returns: 
        The offset for green then red images. Follows the format: (x, y),
        where x is axis 1 and y is axis 0.
    """
    height   = len(imgs[0])
    width    = len(imgs[0][0])
    r        = 3
    
    # Base Case
    if height < 150 or width < 150:
        print("got to smalled image size yay")
        
        range_xy = [[-r,r],[-r,r]]
        offset_g = single_scale_align_edge(imgs[0], imgs[1], range_xy)[1]
        offset_r = single_scale_align_edge(imgs[0], imgs[2], range_xy)[1]
        return (offset_g, offset_r)
    
    # Blur, shrink image and make a recursive call
    else:
        height = int(height/2)
        width  = int(width/2)
        
        print("in multi, height:", height, "width", width)
        
        copy_imgs = copy.deepcopy(imgs)
        
        for i in range(0,3):
            # imgs[i] = scipy.ndimage.filters.gaussian_filter(imgs[i], 1)
            PIL = Image.fromarray(np.uint8(copy_imgs[i])*255) # Convert arr to PIL img
            PIL = PIL.resize((width, height), Image.ANTIALIAS)     
            # PIL = PIL.resize((width, height))           
            copy_imgs[i] = np.asarray(PIL, dtype=np.uint8)   # Convert PIL img to arr 
        
        new_offset = multi_helper(copy_imgs)
        #new_offset = np.multiply(new_offset, 2)
        
        range_g = [[new_offset[0][0]-r, new_offset[0][0]+r],
                   [new_offset[0][1]-r, new_offset[0][1]+r]]
        range_r = [[new_offset[1][0]-r, new_offset[1][0]+r],
                   [new_offset[1][1]-r, new_offset[1][1]+r]]
        
        offset_g = single_scale_align_edge(imgs[0], imgs[1], range_g)[1]
        offset_r = single_scale_align_edge(imgs[0], imgs[2], range_r)[1]        
        return (offset_g, offset_r)
        

def single_scale_align(img0, img1, offset):
    """Single scale aligns two images
    
    Aligns img1 to img0, which acts like the anchor. 
    We exhaustively search over a window of possible displacements given
    by the offset (e.g. [-15,15] pixels), then score each offset image
    using some image matching metric (the sum of squared distances) and
    finally choose the displacement with the best score.
    
    Args: 
        img0: Anchor image to be aligned to
        img1: Image to be aligned
        offset: List of two offset ranges. First is for green, second
        is for red. Each offset range contains a x range first, then a
        y range
        
    Returns:
        A single aligned img and the alignment array, which contains
        alignemtn of green (index 0) and alignement of red (index 1)
    """
    min_score    = float("inf")
    aligned_img  = []
    final_offset = []
    
    imshow(img1)
    
    for x in range(offset[0][0], offset[0][1]):
        for y in range(offset[1][0], offset[1][1]):
             roll = np.roll(np.roll(img1, y, axis=0), x, axis=1)
             score = np.sum(np.power(np.array(img0)-np.array(roll), 2))
             
             if min_score >= score:
                 min_score    = score
                 aligned_img  = roll
                 final_offset = (x,y)
    
    print("final offset", final_offset)
    
    return (aligned_img, final_offset)
    

# def sobel_edge_filter(img):
#     dx = ndimage.sobel(img, 1)  # horizontal derivative
#     dy = ndimage.sobel(img, 0)  # vertical derivative
#     mag = np.hypot(dx, dy)      # magnitude
#     mag *= 255.0 / np.max(mag)  # normalize (Q&D)
#     return mag


def single_scale_align_edge(img0, img1, offset):
    """Single scale aligns two images using edge detection
    
    Similar to single_scale_align(), but also runs an edge detection
    filter on top of the image before the search starts.
    Aligns img1 to img0, which acts like the anchor. 
    We exhaustively search over a window of possible displacements given
    by the offset (e.g. [-15,15] pixels), then score each offset image
    using some image matching metric (the sum of squared distances) and
    finally choose the displacement with the best score.
    
    Args: 
        img0: Anchor image to be aligned to
        img1: Image to be aligned
        offset: List of two offset ranges. First is for green, second
        is for red. Each offset range contains a x range first, then a
        y range
        
    Returns:
        A single aligned img and the alignment array, which contains
        alignemtn of green (index 0) and alignement of red (index 1)
    """
    min_score  = float("inf")
    final_offset = []

    # Edge detection
    edge_sobel_0 = ndimage.sobel(img0, 1)  # horizontal derivative
    edge_sobel_1 = ndimage.sobel(img1, 1)  # horizontal derivative
    array0 = np.array(edge_sobel_0)
    
    # imshow(img1)
    imshow(concat_n_images([img0, edge_sobel_0, img1, edge_sobel_1]))
    
    
    for x in range(offset[0][0], offset[0][1]):
        for y in range(offset[1][0], offset[1][1]):
             roll  = np.roll(np.roll(edge_sobel_1, y, axis=0), x, axis=1)
             score = np.sum(np.power(array0 - np.array(roll), 2))
             
             if min_score >= score:
                 min_score    = score
                 final_offset = (x,y)
    
    aligned = np.roll(np.roll(img1, final_offset[1], 0), final_offset[0], 1)
    
    print("edge align offset", final_offset)
    
    return (aligned, final_offset)


def trim_border(im):
    """ Trims the solid color border from an image
    
    It gets the border colour from the top left pixel, using getpixel,
    so you don't need to pass the colour. Subtracts a scalar from the
    differenced image, this is a quick way of saturating all values
    under 100, 100, 100 (in my example) to zero. So is a neat way to
    remove any 'wobble' resulting from compression.
    
    https://stackoverflow.com/questions/10615901/trim-whitespace-using-pil
    
    Args:
        im: Single image to be trimmed
    
    Returns:
        An image if it was trimmed
    """
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    
    if bbox:
        return im.crop(bbox)

    
def trim_left_right(img):
    """ Trims black border off the left and right of a single image
    
    Trim off a non-uniform black border from the left and right edges
    of the image. The edge can be very noisy and won't necessarily be
    the same value of black. We increase contrast, convert image to
    only black and white pixels, count up the number of white pixels
    along the columns, find the point where we start having a lot of
    white pixels and set that as the cut off point. Finally, we crop
    the image.
    
    Assumes border is within 10% of image.
    Assumes white specks within 1.7% of the image are ok.
    
    Args:
        img: Single image to crop
    
    Returns:
        Cropped single image
    """    
    height         = len(img)
    width          = len(img[0])
    white_ok_width = int(width * .017)
    max_crop_width = int(width * .10)
    threshold      = height * .3
    
    img_bw  = np.around(np.multiply(img, 1/180))
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
    img_crop  = np.delete(img, list(range(crop_left)), axis=1)
    width     = len(img_crop[0])
    to_delete = list(range(width-crop_right, width)) 
    img_crop  = np.delete(img_crop, to_delete, axis=1)
    
    return img_crop


def trim_top_bot(imgs):
    """ Trims black border off the top and bot of three images
    
    Trim off a non-uniform black border from the top and bottom edges
    of the three images. The edge can be very noisy and won't
    necessarily have the same value of black. We increase contrast,
    convert image to only black and white pixels, count up the number
    of white pixels along the rows, find the point where we start having
    a lot of white pixels and set that as the cut off point. Then,
    we crop the image. At the very end, we need to re-crop each image
    so they all have the
    same height.
    
    Assumes border is within 10% of image.
    Assumes white specks within 1.7% of the image are ok.
    
    Args:
        imgs: 3 images to crop
    
    Returns:
        3 trimmed images
    """   
    height          = len(imgs[0])
    width           = len(imgs[0][0])
    white_ok_height = int(height * .02)
    max_crop_height = int(height * .10)
    threshold       = width * .3
    
    # Contrast and b/w
    img_bw_b = np.around(np.multiply(imgs[0], 1/180))
    img_bw_g = np.around(np.multiply(imgs[1], 1/180))
    img_bw_r = np.around(np.multiply(imgs[2], 1/180))
    
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
    """ Show an image with values in range 0 to 255
    
    Args:
        img: Single image to display
    """
    plt.imshow(img, plt.get_cmap('gray'), vmin = 0, vmax=255)
    plt.show()
    

def concat_images(imga, imgb):
    """ Combines two single channel image ndarrays side-by-side.
    
    Args:
        imga: First image to concatinate
        imgb: Second image to concatinate
        
    Returns:
        Concatinated imga and imgb
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
    """ Combines N color images from a list of images.
    
    Args:
        image_list: List of images to concatinate
    
    Returns:
        Concatinated image
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
        imgs: Tuple of three channels - blue, green, then red
    
    Return:
        Single full color image
    """
    height   = len(imgs[0])
    width    = len(imgs[0][0])
    rgbArray = np.zeros((height, width, 3), "uint8")
    rgbArray[..., 0] = imgs[2]   # red
    rgbArray[..., 1] = imgs[1]      # green
    rgbArray[..., 2] = imgs[0]   # blue
    return Image.fromarray(rgbArray)
    
 
def main(argv = sys.argv):
    PIL_img       = Image.open("finalImages/church.tif")
    trimmed_img   = trim_border(PIL_img) # remove white border
    ndarray_img   = np.asarray(trimmed_img, dtype=np.uint8)
    trimmed_img   = trim_left_right(ndarray_img)
    cropped_img   = crop_thirds(trimmed_img)
    retrim_img    = trim_top_bot(cropped_img)
    
    # Single Scale Align
    # range = [[-15,15],[-15,15]]
    # aligned_g = single_scale_align(retrim_img[0], retrim_img[1], range)[0]
    # aligned_r = single_scale_align(retrim_img[0], retrim_img[2], range)[0]
    # final_img = overlay_images((retrim_img[0], aligned_g, aligned_r))
    
    # Multi-Scale Align
    aligned_img = multi_scale_align(retrim_img)
    final_img   = overlay_images(aligned_img)
    
    #imshow(concat_n_images([retrim_img[0],aligned_g, aligned_r]))
    imshow(final_img)
    
    #scipy.misc.toimage(final_img, cmin=0.0, cmax=255).save('finalImages/00540u_color.tif')
    return 0    


if __name__ == "__main__":
    # Exit status is the return value from main
    # sys.exit(main())
    main()