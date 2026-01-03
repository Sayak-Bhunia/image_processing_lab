#2 read create neg and save it

img = imread('trial.jpg')
neg = 255 - img
mkdir('output')
imwrite(neg, 'output/neg.jpg')
imshow(neg)
title('negative image')

import cv2
import os
# Create folder if not exists
if not os.path.exists('output_folder'):
 os.makedirs('output_folder')
img = cv2.imread('input.jpg')
# Invert the image
negative_img = 255 - img
# Alternatively: negative_img = cv2.bitwise_not(img)
cv2.imwrite('output_folder/negative_image.jpg', negative_img)

#1 psnr val

img = imread('trial.jpg')
neg = 255 - img
psnr_val = psnr(img, neg)
disp(psnr_val)

import cv2
import numpy as np
import math
def calculate_psnr(img1, img2):
 mse = np.mean((img1 - img2) ** 2)
 if mse == 0:
 return 100 # No noise, images are identical
 pixel_max = 255.0
 psnr = 20 * math.log10(pixel_max / math.sqrt(mse))
 return psnr
# Example usage
original = cv2.imread('image1.jpg')
compressed = cv2.imread('image2.jpg') # Must be same size
print(f"PSNR value: {calculate_psnr(original, compressed)} dB")


#3 read use histogram equalization to enhance image and save it

img = imread('trial.jpg');
gray = rgb2gray(img);
enhanced = histeq(gray);
imwrite(enhanced, 'output/hist_eq.jpg');
imshowpair(gray, enhanced, 'montage')

import cv2 
img = cv2.imread('input.jpg', 0) # Read as grayscale 
# Apply Histogram Equalization 
equ_img = cv2.equalizeHist(img) 
cv2.imwrite('output_folder/equalized_image.jpg', equ_img

#4 read 2 images create 2 images by image addition and image substraction and save all those images

img1 = imread('trial2.jpg');
img2 = imread('trial2.jpg');
add_res = imadd(img1, img2);
sub_res = imsubtract(img1, img2);
imwrite(add_res, 'output/add.jpg');
imwrite(sub_res, 'output/sub.jpg');
imshowpair(add_res, sub_res, 'montage');

import cv2 
img1 = cv2.imread('input1.jpg') 
img2 = cv2.imread('input2.jpg') 
# Resize img2 to match img1 if necessary 
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0])) 
# Addition 
added_img = cv2.add(img1, img2) 
# Subtraction 
subtracted_img = cv2.subtract(img1, img2) 
cv2.imwrite('output_folder/added_image.jpg', added_img) 
cv2.imwrite('output_folder/subtracted_image.jpg', subtracted_img) 


#5 read and resize by reducing 50% and increasing 75% and save

img = imread('trial.jpg');
dcr = imresize(img, 0.5);
incr = imresize(img, 1.75);
imwrite(dcr, 'output/reduced_img.jpg');
imwrite(incr, 'output/increased_img.jpg')
imshowpair(dcr, incr, 'montage');

import cv2 
img = cv2.imread('input.jpg') 
# Reduce by 50% (0.5 scale) 
half_img = cv2.resize(img, None, fx=0.5, fy=0.5) 
# Increase by 75% (1.75 scale) 
# Note: Prompt says "increasing 75%", implies original + 75% = 175% 
large_img = cv2.resize(img, None, fx=1.75, fy=1.75) 
cv2.imwrite('output_folder/resized_50_percent.jpg', half_img) 
cv2.imwrite('output_folder/resized_175_percent.jpg', large_img)


#6 read an image and rotate it by 45 degrees

img = imread('trial.jpg');
rot = imrotate(img, 45);
imwrite(rot, 'output/rotated_img.jpg');
imshow(rot);

import cv2 
img = cv2.imread('input.jpg') 
(h, w) = img.shape[:2] 
center = (w // 2, h // 2) 
# Calculate rotation matrix 
# 45 is the angle, 1.0 is the scale 
M = cv2.getRotationMatrix2D(center, 45, 1.0) 
# Perform the rotation 
rotated_img = cv2.warpAffine(img, M, (w, h)) 
cv2.imwrite('output_folder/rotated_45.jpg', rotated_img)

#7 read and corp it in diff sizes and save

img = imread('trial.jpg');
a = imcrop(img, [50, 50, 200, 200]);
b = imcrop(img, [100, 100, 300, 300]);
imwrite(a, 'output/corp1.jpg');
imwrite(b, 'output/crop2.jpg');
imshowpair(a, b, 'montage');

import cv2 
img = cv2.imread('input.jpg') 
# Crop a region (example: from y=50 to 200, x=100 to 300)
crop1 = img[50:200, 100:300] 
crop2 = img[0:100, 0:100] 
cv2.imwrite('output_folder/crop_size1.jpg', crop1) 
cv2.imwrite('output_folder/crop_size2.jpg', crop2)


#8 apply 2% salt and pepper noise to an image and save it then apply median filter to remove noise

img = imread('trial.jpg');
gray = rgb2gray(img);
noisy = imnoise(gray, 'salt & pepper', 0.02);
imwrite(noisy, 'output/noisy.jpg');
filtered = medfilt2(noisy);
imwrite(filtered, 'output/filtered.jpg');
imshowpair(noisy, filtered, 'montage');

import cv2 
import numpy as np 
import random 
def add_salt_and_pepper_noise(image, percentage): 
	row, col, ch = image.shape 
	noisy = np.copy(image) 
	# Number of pixels to alter 
	num_pixels = int(row * col * percentage) 
	for _ in range(num_pixels): 
		# Pick a random pixel 
		y = random.randint(0, row - 1) 
		x = random.randint(0, col - 1)
		# Randomly choose Salt (255) or Pepper (0) 
		if random.random() < 0.5: 
			noisy[y, x] = [255, 255, 255] # White 
		else: noisy[y, x] = [0, 0, 0] # Black 
	return noisy 
img = cv2.imread('input.jpg') 
# 1. Add 2% noise (0.02) 
noisy_img = add_salt_and_pepper_noise(img, 0.02) 
cv2.imwrite('output_folder/noisy_image.jpg', noisy_img) 
# 2. Remove noise using Median Filter 
# kernel size must be odd (e.g., 3 or 5) 
denoised_img = cv2.medianBlur(noisy_img, 3) 
cv2.imwrite('output_folder/denoised_image.jpg', denoised_img)

