#2 read create neg and save it

img = imread('trial.jpg')
neg = 255 - img
mkdir('output')
imwrite(neg, 'output/neg.jpg')
imshow(neg)
title('negative image')

#1 psnr val

img = imread('trial.jpg')
neg = 255 - img
psnr_val = psnr(img, neg)
disp(psnr_val)

#3 read use histogram equalization to enhance image and save it

img = imread('trial.jpg');
gray = rgb2gray(img);
enhanced = histeq(gray);
imwrite(enhanced, 'output/hist_eq.jpg');
imshowpair(gray, enhanced, 'montage')

#4 read 2 images create 2 images by image addition and image substraction and save all those images

img1 = imread('trial2.jpg');
img2 = imread('trial2.jpg');
add_res = imadd(img1, img2);
sub_res = imsubtract(img1, img2);
imwrite(add_res, 'output/add.jpg');
imwrite(sub_res, 'output/sub.jpg');
imshowpair(add_res, sub_res, 'montage');

#5 read and resize by reducing 50% and increasing 75% and save

img = imread('trial.jpg');
dcr = imresize(img, 0.5);
incr = imresize(img, 1.75);
imwrite(dcr, 'output/reduced_img.jpg');
imwrite(incr, 'output/increased_img.jpg')
imshowpair(dcr, incr, 'montage');

#6 read an image and rotate it by 45 degrees

img = imread('trial.jpg');
rot = imrotate(img, 45);
imwrite(rot, 'output/rotated_img.jpg');
imshow(rot);

#7 read and corp it in diff sizes and save

img = imread('trial.jpg');
a = imcrop(img, [50, 50, 200, 200]);
b = imcrop(img, [100, 100, 300, 300]);
imwrite(a, 'output/corp1.jpg');
imwrite(b, 'output/crop2.jpg');
imshowpair(a, b, 'montage');

#8 apply 2% salt and pepper noise to an image and save it then apply median filter to remove noise

img = imread('trial.jpg');
gray = rgb2gray(img);
noisy = imnoise(gray, 'salt & pepper', 0.02);
imwrite(noisy, 'output/noisy.jpg');
filtered = medfilt2(noisy);
imwrite(filtered, 'output/filtered.jpg');
imshowpair(noisy, filtered, 'montage');

