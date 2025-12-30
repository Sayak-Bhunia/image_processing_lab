img = imread('trial.jpg')
neg = 255 - img
psnr_val = psnr(img, neg)
disp(psnr_val)
