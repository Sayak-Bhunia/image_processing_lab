img = imread('trial.jpg')
neg = 255 - img
mkdir('output')
imwrite(neg, 'output/neg.jpg')
imshow(neg)
title('negative image')
