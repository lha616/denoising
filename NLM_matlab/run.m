
close all;
clear all;
clc
I=double(imread('Cameraman256.png'));
%imshow(I/ 255)
noisy=I+15*randn(size(I));
tic
% 
O1 = NLm(I, 2, 5, 15)
psnr_noisy = psnr(noisy, I)
psnr_result = psnr(O1, I)
toc
imshow([I,noisy,O1],[]);
h(1) = I
h(2) = noisy
h(3) = O1

savefig(h, 'result.fig')
%imshow(O1 / 255)



