
clear all
close all


    imgName = 'house.png';
%     imgName = 'Cameraman256.png'
    original_image = double(imread('house.png')); % 读入原图片

    sigma = 15;   % AWGN 的噪声水平
   
    randn('seed', 0);   % 创建随机种子
    
    noisy_image = original_image + sigma * randn(size(original_image)); %  给原图添加噪声
    
    original_image = original_image / 255; % 将图片和噪声图片的像素值归一化
    
    noisy_image = noisy_image / 255;
    
    imwrite(original_image, 'srcImg.png');
    imwrite(noisy_image, 'noisy_img.png');
%     figure,imshow(original_image); title('original neat image');
%     figure,imshow(noisy_image,[]); title('noisy image');
    PSNR_noisy = psnr(noisy_image * 255, original_image * 255); % 计算噪声图像的psnr
    fprintf( 'noisy_Image: Sigma = %2.1f, PSNR = %2.2f \n', sigma, PSNR_noisy );

 
tic,

[basic_estimation] = first_step(noisy_image,sigma); %% BM3D去噪初步估计，硬阈值处理
% pause
% 
[denoised_image] = final_step(noisy_image,basic_estimation,sigma); %% BM3D 最终估计, 使用维纳滤波.


figure,imshow(original_image); title('original neat image');
figure,imshow(noisy_image); title('noisy image');
figure,imshow(basic_estimation); title('basic estimation image');
imwrite(basic_estimation, 'basic_Img.png');
figure,imshow(denoised_image);title('denoised image');
imwrite(denoised_image, 'final_Img.png');
PSNR_basic = psnr(original_image * 255, basic_estimation * 255);

fprintf( 'basic_estimation_Image: PSNR = %2.2f \n', PSNR_basic );
PSNR_final = psnr(original_image * 255,denoised_image * 255);  %% 最终估计的 PSNR
SSIM = ssim(original_image,denoised_image);
fprintf( 'final_estimation_Image: PSNR = %2.2f \n', PSNR_final );
fprintf( 'final_estimation_Image: SSIM = %2.2f \n', SSIM );

s = sprintf('sigma: %d, PSNR:%.2f, SSIM: %.2f',sigma,PSNR_final, SSIM);
figure,imshow([original_image, noisy_image, denoised_image], []);title('s');

toc,