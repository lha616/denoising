clear all;
close all;


path = 'imgs';
sigma = 20;
 images_ = dir(fullfile(path, '*.png'));
 for i = 1 : length(images_)
     image_name = [path '/' images_(i).name];
% filename = 
    im = double(imread(image_name));
%       imwrite(im, 'result\Cameraman256 PSNR_30.57.png');
%       pause;
    figure(),imshow(im, []),title('src');
    
%     pause

    

    im_noisy = im + sigma * randn(size(im));

    % imshow(im_noisy);

    denoised_image = BAS( im_noisy, sigma );

    figure(),imshow(denoised_image, []),title('denoising');
    
    PSNR = psnr(denoised_image, im);
    
    s = sprintf('%s PSNR: %.2f', image_name(6:end - 4), PSNR);
    
    disp(s);
    
    save_s = sprintf('result\\%s PSNR_%.2f.png', image_name(6:end - 4), PSNR);
    disp(save_s);
    imwrite(denoised_image, save_s);
    
    
%     pause;
 end

