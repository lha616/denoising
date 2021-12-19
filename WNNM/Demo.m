  

    nSig  = 15;
    path = 'data' ;% 图像路径
    images_ = dir(fullfile(path, '*.png'));
    for i = 1 : length(images_)
        image_name = [path '/' images_(i).name];
        O_Img = double(imread(image_name));

        randn('seed', 0);
        % 添加噪声
        N_Img = O_Img + nSig* randn(size(O_Img));                                   %Generate noisy image
        PSNR  =  psnr( N_Img, O_Img);
%         fprintf( '%s_Noisy Image: nSig = %2.3f, PSNR = %2.2f \n\n\n',, nSig, PSNR );
        
        s = sprintf('%s_noisyImg_P%.2f',image_name(6:end - 4), PSNR);
        disp(s);
        imwrite(N_Img/255, ['result/' s '.png']);

        Par   = ParSet(nSig);   
        E_Img = WNNM_DeNoising( N_Img, O_Img, Par );                                %WNNM denoisng function
        PSNR = psnr(O_Img, E_Img);

        fprintf( 'Estimated Image: nSig = %2.3f, PSNR = %2.2f \n\n\n', nSig, PSNR );
        imshow([O_Img, N_Img, E_Img], []);
        s = sprintf('%s_DenoisingImg_P%.2f',image_name(6:end - 4), PSNR);
        disp(s)
        imwrite(E_Img / 255,['result/' s '.png']);
    end
     