from utils import add_gaussian_noise, symetrize
from bm3d_1st_step import bm3d_1st_step
from bm3d_2nd_step import bm3d_2nd_step
from psnr import compute_psnr


def run_bm3d(noisy_im, sigma,
             n_H, k_H, N_H, p_H, tauMatch_H, useSD_H, tau_2D_H, lambda3D_H,
             n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W):
    k_H = 8 if (tau_2D_H == 'BIOR' or sigma < 40.) else 12
    k_W = 8 if (tau_2D_W == 'BIOR' or sigma < 40.) else 12

    noisy_im_p = symetrize(noisy_im, n_H)
    img_basic = bm3d_1st_step(sigma, noisy_im_p, n_H, k_H, N_H, p_H, lambda3D_H, tauMatch_H, useSD_H, tau_2D_H)
    img_basic = img_basic[n_H: -n_H, n_H: -n_H]

    assert not np.any(np.isnan(img_basic))
    img_basic_p = symetrize(img_basic, n_W)
    noisy_im_p = symetrize(noisy_im, n_W)
    img_denoised = bm3d_2nd_step(sigma, noisy_im_p, img_basic_p, n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W)
    img_denoised = img_denoised[n_W: -n_W, n_W: -n_W]

    return img_basic, img_denoised


if __name__ == '__main__':
    import os
    import cv2
    import numpy as np

    # <hyper parameter> -------------------------------------------------------------------------------
    n_H = 16
    k_H = 8
    N_H = 16
    p_H = 3
    lambda3D_H = 2.7  # ! Threshold for Hard Thresholding 硬阈值
    useSD_H = False
    tau_2D_H = 'BIOR'

    n_W = 16
    k_W = 8
    N_W = 16
    p_W = 3
    useSD_W = True
    tau_2D_W = 'DCT'  # 使用DCT变换
    # <\ hyper parameter> -----------------------------------------------------------------------------


    # e1 = cv2.getTickCount()
    sigma = 15

    tauMatch_H = 2500 if sigma < 35 else 5000  # ! 1Step patches 之间的相似度阈值
    tauMatch_W = 400 if sigma < 35 else 3500  # ! 2Step patches 之间的相似度阈值

    data_dir = 'data'
    save_dir = 'result/'
    for img in os.listdir(data_dir):
        imgDir = data_dir + '/' + img
        im = cv2.imread(imgDir, cv2.IMREAD_GRAYSCALE)  # 灰度图形式读取原图像
        # im = cv2.resize(im, (400, 400))
        noisy_im = im + np.random.randn(im.shape[0], im.shape[1]) * sigma  # 添加高斯噪声

        # cv2.imshow('source', im)
        # cv2.imshow('noise', noisy_im / 255)

        print('Src2Noisy_PSNR %f' % (compute_psnr(im, noisy_im)))
        # print(img)
        # 计算BM3D初步估计图像&最终估计图像
        im1, im2 = run_bm3d(noisy_im, sigma,
                            n_H, k_H, N_H, p_H, tauMatch_H, useSD_H, tau_2D_H, lambda3D_H,
                            n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W)

        psnr_1st = compute_psnr(im, im1)
        psnr_2nd = compute_psnr(im, im2)
        im1 = (np.clip(im1, 0, 255)).astype(np.uint8)
        im2 = (np.clip(im2, 0, 255)).astype(np.uint8)

        saveName = save_dir + img[:-4] + '_noisy_s' + str(sigma) + '_p%.4f' % compute_psnr(noisy_im, im) + '.png'
        cv2.imwrite(saveName, noisy_im)
        saveName = save_dir + img[:-4] + '_Basic_s' + str(sigma) + '_p%.4f' % psnr_1st + '.png'
        cv2.imwrite(saveName, im1)

        saveName = save_dir + img[:-4] + '_Final_s' + str(sigma) + '_p%.4f' % psnr_2nd + '.png'
        cv2.imwrite(saveName, im2)

        # print(img[:-4] + '_1st_BasicPSNR %f' % psnr_1st)
        # print(img[:-4] + '2nd_FinalPSNR %f' % psnr_2nd)

    # e2 = cv2.getTickCount()
    # time = (e2 - e1) / cv2.getTickFrequency()  # 计算函数执行时间

    # print("The Processing time of MyBM3D is %f s" % time)

    # cv2.waitKey(0)