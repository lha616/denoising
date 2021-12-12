import cv2
import numpy as np
import math
import _thread


# 计算PSNR
def psnr(A, B):
    val = 255
    mse = ((A.astype(np.float)-B.astype(np.float))**2).mean()
    return 10*np.log10((val*val)/mse)


# 计算一个高斯核
def make_kernel(f):
    kernel = np.zeros((2 * f + 1, 2 * f + 1))
    for d in range(0, f + 1):
        kernel[d: 2 * f  + 1 - d, d: 2 * f + 1 - d] += (d + 1)
    # print(kernel)
    kernel = 1.0 / kernel
    # print(kernel / kernel.sum())
    return kernel / kernel.sum()

# 获取一个kaiser窗kHW * kHW
def get_kaiserWindow(kHW):
    k = np.kaiser(kHW, 1)
    # print(k)
    k_2d = k[:, np.newaxis] @ k[np.newaxis, :]
    # ret = 1.0 / k_2d
    return k_2d / k_2d.sum()

# # 生成一个高斯核
# def make_kernel(f):
#     kernel = np.zeros((2 * f + 1, 2 * f + 1))
#     for i in range(0, 2 * f):
#         for j in range(0, 2 * f):
#             kernel[i, j] = 1.0 / ((i - f) ** 2 + (j - f) ** 2)
#         # kernel[f - d: f + d + 1, f - d: f + d + 1] += (1.0 / ((2 * d + 1) ** 2))
#     print(kernel)
#     print(kernel / kernel.sum())
#     return kernel / kernel.sum()


# 对称填充, 填充图片防止边界没有扫描到
def symetrize(img, N):
    img_pad = np.pad(img, ((N, N), (N, N)), 'symmetric')
    return img_pad

def progress(percent, width=50):
    '''
    进度打印
    '''
    if percent >= 100:
        percent = 100
    show_str = ('[%%-%ds]' % width) % (int(width * percent / 100) * "#")  # 字符串拼接的嵌套使用
    print('\r%s %d%%' % (show_str, percent), end='')


# NLM简单实现
def NLMeans(img, sigma, search_range=10, h = 2.2):
    h = - h * (sigma ** 2)
    target = img.copy()
    r = 1
    img = symetrize(img, r)
    size = img.shape  # 图片的 h w c

    kernel = get_kaiserWindow(2 * r + 1)
    # print(kernel)

    for y in range(r, size[0] - r):
        # print(y)
        progress(int((y + r) * 100 / size[0]), 30)
        for x in range(r, size[1] - r):

            srcblock = img[y - r:y + r + 1, x - r:x + r + 1]  # 是y,x

            # 搜索范围
            y_start = max(y - search_range, r)
            x_start = max(x - search_range, r)
            y_end = min(y + search_range, size[0] - r - 1)
            x_end = min(x + search_range, size[1] - r - 1)

            w = np.zeros([y_end - y_start + 1, x_end - x_start + 1])

            for yi in range(y_start, y_end + 1):
                for xi in range(x_start, x_end + 1):
                    refblock = img[yi - r:yi + r + 1, xi - r:xi + r + 1]

                    delta = np.sum((np.square(srcblock - refblock) * kernel))

                    w[yi - y_start, xi - x_start] = math.exp(delta / h)  # 计算相似度权重

            # 非局部均值
            target[y - r, x - r] = np.sum(w * img[y_start:y_end + 1, x_start:x_end + 1]) / np.sum(w)
    # print()
    return target

# 给图像添加高斯噪声
def add_gauss_noise(img, sigma):
    noise = np.random.randn(*img.shape) * sigma  # 生成与img相同形状的高斯噪声矩阵
    return img + noise


# 去噪test
def test(filename='Cameraman256', dir = 'data/', result_dir='result_Compare/', h=2.2):

    gray = cv2.imread((dir + filename), 0)  # 以灰度图形式读取
    gray = gray.astype('float32') # 转化成浮点数易于计算
    sigma = 15  # 噪声水平
    noised = add_gauss_noise(gray, sigma)  # 对图像加上高斯噪声

    # diff = noised - gray
    dest = NLMeans(noised, sigma, h=h)
    FstNlM = cv2.fastNlMeansDenoising(np.clip(np.round(noised*1.0), 0, 255).astype(np.uint8), None, sigma, 5, 15)  # 利用opencv自带的快速NLM去噪

    # cv2.imshow('FstNLM', FstNlM / 255.)
    # cv2.imshow('source', gray / 255.)
    # cv2.imshow('noise', noised / 255.)
    # cv2.imshow('MyNLM', dest / 255.)
    # cv2.imshow('noisy', diff)

    # cv2.imwrite('noisy.png', diff)
    # write = result_dir + filename

    save_name = result_dir + filename[:-4] + '_s' + str(sigma) + '_p%.4f' % (psnr(gray, noised)) + '_Noisy.png'
    cv2.imwrite(save_name, noised)
    save_name = result_dir + filename[:-4] + '_s' + str(sigma) + '_p%.4f' % (psnr(gray, dest)) + '_MyNLM.png'
    cv2.imwrite(save_name, dest)
    save_name = result_dir + filename[:-4] + '_s' + str(sigma) + '_p%.4f' % (psnr(gray, FstNlM)) + '_FstNLM.png'
    cv2.imwrite(save_name, FstNlM)

    # cv2.imwrite((write + '_noisy_p%.2f') % (psnr(noised, gray)), noised)
    # cv2.imwrite((write + '_MyNLM_p%.2f') % (psnr(dest, gray)), dest)
    # cv2.imwrite((write + '_FastNLM_p%.2f') % (psnr(FstNlM, gray)), FstNlM)

    # print('AddNoise_PSNR: %f' % (psnr(noised, gray)))
    # print('MyNLM_PSNR: %f' % (psnr(dest, gray)))
    # print('FstNLM_PSNR: %f' % (psnr(FstNlM, gray)))
    # print("The Processing time of MyNLMeans is %f s" % time)
    # cv2.waitKey(0)


if __name__ == '__main__':
    import os

    img_dir = 'data'
    for img in os.listdir(img_dir):
        # print(img)
        test(filename=img)
        print(' ' + img + ' over')
    # test(filename='Cameraman256.png')