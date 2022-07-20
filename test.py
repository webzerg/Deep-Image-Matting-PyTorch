import math

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from config import device, fg_path_test, a_path_test, bg_path_test
from data_gen import data_transforms, fg_test_files, bg_test_files
from utils import compute_mse, compute_sad, AverageMeter, get_logger


def gen_test_names():
    num_fgs = 20
    num_bgs = 1000
    num_bgs_per_fg =50

    names = []
    bcount = 0
    for fcount in range(num_fgs):
        for i in range(num_bgs_per_fg):
            names.append(str(fcount) + '_' + str(bcount) + '.png')
            bcount += 1

    return names


def process_test(im_name, bg_name):
    # print(bg_path_test + bg_name)
    im = cv.imread(fg_path_test + im_name)
    a = cv.imread(a_path_test + im_name, 0)
    h, w = im.shape[:2]
    bg = cv.imread(bg_path_test + bg_name)
    bh, bw = bg.shape[:2]
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)

    return composite4_test(im, bg, a, w, h)


# def composite4_test(fg, bg, a, w, h):
#     fg = np.array(fg, np.float32)
#     bg_h, bg_w = bg.shape[:2]
#     x = max(0, int((bg_w - w)/2))
#     y = max(0, int((bg_h - h)/2))
#     bg = np.array(bg[y:y + h, x:x + w], np.float32)
#     alpha = np.zeros((h, w, 1), np.float32)
#     alpha[:, :, 0] = a / 255.
#     im = alpha * fg + (1 - alpha) * bg
#     im = im.astype(np.uint8)
#     print('im.shape: ' + str(im.shape))
#     print('a.shape: ' + str(a.shape))
#     print('fg.shape: ' + str(fg.shape))
#     print('bg.shape: ' + str(bg.shape))
#     return im, a, fg, bg


def composite4_test(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg_h, bg_w = bg.shape[:2]
    x = max(0, int((bg_w - w) / 2))
    y = max(0, int((bg_h - h) / 2))
    crop = np.array(bg[y:y + h, x:x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    # trimaps = np.zeros((h, w, 1), np.float32)
    # trimaps[:,:,0]=trimap/255.

    im = alpha * fg + (1 - alpha) * crop
    im = im.astype(np.uint8)

    new_a = np.zeros((bg_h, bg_w), np.uint8)
    new_a[y:y + h, x:x + w] = a
    # new_trimap = np.zeros((bg_h, bg_w), np.uint8)
    # new_trimap[y:y + h, x:x + w] = trimap
    # cv.imwrite('images/test/new/' + trimap_name, new_trimap)
    new_im = bg.copy()
    new_im[y:y + h, x:x + w] = im
    # cv.imwrite('images/test/new_im/'+trimap_name,new_im)
    return new_im, new_a, fg, bg


if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model'].module
    model = model.to(device)
    model.eval()

    transformer = data_transforms['valid']

    names = gen_test_names()

    mse_losses = AverageMeter()
    sad_losses = AverageMeter()

    logger = get_logger()
    i = 0
    for name in tqdm(names):
        fcount = int(name.split('.')[0].split('_')[0])
        bcount = int(name.split('.')[0].split('_')[1])
        im_name = fg_test_files[fcount]
        # print(im_name)
        bg_name = bg_test_files[bcount]
        trimap_name = im_name.split('.')[0] + '_' + str(i) + '.png'
        # print('trimap_name: ' + str(trimap_name))

        # trimap = cv.imread('data/Combined_Dataset/Test_set/Adobe-licensed images/trimaps/' + trimap_name, 0)
        # print('trimap: ' + str(trimap))

        i += 1
        if i == 20:
            i = 0

        img, alpha, fg, bg = process_test(im_name, bg_name)
        h, w = img.shape[:2]
        # mytrimap = gen_trimap(alpha)
        # cv.imwrite('images/test/new_im/'+trimap_name,mytrimap)

        x = torch.zeros((1, 3, h, w), dtype=torch.float)
        img = img[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)  # [3, 320, 320]
        img = transformer(img)  # [3, 320, 320]
        x[0:, 0:3, :, :] = img
        # x[0:, 3, :, :] = torch.from_numpy(new_trimap.copy() / 255.)

        # Move to GPU, if available
        x = x.type(torch.FloatTensor).to(device)  # [1, 4, 320, 320]
        """
        get the ground truth for the trimap of test data
        pred_trimap_argmax
        0: background
        1: unknown
        2: foreground
        """
        gt_trimap = np.zeros(alpha.shape)
        gt_trimap.fill(1)
        gt_trimap[alpha <= 0] = 0
        gt_trimap[alpha >= 255] = 2

        alpha = alpha / 255.

        with torch.no_grad():
            # pred = model(x)  # [1, 4, 320, 320]
            trimap_out = model(x)

        pred = trimap_out.argmax(dim=1)
        pred = pred.cpu().numpy()
        pred = pred.reshape((h, w))  # [320, 320]

        # pred[new_trimap == 0] = 0.0
        # pred[new_trimap == 255] = 1.0
        # cv.imwrite('images/test/out/' + trimap_name, pred * 255)

        # Calculate loss
        # loss = criterion(alpha_out, alpha_label)
        mse_loss = compute_mse(pred, gt_trimap)
        sad_loss = compute_sad(pred, gt_trimap)

        # Keep track of metrics
        mse_losses.update(mse_loss.item())
        sad_losses.update(sad_loss.item())
        print("sad:{} mse:{}".format(sad_loss.item(), mse_loss.item()))
        print("Avg sad:{} Avg mse:{}".format(sad_losses.avg, mse_losses.avg))
    print("Avg sad:{} mse:{}".format(sad_losses.avg, mse_losses.avg))
