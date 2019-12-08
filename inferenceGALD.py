import os
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from libs.models.GALDNet import GALD_res101, GALD_res50


color_list = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
color_map = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
             (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
             (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]


def transform(img):
    img = cv2.imread(img)
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    img = img - IMG_MEAN
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).cuda()
    return img


def inference(args):
    net = GALD_res101(19).cuda()
    net.eval()
    saved_state_dict = torch.load(args.resume)
    net.load_state_dict(saved_state_dict)
    img_list = makelist(args.input_dir)
    h, w = 1024, 2048

    for i ,name in enumerate(img_list):
        with torch.no_grad():
            img = transform(name)
            out = net(img)
            out_res = out[0]
            out = F.upsample(out_res, size=(h, w), mode='bilinear', align_corners=True)
            result = out.argmax(dim=1)[0]
            result = result.data.cpu().squeeze().numpy()
            row, col = result.shape
            dst = np.zeros((row, col, 3), dtype=np.uint8)
            for i in range(19):
                dst[result == i] = color_map[i]
            print(name, " done!")
            save_name = os.path.join(args.output_dir, "/".join(name.split('/')[4:]))
            save_dir = "/".join(save_name.split("/")[:-1])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(save_name, dst[:,:,::-1])


def makelist(dir):
    out = []
    l = os.listdir(dir)
    for i in l:
        out.append(os.path.join(dir, i))
    out.sort()
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch \
                    Segmentation Crop Prediction')
    parser.add_argument('--input_dir', type=str,
                        default="/home/lxt/imgs",
                        help='training dataset folder (default: \
                                  $(HOME)/data)')
    parser.add_argument('--output_dir', type=str, default="/home/lxt/Documents/outputs/",
                        help='output directory of the model, for saving the seg_models')
    parser.add_argument("--resume", type=str,
                        default="/home/lxt/pretrained/GALD_res101_map_831.pth")

    args = parser.parse_args()
    inference(args)