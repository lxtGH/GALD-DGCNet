# Author: Xiangtai Li
# Email: lxtpku@pku.edu.cn
"""
    Distribute Training Code For Fast training.
"""

import argparse
import os
import os.path as osp
import timeit
import numpy as np


import torch
from torch.utils import data
import torch.optim as optim
import torch.backends.cudnn as cudnn

from libs.utils.logger import Logger as Log
from libs.utils.tools import adjust_learning_rate, all_reduce_tensor
from libs.datasets.cityscapes import Cityscapes

from libs.core.loss import CriterionOhemDSN, CriterionDSN


try:
    import apex
    from apex import amp
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex.")


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    """
    Parse all the arguments
    Returns: args
    A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch_size_per_gpu", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of images sent to the network in one step.")
    parser.add_argument('--gpu_num',type=int, default=8)
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data_list", type=str, default="./data/cityscapes/train.txt",
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--data_set", type=str, default="cityscapes", help="dataset to train")
    parser.add_argument("--arch", type=str, default="CascadeRelatioNet_res50", help="network architecture")
    parser.add_argument("--ignore_label", type=int, default=255,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input_size", type=int, default=832 ,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning_rate", type=float, default=1e-2,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num_classes", type=int, default=19,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num_steps", type=int, default=50000,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--random_mirror", action="store_true", default=True,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random_scale", action="store_true", default=True,
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random_seed", type=int, default=1234,
                        help="Random seed to have reproducible results.")

    # ***** Params for save and load ******
    parser.add_argument("--restore_from", type=str, default="./pretrained",
                        help="Where restore models parameters from.")
    parser.add_argument("--save_pred_every", type=int, default=5000,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Where to save snapshots of the models.")
    parser.add_argument("--save_start",type=int, default=40000)
    parser.add_argument("--gpu", type=str, default=None,
                        help="choose gpu device.")
    parser.add_argument("--ft", type=bool, default=False,
                        help="fine-tune the models with large input size.")
    # **** Params for OHEM **** #
    parser.add_argument("--ohem", type=str2bool, default='False',
                        help="use hard negative mining")
    parser.add_argument("--ohem_thres", type=float, default=0.7,
                        help="choose the samples with correct probability underthe threshold.")
    parser.add_argument("--ohem_keep", type=int, default=100000,
                        help="choose the samples with correct probability underthe threshold.")
    # ***** Params for logging ***** #
    parser.add_argument('--log_level', default="info", type=str,
                        dest='log_level', help='To set the log level to files.')
    parser.add_argument('--log_file', default="./log/train.log", type=str,
                        dest='log_file', help='The path of log files.')
    parser.add_argument("--log_format", default="%(asctime)s %(levelname)-7s %(message)s", type=str,
                        dest="log_format", help="format of log files"
                        )
    parser.add_argument('--stdout_level', default="info", type=str,
                        dest='stdout_level', help='To set the level to print to screen.')
    parser.add_argument("--rewrite", default=False, type=bool,
                        dest="rewrite", help="whether write the file when using log"
                        )
    parser.add_argument("--rgb", type=str2bool, default='False')
    # ***** Params for Distributed Traning ***** #
    parser.add_argument('--apex', action='store_true', default=False,
                        help='Use Nvidia Apex Distributed Data Parallel')
    parser.add_argument("--local_rank", default=0, type=int, help="parameter used by apex library")
    args = parser.parse_args()
    return args


start = timeit.default_timer()

args = get_arguments()


def main():

    # make save dir
    if args.local_rank == 0:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
    # launch the logger
    Log.init(
        log_level=args.log_level,
        log_file=osp.join(args.save_dir, args.log_file),
        log_format=args.log_format,
        rewrite=args.rewrite,
        stdout_level=args.stdout_level
    )
    # RGB or BGR input(RGB input for ImageNet pretrained models while BGR input for caffe pretrained models)
    if args.rgb:
        IMG_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
        IMG_VARS = np.array((0.229, 0.224, 0.225), dtype=np.float32)
    else:
        IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        IMG_VARS = np.array((1, 1, 1), dtype=np.float32)

    # set models
    import libs.models as models
    deeplab = models.__dict__[args.arch](num_classes=args.num_classes, data_set=args.data_set)
    if args.restore_from is not None:
        saved_state_dict = torch.load(args.restore_from,map_location=torch.device('cpu'))
        new_params = deeplab.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not i_parts[0] == 'fc':
                new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
        Log.info("load pretrined models")
        if deeplab.backbone is not None:
            deeplab.backbone.load_state_dict(new_params, strict=False)
        else:
            deeplab.load_state_dict(new_params, strict=False)
    else:
        Log.info("train from stracth")

    args.world_size = 1

    if 'WORLD_SIZE' in os.environ and args.apex:
        args.apex = int(os.environ['WORLD_SIZE']) > 1
        args.world_size = int(os.environ['WORLD_SIZE'])
        print("Total world size: ", int(os.environ['WORLD_SIZE']))

    if not args.gpu == None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    h, w = args.input_size, args.input_size
    input_size = (h, w)


     # Set the device according to local_rank.
    torch.cuda.set_device(args.local_rank)
    Log.info("Local Rank: {}".format(args.local_rank))
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    # set optimizer
    optimizer = optim.SGD(
        [{'params': filter(lambda p: p.requires_grad, deeplab.parameters()), 'lr': args.learning_rate}],
        lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # models transformation
    model = DistributedDataParallel(deeplab)
    model = apex.parallel.convert_syncbn_model(model)
    model.train()
    model.float()
    model.cuda()

    # set loss function
    if args.ohem:
        criterion = CriterionOhemDSN(thresh=args.ohem_thres, min_kept=args.ohem_keep)  # OHEM CrossEntrop
    else:
        criterion = CriterionDSN()  # CrossEntropy
    criterion.cuda()

    cudnn.benchmark = True

    if args.world_size == 1:
        print(model)

    # this is a little different from mul-gpu traning setting in distributed training
    # because each trainloader is a process that sample from the dataset class.
    batch_size = args.gpu_num * args.batch_size_per_gpu
    max_iters = args.num_steps * batch_size / args.gpu_num
    # set data loader
    data_set = Cityscapes(args.data_dir, args.data_list, max_iters=max_iters, crop_size=input_size,
                  scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN,vars=IMG_VARS, RGB= args.rgb)

    trainloader = data.DataLoader(
        data_set,
        batch_size=args.batch_size_per_gpu, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    print("trainloader", len(trainloader))

    torch.cuda.empty_cache()

    # start training:
    for i_iter, batch in enumerate(trainloader):
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()
        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, args, i_iter, len(trainloader))
        preds = model(images)

        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        reduce_loss = all_reduce_tensor(loss,
                                        world_size=args.gpu_num)
        if args.local_rank == 0:
            Log.info('iter = {} of {} completed, lr={}, loss = {}'.format(i_iter,
                                                                      len(trainloader), lr, reduce_loss.data.cpu().numpy()))
            if i_iter % args.save_pred_every == 0 and i_iter > args.save_start:
                print('save models ...')
                torch.save(deeplab.state_dict(), osp.join(args.save_dir, str(args.arch) + str(i_iter) + '.pth'))

    end = timeit.default_timer()

    if args.local_rank == 0:
        Log.info("Training cost: "+ str(end - start) + 'seconds')
        Log.info("Save final models")
        torch.save(deeplab.state_dict(), osp.join(args.save_dir, str(args.arch) + '_final' + '.pth'))


if __name__ == '__main__':
    main()
