import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from datasets.dataset_kpi import KPIsTestDataset

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/test', help='root dir for data')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--snapshot_path', type=str,
                    default='TransUNet-best.pth', help='snapshot_path')
parser.add_argument('--save_path', type=str,
                    default='./inference', help='save_path')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--is_pretrain', type=bool,
                    default=True, help='is_pretrain')
parser.add_argument('--metric_only', type=bool,
                    default=True, help='is_pretrain')
args = parser.parse_args()


def inference(args, net):
    db_test = KPIsDataset(root_dir=args.root_path)
    print("The length of test set is: {}".format(len(db_test)))
    testloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    model.eval()
    count = 0
    for sample in testloader:
        batch_imgs, batch_names = sample['image'], sample['case_name']
        batch_imgs = batch_imgs.cuda()
        outputs = net(batch_imgs)
        for name, im in zip(batch_names, outputs):
            torch.save(im, os.path.join(args.save_path, name + '.pth'))
            count += 1
            print(f'infered {count} out of {len(db_test)} samples')

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net.load_state_dict(torch.load(args.snapshot_path))

    inference(args, net)