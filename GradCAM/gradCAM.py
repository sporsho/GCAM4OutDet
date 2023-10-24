#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import sys

import numpy as np
import torch
import yaml
from tqdm import tqdm
import importlib
od = importlib.import_module("3D_OutDet")
deterministic = importlib.import_module("3D_OutDet.deterministic")
collate = importlib.import_module("3D_OutDet.dataset.utils.collate")
ds = importlib.import_module("3D_OutDet.dataset")
import warnings

from wrapper import OutDetWithGCAM

warnings.filterwarnings("ignore")





def get_seq_name_from_path(path):
    tmps = path.split(os.path.sep)
    seq = tmps[-3]
    name = tmps[-1]
    tmps2 = name.split(".")
    name = tmps2[0]
    return seq, name


def main(args):
    data_path = args.data_dir
    test_batch_size = args.test_batch_size
    model_save_path = args.model_save_path
    device = torch.device(args.device)
    dilate = 1

    with open(args.label_config, 'r') as stream:
        config = yaml.safe_load(stream)

    class_strings = config["labels"]
    class_inv_remap = config["learning_map_inv"]
    num_classes = len(class_inv_remap)

    keys = class_inv_remap.keys()
    max_key = max(keys)
    look_up_table = np.zeros((max_key + 1), dtype=np.int32)
    for k, v in class_inv_remap.items():
        look_up_table[k] = v

    ordered_class_names = [class_strings[class_inv_remap[i]] for i in range(num_classes)]
    # prepare model
    model = OutDetWithGCAM(num_classes=num_classes, kernel_size=args.K, depth=1, dilate=dilate)
    model = model.to(device)
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
    else:
        raise ValueError()

    # prepare dataset
    tree_k = int(np.round(args.K * args.K))
    test_dataset = ds.WadsPointCloudDataset(device, data_path + '/sequences/', imageset='test',
                                           label_conf=args.label_config, k=tree_k,
                                         desnow_root=args.desnow_root, pred_folder=args.pred_folder,
                                         snow_label=args.snow_label, recalculate=True, save_ind=False)
    if test_dataset.save_ind:
        collate_fn = collate.collate_fn_cp
    else:
        collate_fn = collate.collate_fn_cp_inference
    test_dataset_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                      batch_size=test_batch_size,
                                                      shuffle=False,
                                                      num_workers=0,
                                                      collate_fn=collate_fn)

    # validation
    print('*' * 80)
    print('Test network performance on validation split')
    print('*' * 80)
    pbar = tqdm(total=len(test_dataset_loader))
    for i_iter_val, batch in enumerate(
            test_dataset_loader):
        data = batch['data'][0].to(device)
        ind = batch['ind'][0]
        dist = batch['dist'][0].to(device)
        label = batch['label'][0].long().to(device)
        model.eval()
        with torch.no_grad():
            act = model.forward_to_last_activation(data, dist, ind)
            feats = model.drop(act)
        with torch.enable_grad():
            grads = list()
            feats.requires_grad = True
            feats.register_hook(lambda g: grads.append(g))
            logit = model.fc(feats)
            sm = torch.softmax(logit, dim=1)
            grad_y = torch.zeros_like(sm)
            grad_y[:, 1] = 1
            sm.backward(grad_y)
        grads = grads[0]
        grads = grads * act
        grads = grads.sum(1)
        grads = torch.nn.functional.relu(grads)
        grads = grads.detach().cpu().numpy()
        grads = grads / np.max(grads)
        pbar.update()
        if args.save_grad:
            grads = grads.astype(np.float32)
            path_seq, name = get_seq_name_from_path(test_dataset.im_idx[i_iter_val])
            path_name = name + ".grad"
            path = os.path.join(args.test_output_path, "sequences",
                                path_seq, "grads", path_name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            grads.tofile(path)

    pbar.close()




if __name__ == '__main__':
    # Testing settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data_dir', default='/var/local/home/aburai/DATA/WADS2')
    parser.add_argument("-label_config", type=str, default='../3D_OutDet/binary_desnow_wads.yaml')
    parser.add_argument('-p', '--model_save_path',
                        default='/var/local/home/aburai/DATA/3D_OutDet/bin_desnow_wads/outdet.pt')
    parser.add_argument('-o', '--test_output_path',
                        default='/var/local/home/aburai/DATA/3D_OutDet/bin_desnow_wads/outputs')
    parser.add_argument('-m', '--model', choices=['polar', 'traditional'], default='polar',
                        help='training model: polar or traditional (default: polar)')
    parser.add_argument('--device', type=str, default='cuda:0', help='validation interval (default: 4000)')
    parser.add_argument('--K', type=int, default=3, help='batch size for training (default: 2)')

    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for training (default: 1)')

    parser.add_argument(
        '--desnow_root', '-dr',
        type=str,
        default=None,
        help='Set this if you want to use the Uncertainty Version'
    )
    parser.add_argument("--pred_folder",
                        type=str,
                        default=None)
    parser.add_argument('--snow_label',
                        type=int,
                        default=None)

    parser.add_argument('--save_grad', type=bool, default=False)
    args = parser.parse_args()


    print(' '.join(sys.argv))
    print(args)
    torch.backends.cuda.matmul.allow_tf32 = True
    deterministic.configure_randomness(12345)
    main(args)
