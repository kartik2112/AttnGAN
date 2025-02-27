from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from datasets import TextDataset, TextDataset_Generator, DistilTextDataset
from trainer import condGANTrainer as trainer

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
# sys.path.append("../../improved-gan")
from miscc.inception_score_computer import get_inception_score
from miscc.soa_score_computer import get_soa_score
# from inception_score_pytorch.inception_score import inception_score
from tqdm import tqdm

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_attn2.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def gen_example(wordtoix, algo):
    '''generate images from example sentences'''
    from nltk.tokenize import RegexpTokenizer
    filepath = '%s/example_filenames.txt' % (cfg.DATA_DIR)
    data_dic = {}
    with open(filepath, "r") as f:
        filenames = f.read().split('\n')
        for name in filenames:
            if len(name) == 0:
                continue
            filepath = '%s/%s.txt' % (cfg.DATA_DIR, name)
            with open(filepath, "r") as f:
                print('Load from:', name)
                sentences = f.read().split('\n')
                # a list of indices for a sentence
                captions = []
                cap_lens = []
                for sent in sentences:
                    if len(sent) == 0:
                        continue
                    sent = sent.replace("\ufffd\ufffd", " ")
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(sent.lower())
                    if len(tokens) == 0:
                        print('sent', sent)
                        continue

                    rev = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0 and t in wordtoix:
                            rev.append(wordtoix[t])
                    captions.append(rev)
                    cap_lens.append(len(rev))
            max_len = np.max(cap_lens)

            sorted_indices = np.argsort(cap_lens)[::-1]
            cap_lens = np.asarray(cap_lens)
            cap_lens = cap_lens[sorted_indices]
            cap_array = np.zeros((len(captions), max_len), dtype='int64')
            for i in range(len(captions)):
                idx = sorted_indices[i]
                cap = captions[idx]
                c_len = len(cap)
                cap_array[i, :c_len] = cap
            key = name[(name.rfind('/') + 1):]
            data_dic[key] = [cap_array, cap_lens, sorted_indices]
    algo.gen_example(data_dic)


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        # bshuffle = False
        split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])

    if cfg.TRAIN.NET_G != '' and not cfg.TRAIN.NET_G.endswith('.pth') and (cfg.B_VALIDATION or cfg.DISTIL.FLAG):
        shortlisted_model_fnames = []
        for fname in os.listdir(cfg.TRAIN.NET_G):
            if os.path.isfile(cfg.TRAIN.NET_G + fname) and fname.startswith('netG_epoch'):
                shortlisted_model_fnames.append(int(fname[fname.rfind("_")+1:fname.rfind(".")]))
        shortlisted_model_fnames.sort()
        cfg.TRAIN.NET_G += "netG_epoch_" + str(shortlisted_model_fnames[-1]) + ".pth"
        print("Updated NET_G path to %s" % (cfg.TRAIN.NET_G, ))

    start_t = time.time()
    if cfg.DISTIL.FLAG:
        if cfg.DISTIL.PXL_LOSS is False:
            cfg.DISTIL.PIX_DIST_LAMBDA_START = 0
        if cfg.DISTIL.DISC_LOSS is False:
            cfg.DISTIL.DISC_DIST_LAMBDA_START = 0
        if cfg.DISTIL.TRUE_LOSS is False:
            cfg.DISTIL.TRUE_LOSS_ALPHA = 0

        dataset = DistilTextDataset(cfg.DATA_DIR, split_dir,
                          base_size=cfg.TREE.BASE_SIZE, file_prefix=cfg.FILE_PREFIX,
                          transform=image_transform)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
            drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

        # Define models and go to train/evaluate
        algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)
        algo.distil()
    elif cfg.TRAIN.FLAG:
        dataset = TextDataset(cfg.DATA_DIR, split_dir,
                            base_size=cfg.TREE.BASE_SIZE,
                            transform=image_transform)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
            drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))
        

        # Define models and go to train/evaluate
        algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)
        algo.train()
    else:
        
        if cfg.GEN_IMAGES:
            pass
            dataset = TextDataset_Generator(cfg.DATA_DIR, 'train',
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
            assert dataset
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                drop_last=False, shuffle=False, num_workers=int(cfg.WORKERS))

            # Define models and go to train/evaluate
            algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword, dataset.__len__())

            algo.gen_dataset()
        else:
            
            '''generate images from pre-extracted embeddings'''
            if cfg.B_VALIDATION:
                dataset = TextDataset_Generator(cfg.DATA_DIR, 'val',
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
                assert dataset
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                    drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

                # Define models and go to train/evaluate
                algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)
                
                algo.sampling(split_dir)  # generate images for the whole valid dataset
                images1 = []
                dir1 =cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')] + '/valid/single/'
                for imgname in tqdm(os.listdir(dir1)):
                    img = Image.open(dir1 + imgname).convert('RGB')
                    images1.append(np.array(img))
                print("For model at %s" % (cfg.TRAIN.NET_G,))
                pprint.pprint(cfg)
                print("Inception Scores:",get_inception_score(images1))
                print("SOA Scores:", get_soa_score(images_dir=dir1))

                # print("Inception Scores:",inception_score('../models/coco_AttnGAN2/valid/single/', cuda=True, batch_size=1000, num_workers=int(cfg.WORKERS)))
            else:
                dataset = TextDataset(cfg.DATA_DIR, split_dir,
                                base_size=cfg.TREE.BASE_SIZE,
                                transform=image_transform)
                assert dataset
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                    drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

                # Define models and go to train/evaluate
                algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)
                gen_example(dataset.wordtoix, algo)  # generate images for customized captions
    end_t = time.time()
    print('Total time for task:', end_t - start_t)
