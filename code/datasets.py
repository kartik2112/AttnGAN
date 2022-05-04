from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
from tqdm import tqdm

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def prepare_data(data, added_data=''):
    if added_data == '':
        imgs, captions, captions_lens, class_ids, keys = data
    else:
        imgs, captions, captions_lens, class_ids, keys, extra_data_1, extra_data_2 = data


    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if added_data == 'caps':
        extra_data_1 = np.array(extra_data_1)[sorted_cap_indices]
        extra_data_2 = extra_data_2[sorted_cap_indices]
    elif added_data == 'real_imgs':
        for i in range(len(extra_data_1)):
            extra_data_1[i] = extra_data_1[i][sorted_cap_indices]
        extra_data_2 = extra_data_2[sorted_cap_indices]
    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)
    if added_data == '':
        return [real_imgs, captions, sorted_cap_lens,
            class_ids, keys]
    else:
        return [real_imgs, captions, sorted_cap_lens,
            class_ids, keys, extra_data_1, extra_data_2]


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
            # print(imsize[i])
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Scale(imsize[i])(img)
            else:
                re_img = img
            ret.append(normalize(re_img))

    return ret


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 base_size=64,
                 transform=None, target_transform=None, incl_caption_texts=False):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        # if incl_caption_texts:
        #     self.filenames, self.captions, self.ixtoword, \
        #         self.wordtoix, self.n_words, self.caption_texts = self.load_text_data(data_dir, split, incl_caption_texts)
        # else:
        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in xrange(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, filenames):
        all_captions = []
        all_caption_texts = []
        for i in tqdm(range(len(filenames))):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_caption_texts.append(cap)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions, all_caption_texts

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split, incl_caption_texts=False):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions, _ = self.load_captions(data_dir, train_names)
            test_captions, _ = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        # if incl_caption_texts and split == 'train':
        #     caption_texts = []
        #     cap_fname = "%s/train_captions.txt" % (data_dir,)
        #     print(f"Load captions from {cap_fname}")
        #     with open(cap_fname, "r") as f:
        #         for line in f:
        #             if not line:
        #                 continue
        #             caption_texts.append(line)
        #     assert len(caption_texts) == len(filenames) * 5
        #     # if split == 'train':
        #     #     train_caption_texts = self.load_captions(data_dir, train_names)
        #     #     caption_texts = train_caption_texts
        #     # else:  # split=='test'
        #     #     test_caption_texts = self.load_captions(data_dir, test_names)
        #     #     caption_texts = test_caption_texts
        #     return filenames, captions, ixtoword, wordtoix, n_words, caption_texts
        # else:
        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def __getitem__(self, index):
        #
        try:
            key = self.filenames[index]
            cls_id = self.class_id[index]
            #
            if self.bbox is not None:
                bbox = self.bbox[key]
                data_dir = '%s/CUB_200_2011' % self.data_dir
            else:
                bbox = None
                data_dir = self.data_dir
            #
            img_name = '%s/images/%s.jpg' % (data_dir, key)
            imgs = get_imgs(img_name, self.imsize,
                            bbox, self.transform, normalize=self.norm)
            # random select a sentence
            sent_ix = random.randint(0, self.embeddings_num)
            new_sent_ix = index * self.embeddings_num + sent_ix
            caps, cap_len = self.get_caption(new_sent_ix)
        except Exception as e:
            print(index, key, cls_id)
            raise e
        return imgs, caps, cap_len, cls_id, key


    def __len__(self):
        return len(self.filenames)

class TextDataset_Generator(TextDataset):
    def __init__(self, data_dir, split='train',
                 base_size=64,
                 transform=None, target_transform=None, incl_caption_texts=True):
        super().__init__(data_dir, split, base_size, transform, target_transform, incl_caption_texts)
        self.filenames = self.load_filenames(data_dir, "minicoco_%s_fnames" % (split,))
        if os.path.isfile("%s/minicoco_%s_captions.pickle" % (data_dir, split)):
            self.caption_texts, self.captions = pickle.load(open("%s/minicoco_%s_captions.pickle" % (data_dir, split), 'rb'))
            
        else:
            caption_text_tokens, self.caption_texts = self.load_captions(data_dir, self.filenames)
            self.captions = self.encode_captions(caption_text_tokens)
            pickle.dump([self.caption_texts, self.captions], open("%s/minicoco_%s_captions.pickle" % (data_dir, split), 'wb'))
        
        split_dir = os.path.join(data_dir, split)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

    def load_filenames(self, data_dir, fname):
        filepath = '%s/%s.pickle' % (data_dir, fname)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def encode_captions(self, captions):
        captions_new = []
        for t in captions:
            rev = []
            for w in t:
                if w in self.wordtoix:
                    rev.append(self.wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            captions_new.append(rev)
        return captions_new

    def __getitem__(self, index):
        #
        try:
            key = self.filenames[index//self.embeddings_num]
            cls_id = self.class_id[index//self.embeddings_num]
            bbox = None
            data_dir = self.data_dir
            #
            img_name = '%s/images/%s.jpg' % (data_dir, key)
            imgs = get_imgs(img_name, self.imsize,
                            bbox, self.transform, normalize=self.norm)
            caps, cap_len = self.get_caption(index)
        except Exception as e:
            print(index, key, cls_id)
            raise e
        return imgs, caps, cap_len, cls_id, key, self.caption_texts[index], index
    
    def __len__(self):
        return len(self.caption_texts)
    
class DistilTextDataset(TextDataset):
    # TODO: Determine how many images to retrieve from reference file
    def __init__(self, data_dir, split='train',
                 base_size=64,
                 transform=None, target_transform=None, file_prefix="AttnGAN_COCO", fake_count=-1):
        '''
        Main Files:
        self.filenames
        self.captions
        self.noise
        '''
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        # self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        self.bbox = None
        split_dir = os.path.join(data_dir, split)

        _, _, self.ixtoword, self.wordtoix, self.n_words = self.load_text_data("../data/coco", split)

        self.file_prefix = file_prefix

        self.noise = np.load("%s/%s_noise.npy" % (self.data_dir, self.file_prefix)).astype(np.float32)
        self.filenames = []
        with open("%s/%s_filenames.csv" % (self.data_dir, self.file_prefix)) as f:
            for line in f:
                line = line.strip().split(",")
                line[1] = int(line[1])
                line[2] = int(line[2])
                self.filenames.append(line)
        # self.noise = self.noise[:len(self.filenames),:]
        assert self.noise.shape[0] == len(self.filenames), (self.noise.shape, len(self.filenames))
        self.captions = self.load_and_encode_captions()

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)
    
    def load_and_encode_captions(self):
        if os.path.isfile("%s/%s_captions.pickle" % (self.data_dir, self.file_prefix)):
            return pickle.load(open("%s/%s_captions.pickle" % (self.data_dir, self.file_prefix), 'rb'))

        captions = []
        with open("%s/%s_captions.txt" % (self.data_dir, self.file_prefix)) as f:
            for cap in f:
                cap = cap.strip()
                cap = cap.replace("\ufffd\ufffd", " ")
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(cap.lower())
                if len(tokens) == 0:
                    print('cap', cap)
                    continue
                tokens_new = []
                for t in tokens:
                    t = t.encode('ascii', 'ignore').decode('ascii')
                    if len(t) > 0:
                        tokens_new.append(t)
                captions.append(tokens_new)
        captions_new = []
        for t in captions:
            rev = []
            for w in t:
                if w in self.wordtoix:
                    rev.append(self.wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            captions_new.append(rev)
        pickle.dump(captions_new, open("%s/%s_captions.pickle" % (self.data_dir, self.file_prefix), "wb"))
        return captions_new
    
    def __getitem__(self, index):
        #
        try:
            key = self.filenames[index][0]
            real_key = self.filenames[index][3]
            # cls_id = self.class_id[index//self.embeddings_num]
            cls_id = self.class_id[index]
            bbox = None
            data_dir = self.data_dir
            # #
            gen_img_name = '%s/images/%s' % (data_dir, key)
            gen_img = get_imgs(gen_img_name, self.imsize,
                            bbox, self.transform, normalize=self.norm)
            caps, cap_len = self.get_caption(self.filenames[index][1])

            real_img_name = '%s/images/%s.jpg' % ('../data/coco/', real_key)
            real_img = get_imgs(real_img_name, self.imsize,
                            bbox, self.transform, normalize=self.norm)
            noise = self.noise[index]
        except Exception as e:
            print(index, key, cls_id)
            raise e
        return gen_img, caps, cap_len, cls_id, key, real_img, noise
    
    def __len__(self):
        return len(self.filenames)