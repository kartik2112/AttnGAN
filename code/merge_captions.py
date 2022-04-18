import pickle as pkl
from tqdm import tqdm

data_dir = "../data/coco/text"

with open('../data/coco/train/filenames.pickle', 'rb') as f:
	filenames = pkl.load(f)
	with open('../data/coco/train_captions.txt', 'w') as f_w:
		for fname in tqdm(filenames):
			cap_path = "%s/%s.txt" % (data_dir, fname)
			cnt = 5
			with open(cap_path, 'r') as f_cap:
				captions = f_cap.read().split('\n')
				for cap in captions:
					if not cap.strip():
						continue
					cap = cap.replace("\ufffd\ufffd", " ")
					f_w.write(cap+"\n")
					cnt -= 1
					if cnt == 0:
						break

# with open('../data/coco/train_captions.txt') as f:
# 	with open('../data/coco/train_captions_1.txt', 'w') as f_w:
# 		for line in f:
# 			if not line.strip():
# 				continue
# 			f_w.write(line)