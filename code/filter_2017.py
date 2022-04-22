import pandas as pd
import os
import pickle as pkl

abc = pd.read_csv("../data/coco/coco_minitrain2017.csv", header=None)
abc = abc[0]
fpath = "../data/coco/images/"
train_abc = abc[abc.apply(lambda p: os.path.exists(fpath+"COCO_train2014_"+p))]
val_abc = abc[abc.apply(lambda p: os.path.exists(fpath+"COCO_val2014_"+p))]

train_abc = train_abc.apply(lambda p: "COCO_train2014_"+p.split(".")[0])
val_abc = val_abc.apply(lambda p: "COCO_val2014_"+p.split(".")[0])
train_abc = pd.unique(train_abc)
with open("../data/coco/minicoco_train_fnames.txt", "w") as f:
	def1 = "\n".join(train_abc)
	f.write(def1)
pkl.dump(train_abc, open("../data/coco/minicoco_train_fnames.pickle", "wb"))

val_abc = pd.unique(val_abc)
with open("../data/coco/minicoco_val_fnames.txt", "w") as f:
	def1 = "\n".join(val_abc)
	f.write(def1)
pkl.dump(val_abc, open("../data/coco/minicoco_val_fnames.pickle", "wb"))