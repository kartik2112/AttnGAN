import pandas as pd

a = pd.read_csv('lafite_filename.csv', header=None)
a = a.iloc[55085:,]
b = pd.read_csv('AttnGAN_COCO_filenames.csv', header=None)

a = a.sort_values(by=[1,2]).reset_index().drop(columns=['index'])
b = b.sort_values(by=[1,2]).reset_index().drop(columns=['index'])

a[3] = b[3]
a[1] = a[1] - 1
a[2] = a[2] + 1
a[0] = a[0].apply(lambda p: p[p.rfind('/')+1:])

a.to_csv('Lafite_COCO_filenames.csv', header=None, index=None)