import argparse
# import sys
# sys.path.insert(1, 'AttnGAN/code/SOA')
from SOA.calculate_soa import *
from shutil import copy, rmtree
from collections import defaultdict
from tqdm import tqdm

"""
wget https://www2.informatik.uni-hamburg.de/wtm/software/semantic-object-accuracy/yolov3.weights.tar.gz -O AttnGAN/code/SOA/yolov3.weights.tar.gz
tar -xf AttnGAN/code/SOA/yolov3.weights.tar.gz -C AttnGAN/code/SOA/
rm -rf AttnGAN/code/SOA/yolov3.weights.tar.gz
"""

def organize_generated_images(source_img_dir: str, target_img_dir: str):
    captions_dir = "SOA/captions"
    all_caption_filenames = os.listdir(captions_dir)
    all_image_filenames = os.listdir(source_img_dir)

    # create map for captions:
    caption_map = defaultdict(list)
    for caption_filename in all_caption_filenames:
        with open(f'{captions_dir}/{caption_filename}', 'rb') as f:
            captions = pickle.load(f)
        for caption in captions:
            caption_map[caption['image_id']].append(caption_filename[:-4])

    # Copy imgs
    for filename in tqdm(all_image_filenames, 'copy images'):
        image_id = int(filename.split('_')[2])
        for caption_filename in caption_map[image_id]:
            destination_folder = f"{target_img_dir}/{caption_filename}"
            if not os.path.isdir(destination_folder):
                os.mkdir(destination_folder)
            copy(f"{source_img_dir}/{filename}", destination_folder)

    

def get_soa_score(images_dir: str, ):
    target_img_dir = 'SOA/data/generated_images'
    output_dir = 'SOA/data/output'

    if not os.listdir(target_img_dir):
        organize_generated_images(images_dir, target_img_dir)

    if os.path.isdir(output_dir):
        rmtree(output_dir)

    args = argparse.Namespace()
    args.images = target_img_dir
    args.output = output_dir
    args.bs = 50
    args.confidence = 0.5
    args.nms_thresh = 0.4
    args.cfgfile = 'SOA/cfg/yolov3.cfg'
    args.weightsfile = 'SOA/yolov3.weights'
    args.resolution = "256"
    args.image_size = 256
    args.iou = False
    args.gpu = 0

    run_yolo(args)
    soa_result = calc_soa(args)
    
    return soa_result

