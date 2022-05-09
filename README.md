# Tmage

Repository extended from [AttnGAN](https://github.com/taoxugit/AttnGAN) with distillation losses and functionalities.


<img src="framework.png" width="900px" height="350px"/>

# Setup

In a new conda environment, execute `sh setup.sh` to download all the data and metadata.

# Generating Dataset from Teacher model

We use 5 Teacher models:
1. AttnGAN
2. OP-GAN
3. CP-GAN
4. XMC-GAN
5. Lafite

# Work flow
![](https://github.com/kartik2112/Tmage/blob/master/Training.gif)


**Training**
- Pre-train DAMSM models:
  - For coco dataset: `python pretrain_DAMSM.py --cfg cfg/DAMSM/coco.yml --gpu 1`
 
- Train AttnGAN models:
  - For coco dataset: `python main.py --cfg cfg/coco_attn2.yml --gpu 0`

- `*.yml` files are example configuration files for training/evaluation our models.

**Sampling**
- Run `python main.py --cfg cfg/eval_coco.yml --gpu 1` to generate examples from captions in files listed in "./data/coco/example_filenames.txt". Results are saved to `DAMSMencoders/`. 
- Change the `eval_*.yml` files to generate images from other pre-trained models. 
- Input your own sentence in "./data/coco/example_captions.txt" if you want to generate images from customized sentences. 

**Validation**
- To generate images for all captions in the validation dataset, change B_VALIDATION to True in the eval_*.yml. and then run `python main.py --cfg cfg/eval_coco.yml --gpu 1`
- We compute inception score for models trained on coco using [improved-gan/inception_score](https://github.com/openai/improved-gan/tree/master/inception_score).
