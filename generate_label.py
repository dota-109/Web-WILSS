from segmentation_module import make_model
import argparser
import tasks
import os
import torch
import numpy as np
from PIL import Image
from dataset import transform
from tqdm import tqdm
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "5"


_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

def make_model_old(opts):
    model_old = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))
    device = torch.device('cuda')
    checkpoint = torch.load(opts.ckpt, map_location="cpu")
    net_dict = model_old.state_dict()
    pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state'].items() if
                        (k.replace('module.', '') in net_dict) and (
                                    v.shape == net_dict[k.replace('module.', '')].shape)}
    net_dict.update(pretrained_dict)
    model_old.load_state_dict(net_dict)
    del checkpoint
    model_old.to(device)
    # freeze old model and set eval mode
    for par in model_old.parameters():
        par.requires_grad = False
    model_old.eval()
    return model_old, device

def multi_predict(file_path, opts):
    model, device = make_model_old(opts)
    files = sorted(os.listdir(file_path))
    for file in tqdm(files):
        imgs = sorted(os.listdir( os.path.join(file_path, file+"/image") ))
        dst_lbl_dir = os.path.join(file_path, file+"/label")
        tmp_img_dir = os.path.join(file_path, file+"/image")
        if len(imgs):
            if not os.path.exists(dst_lbl_dir):
                os.makedirs(dst_lbl_dir)
            for img_name in imgs:
                if os.path.isfile( os.path.join(dst_lbl_dir, img_name[:-4]+".png") ):
                    continue
                img = Image.open( os.path.join(tmp_img_dir, img_name) ).convert('RGB')
                img = _transform(img)
                img = img.to(device, dtype=torch.float32)
                label_predicted = model(img.unsqueeze(0))[0].cpu().numpy().squeeze()
                label_predicted = np.argmax(label_predicted, axis=0).astype(np.uint8)
                # visualizaiton = label2color(label_predicted)[0].astype(np.uint8)
                label_predicted = Image.fromarray(label_predicted)
                # visulized = Image.fromarray(visualizaiton)
                label_predicted.save( os.path.join(dst_lbl_dir, img_name[:-4]+".png") )
                # continue

def single_predict(img_path, opts):
    model, device = make_model_old(opts)
    img = Image.open( img_path ).convert('RGB')
    img = _transform(img)
    img = img.to(device, dtype=torch.float32)
    label_predicted = model(img.unsqueeze(0))[0].cpu().numpy().squeeze()
    label_predicted = np.argmax(label_predicted, axis=0).astype(np.uint8)
    label_predicted = Image.fromarray(label_predicted)
    label_predicted.save( "predict.png" )

def psc_predict(file_path, dst_path, opts):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    model, device = make_model_old(opts)
    img_list = sorted(glob.glob( os.path.join(file_path, "*.jpg") ))
    for im_path in tqdm(img_list):
        img_name = im_path[:-4].split("/")[-1]
        im_dst_path = os.path.join( dst_path, img_name+".png" )
        if os.path.isfile(im_dst_path):
            continue
        img = Image.open( im_path ).convert('RGB')
        img = _transform(img)
        img = img.to(device, dtype=torch.float32)
        label_predicted = model(img.unsqueeze(0))[0].cpu().numpy().squeeze()
        label_predicted = np.argmax(label_predicted, axis=0).astype(np.uint8)
        label_predicted = Image.fromarray(label_predicted)
        im_dst_path = os.path.join( dst_path, img_name+".png" )
        label_predicted.save( im_dst_path )


if __name__ == "__main__":

    parser = argparser.get_argparser()
    opts = parser.parse_args()
    opts.dataset = 'voc'
    opts.task = '10-10'
    opts.step = 0
    opts.local_rank = 0
    opts.test = True
    opts.ckpt = r"./checkpoints/step/voc-10-10/FT_bce_0.pth"
    opts = argparser.modify_command_options(opts)
    file_path = r"./replay_data/10-10-ov/"
    multi_predict(file_path, opts)

# dictionary tree: 
# Before run script:
# /10-10-ov/--->/2008_002372/--->/image/--->0001.jpg
#           --->/2008_002373/--->/image/--->0001.jpg
# After run script:
# /10-10-ov/--->/2008_002372/--->/image/--->0001.jpg
#                            --->/label/--->0001.png