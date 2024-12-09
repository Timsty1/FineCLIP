import torch
import sys
sys.path.append("..")
from open_clip import create_model_and_transforms, get_tokenizer
from PIL import Image
import json
import tqdm
import os
import numpy as np

model_name = "EVA02-CLIP-B-16" 
pretrained = "/home/xiaolong_he/works/paper/logs/new_ours_only_itc_lr1e_5_bs32_epoch10/checkpoints/epoch_10.pt" # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, _, preprocess = create_model_and_transforms(
        "EVA02-CLIP-B-16",
        "eva",
        "amp",
        device="cpu",
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=None,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        aug_cfg={},
        output_dict=True,
        cache_dir=pretrained,
        det_image_size=224,
        dataset_type="grid_distill",
    )


tokenizer = get_tokenizer(model_name)
model = model.to(device)
text = torch.tensor(np.load("/home/xiaolong_he/works/paper/data/voc/text_tokens.npy")).to(device)

text_features = model.encode_text(text)
np.save("/home/xiaolong_he/works/paper/data/voc/text_embedding.npy", text_features.cpu().numpy())
print(text_features.shape)


