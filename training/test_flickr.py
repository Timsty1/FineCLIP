import torch
from open_clip import create_model_and_transforms, get_tokenizer
from PIL import Image
import json
import tqdm
import os
import argparse


def main(args):
    model_name = args.model_name
    pretrained = args.pretrained
    data = args.data
    image_path = args.image_path
    image_size = args.image_size
    device = args.device
    model, _, preprocess = create_model_and_transforms(
            model_name,
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
            det_image_size=image_size,
            dataset_type="grid_distill",
        )
    tokenizer = get_tokenizer(model_name)
    model = model.to(device)
    image_list = []
    caption_list = []
    image_fea = []
    caption_fea = []
    txt2img = {}
    img2txt = {}
    txt_id = 0
    with open(data) as f:
        annotation = json.load(f)
        for img_id, ann in enumerate(annotation):
            img2txt[img_id] = []
            image_list.append(ann['image'])
            for caption in ann['caption']:
                caption_list.append(caption)
                img2txt[img_id].append(txt_id)
                txt2img[txt_id] = img_id
                txt_id += 1

    with torch.no_grad(), torch.cuda.amp.autocast():
        tmp_image_list = []
        for i in tqdm.tqdm(range(len(image_list))):
            tmp_image_list.append(preprocess[0](Image.open(os.path.join(image_path, image_list[i]))).unsqueeze(0).to(device))
            if len(tmp_image_list) % 64 == 0 or i == len(image_list)-1:
                tmp_image = torch.cat(tmp_image_list, dim=0)
                image_features = model.encode_image(tmp_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_fea.append(image_features)
                tmp_image_list = []

        tmp_text_list = []
        for i in tqdm.tqdm(range(len(caption_list))):
            tmp_text_list.append(caption_list[i])
            if len(tmp_text_list) % 64 == 0 or i == len(caption_list)-1:
                text = tokenizer(tmp_text_list).to(device)
                text_features = model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                caption_fea.append(text_features)
                tmp_text_list = []

    image_fea_total = torch.cat(image_fea, dim=0)
    caption_fea_total = torch.cat(caption_fea, dim=0)
    sims = image_fea_total@caption_fea_total.t()
    _, topk_idx = sims.topk(k=1, dim=0)
    count = 0

    for i in range(topk_idx.shape[1]):
        if topk_idx[0,i] == txt2img[i]:
            count += 1
    print("文搜图的准确率为:{:.2f}%".format(100*count/topk_idx.shape[1]))

    sims = sims.t()
    _, topk_idx = sims.topk(k=1, dim=0)

    count = 0
    new_list = []
    for i in range(topk_idx.shape[1]):
        if topk_idx[0,i] in img2txt[i]:
            count += 1
        else:
            new_list.append({"image": image_list[i], "caption":caption_list[img2txt[i][0]]})
    print("图搜文的准确率为:{:.2f}%".format(100*count/topk_idx.shape[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="EVA02-CLIP-B-16",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="./checkpoints/coco_vitb16.pt",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./data/flickr30k/flickr30k_test.json"
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="./data/flickr30k/flickr30k_images"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0"
    )
    args = parser.parse_args()
    main(args)