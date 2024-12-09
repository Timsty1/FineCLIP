torchrun --nproc_per_node 8 -m training.main --batch-size=32 \
--model EVA02-CLIP-L-14-336 --pretrained eva --test-type coco_panoptic --train-data="" \
--val-data data/coco/annotations/panoptic_val2017.json \
--embed-path metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTL14x336.npy \
--val-image-root data/coco/val2017  --cache-dir checkpoints/coco_vitl14.pt --extract-type="v2" \
--name test_vitl14 --downsample-factor 14 --det-image-size 336