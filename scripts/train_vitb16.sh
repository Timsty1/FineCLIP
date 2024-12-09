torchrun --nproc_per_node 8 -m training.main --batch-size=32 --lr=1e-5 --wd=0.1 --epochs=10 --workers=4 \
--model EVA02-CLIP-B-16 --pretrained eva --warmup 1000  --zeroshot-frequency 1 --dataset-type proposals_distill \
--test-type coco_panoptic --train-data data/coco/coco_proposals.json --max-boxes 20 \
--val-data data/coco/annotations/panoptic_val2017.json --image-caption-path data/coco/annotations/captions_train2017.json \
--embed-path metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy --train-image-root data/coco/train2017 \
--val-image-root data/coco/val2017  --cache-dir checkpoints/EVA02_CLIP_B_psz16_s8B.pt --log-every-n-steps 50 \
--save-frequency 1 --extract-type="v2" --image-region-caption-path "data/coco/coco_captions.json" \
--name train_vitb16 --downsample-factor 16 --det-image-size 224 --alpha 1 
