python -m training.test_mscoco \
    --model-name "EVA02-CLIP-B-16" \
    --pretrained "./checkpoints/coco_vitb16.pt" \
    --data "./data/coco/coco_test.json" \
    --image-path "./data/coco/val2017" \
    --image-size 224 \
    --device "cuda:0"