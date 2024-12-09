python -m training.test_mscoco \
    --model-name "EVA02-CLIP-L-14-336" \
    --pretrained "./checkpoints/coco_vitl14.pt" \
    --data "./data/coco/coco_test.json" \
    --image-path "./data/coco/val2017" \
    --image-size 336 \
    --device "cuda:0"