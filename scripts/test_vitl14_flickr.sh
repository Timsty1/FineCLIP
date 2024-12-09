python -m training.test_flickr \
    --model-name "EVA02-CLIP-L-14-336" \
    --pretrained "./checkpoints/coco_vitl14.pt" \
    --data "./data/flickr30k/flickr30k_test.json" \
    --image-path "./data/flickr30k/flickr30k_images" \
    --image-size 336 \
    --device "cuda:0"