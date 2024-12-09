python -m training.test_flickr \
    --model-name "EVA02-CLIP-B-16" \
    --pretrained "./checkpoints/coco_vitb16.pt" \
    --data "./data/flickr30k/flickr30k_test.json" \
    --image-path "./data/flickr30k/flickr30k_images" \
    --image-size 224 \
    --device "cuda:0"