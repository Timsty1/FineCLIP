import random
import torch
import torch.nn.functional as F


class CLIPSelf:
    def __call__(self, batch, model, dist_model, itc_loss, device, cast_dtype, distributed, args):
        if distributed:
            model = model.module
            dist_model = dist_model.module
        images, normed_boxes, image_crops, texts, region_texts = batch

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        normed_boxes = normed_boxes.to(device=device, dtype=cast_dtype, non_blocking=True)
        image_crops = image_crops.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.squeeze(1)    # alignment dimension
        region_texts = region_texts.to(device=device, dtype=cast_dtype, non_blocking=True)

        if args.multiscale:
            cur_h, cur_w = images.shape[2:]
            assert cur_h == cur_w
            if cur_h == 1024:
                tar_sizes = [320, 640, 896, 1024]
            elif cur_h == 896:
                tar_sizes = [336, 448, 672, 896]
            else:
                raise NotImplementedError
            tar_size = random.choice(tar_sizes)
            images = F.interpolate(images, size=(tar_size, tar_size), mode='bilinear')

        rois_list = []
        crops_list = []
        region_text_list = []
        for bboxes_per_image, crops_per_image, region_per_text in zip(normed_boxes, image_crops, region_texts):
            valid = bboxes_per_image[:, -1] > 0.5
            rois_list.append(bboxes_per_image[valid, :4])
            crops_list.append(crops_per_image[valid])
            region_text_list.append(region_per_text[valid])

        image_crops = torch.cat(crops_list)
        region_texts = torch.cat(region_text_list)
        teacher_crop_features = model.encode_image(image_crops, normalize=False)
        student_roi_features = model.encode_pseudo_boxes(images, rois_list, normalize=False, extract_type=args.extract_type)
        region_text_features = model.encode_text(region_texts, normalize=False)

        normed_student_features = F.normalize(student_roi_features, dim=-1)
        normed_teacher_features = F.normalize(teacher_crop_features, dim=-1)
        normed_region_text_features = F.normalize(region_text_features, dim=-1)
        losses = {}
        if "distill" in args.loss_type:
            loss_cosine = 1.0 - (normed_student_features * normed_teacher_features).sum(-1).mean()
            losses["distill"] = loss_cosine

        if "global_itc" in args.loss_type:
            image_features, text_features, logit_scale = model(images, texts)
            global_itc_loss, _ = itc_loss(image_features, text_features, logit_scale)
            losses["global_itc"] = global_itc_loss

        if "region_itc" in args.loss_type:
            assert normed_student_features.shape[0] == normed_region_text_features.shape[0]
            all_student_features = torch.zeros((args.batch_size*args.max_boxes, normed_student_features.shape[1])).to(device=device, dtype=cast_dtype, non_blocking=True)
            all_student_features[0:normed_student_features.shape[0]] = normed_student_features

            all_region_text_features = torch.zeros((args.batch_size*args.max_boxes, normed_student_features.shape[1])).to(device=device, dtype=cast_dtype, non_blocking=True)
            all_region_text_features[0:normed_region_text_features.shape[0]] = normed_region_text_features
            
            region_itc_loss, _ = itc_loss(all_student_features, all_region_text_features, model.logit_scale.exp())
            losses["region_itc"] = region_itc_loss

        return losses, len(images), model.logit_scale.exp()
