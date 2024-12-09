import logging
import torch
import torch.nn.functional as F
from training.dist_utils import all_gather
from tqdm import tqdm
from .distributed import is_master
from open_clip import get_cast_dtype
from .precision import get_autocast


def run(model, dataloader, args):
    cls_embeddings = dataloader.dataset.embeddings
    cls_embeddings = F.normalize(torch.from_numpy(cls_embeddings).float(), dim=-1)
    cls_embeddings = cls_embeddings.to(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    if cast_dtype is not None:
        cls_embeddings = cls_embeddings.to(dtype=cast_dtype)
    with torch.no_grad():
        correct_rois = []
        all_cls_labels = []
        for images, bboxes, _, gt_masks,  _ \
                in tqdm(dataloader, disable=not is_master(args)):
            images = images.to(args.device)
            bboxes = bboxes.to(args.device)
            gt_masks = gt_masks.to(args.device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
                bboxes = bboxes.to(dtype=cast_dtype)
                gt_masks = gt_masks.to(dtype=cast_dtype)
            cls_labels = []
            rois = []
            for bboxes_per_image in bboxes:
                valid = bboxes_per_image[:, 5] > 0.5
                rois.append(bboxes_per_image[valid, :4])
                cls_labels.append(bboxes_per_image[valid, 4])
            cls_labels = torch.cat(cls_labels, dim=0).to(torch.long)
            if cls_labels.shape[0] == 0:
                continue
            with autocast():
                # predict
                if args.distributed and not args.horovod:
                    module = model.module
                else:
                    module = model
                roi_extractor = module.encode_pseudo_boxes
                roi_features = roi_extractor(images, rois, normalize=True,
                                             extract_type=args.extract_type)

                if cast_dtype is not None:
                    roi_features = roi_features.to(dtype=cast_dtype)

                roi_logits = roi_features @ cls_embeddings.T

            _, roi_top5_inds = roi_logits.topk(5)
            correct_rois.append(roi_top5_inds == cls_labels.view(-1, 1))
            all_cls_labels.append(cls_labels)

        correct_rois = torch.cat(correct_rois).float()
        all_cls_labels = torch.cat(all_cls_labels)
        if args.distributed and not args.horovod:
            correct_rois = multi_gpu_sync(correct_rois)
            all_cls_labels = multi_gpu_sync(all_cls_labels)

    return correct_rois, all_cls_labels


def multi_gpu_sync(x):
    device = x.device
    x_list = all_gather(x.cpu())
    x = torch.cat([res.to(device) for res in x_list])
    return x


def macc_with_box(correct_matrix, all_cls_labels, prefix):
    def _macc(corrects, cls_labels):
        min_id = cls_labels.min().item()
        max_id = cls_labels.max().item()
        cand_labels = list(range(min_id, max_id+1))

        acc_per_cls = []

        for lb in cand_labels:
            corrects_per_cls = corrects[cls_labels == lb]
            if corrects_per_cls.shape[0] == 0:
                continue
            acc_per_cls.append(corrects_per_cls.mean().half().item())

        return sum(acc_per_cls) / len(acc_per_cls)

    results = {}

    box_top1_acc = _macc(correct_matrix[:, 0], all_cls_labels)
    box_top5_acc = _macc(correct_matrix.sum(-1), all_cls_labels)

    results[f'{prefix}.box.macc1'] = box_top1_acc
    results[f'{prefix}.box.macc5'] = box_top5_acc

    return results


def zero_shot_eval(model, data, epoch, args):
    if 'val' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    logging.info('Region classifier')
    results = {}
    correct_rois, all_cls_labels = run(model, data['val'].dataloader, args)
    results.update(macc_with_box(correct_rois, all_cls_labels, 'rois'))
    return results
