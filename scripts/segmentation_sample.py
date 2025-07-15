import argparse
import os
import sys
import random
from collections import OrderedDict

import numpy as np
import torch as th
from PIL import Image
import torchvision.transforms as transforms
import torch.distributed as dist
from scipy.ndimage import median_filter
import imageio
import cv2

sys.path.append(".")
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.custom_dataset_loader import CustomDataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

# 设置随机种子
seed = 10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def save_clean_mask(sample, output_path, input_image=None, overlay_path=None, threshold=0.5, min_area=100):
    prob = sample[0, 1, :, :].cpu().numpy()
    mask = (prob > threshold).astype(np.uint8)
    mask = median_filter(mask, size=3)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned_mask[labels == i] = 255

    Image.fromarray(cleaned_mask).save(output_path)
    print(f"✅ Saved binary mask to {output_path}")

    if input_image and overlay_path:
        if isinstance(input_image, str):
            image = Image.open(input_image).convert("RGB")
        else:
            image = transforms.ToPILImage()(input_image.cpu())
        image = image.resize(cleaned_mask.shape[::-1], Image.BILINEAR)
        overlay = np.array(image).copy()
        overlay_mask = Image.fromarray(cleaned_mask).resize((overlay.shape[1], overlay.shape[0]), Image.NEAREST)
        overlay_mask = np.array(overlay_mask)

        overlay[overlay_mask == 255] = [255, 0, 0]
        Image.fromarray(overlay).save(overlay_path)
        print(f"✅ Saved overlay to {overlay_path}")

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir)

    # Dataset
    if args.data_name == 'ISIC':
        transform_test = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor()
        ])
        ds = ISICDataset(args, args.data_dir, transform_test, mode='Test')
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        transform_test = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size))
        ])
        ds = BRATSDataset3D(args.data_dir, transform_test)
        args.in_ch = 5
    else:
        transform_test = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor()
        ])
        ds = CustomDataset(args, args.data_dir, transform_test, mode='Test')
        args.in_ch = 4

    datal = th.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    data = iter(datal)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))

    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict = state_dict
            break

    model.load_state_dict(new_state_dict)
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    for _ in range(len(data)):
        b, m, path = next(data)
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)

        if args.data_name == 'ISIC':
            slice_ID = os.path.splitext(os.path.basename(path[0]))[0]
        elif args.data_name == 'BRATS':
            slice_ID = path[0].split("_")[-3] + "_" + path[0].split("slice")[-1].split('.nii')[0]
        else:
            slice_ID = os.path.splitext(os.path.basename(path[0]))[0]

        logger.log("sampling...")

        for i in range(args.num_ensemble):
            model_kwargs = {}
            sample_fn = diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known

            start = th.cuda.Event(enable_timing=True)
            end = th.cuda.Event(enable_timing=True)
            start.record()

            sample, *_ = sample_fn(
                model,
                (args.batch_size, 2, args.image_size, args.image_size),
                img,
                step=args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))

            softmax_probs = th.softmax(sample, dim=1)

            # 保存crack probability map
            crack_prob = softmax_probs[:, 1, :, :]
            crack_prob_np = (crack_prob[0].cpu().numpy() * 255).astype(np.uint8)
            prob_path = os.path.join(args.out_dir, f"{slice_ID}_sample{i}_crack_prob.png")
            imageio.imwrite(prob_path, crack_prob_np)

            # 保存argmax mask
            argmax_mask = softmax_probs.argmax(dim=1).float()
            argmax_mask_np = (argmax_mask[0].cpu().numpy() * 255).astype(np.uint8)
            argmax_path = os.path.join(args.out_dir, f"{slice_ID}_sample{i}_argmax_mask.png")
            imageio.imwrite(argmax_path, argmax_mask_np)

            # 保存cleaned binary mask
            save_clean_mask(
                sample=softmax_probs,
                output_path=os.path.join(args.out_dir, f"{slice_ID}_sample{i}_binary_mask.png"),
                input_image=os.path.join(args.data_dir, path[0]),
                overlay_path=os.path.join(args.out_dir, f"{slice_ID}_sample{i}_overlay.png"),
                threshold=0.5,
                min_area=80,
            )

def create_argparser():
    defaults = dict(
        data_name='BRATS',
        data_dir="../dataset/brats2020/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=True,
        model_path="",
        num_ensemble=5,
        gpu_dev="0",
        out_dir='./results/',
        multi_gpu=None,
        debug=False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
