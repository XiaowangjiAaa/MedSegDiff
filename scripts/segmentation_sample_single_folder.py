import argparse, os, sys
from collections import OrderedDict
import numpy as np
import torch as th
from PIL import Image
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from scipy.ndimage import median_filter
import imageio, cv2

sys.path.append(".")
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

class ImageFolderDataset(th.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = [os.path.join(root_dir, f) for f in sorted(os.listdir(root_dir))
                            if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            path = self.image_paths[idx]
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, path
        except Exception as e:
            print(f"⚠️ Failed to load image: {e}")
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return th.utils.data.dataloader.default_collate(batch)

def save_clean_mask(sample, output_path, input_image=None, overlay_path=None, threshold=0.5, min_area=100):
    prob = sample[0, 1, :, :].cpu().numpy()
    mask = (prob > threshold).astype(np.uint8)
    mask = median_filter(mask, size=3)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned_mask[labels == i] = 255

    Image.fromarray(cleaned_mask).save(output_path)

    if input_image and overlay_path:
        image = Image.open(input_image).convert("RGB")
        image = image.resize(cleaned_mask.shape[::-1], Image.BILINEAR)
        overlay = np.array(image)
        overlay_mask = cleaned_mask
        overlay[overlay_mask == 255] = [255, 0, 0]
        Image.fromarray(overlay).save(overlay_path)

def main():
    args = create_argparser().parse_args()
    args.in_ch = 4

    dist_util.setup_dist(args)
    local_rank = int(os.environ["LOCAL_RANK"])
    th.cuda.set_device(local_rank)
    device = th.device(f"cuda:{local_rank}")

    logger.configure(dir=os.path.join(args.out_dir, f"rank{local_rank}"))

    # Load model
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model = DDP(model, device_ids=[local_rank])
    model.eval()

    # Create dataset & dataloader
    print(f"📂 Processing folder: {args.data_dir}")

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])
    dataset = ImageFolderDataset(args.data_dir, transform=transform)
    if len(dataset) == 0:
        print(f"⚠️ No images found in {args.data_dir}")
        return

    sampler = th.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    os.makedirs(args.out_dir, exist_ok=True)

    sampler.set_epoch(0)
    for b, paths in dataloader:
        b = b.to(device)
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)

        for i in range(args.num_ensemble):
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
                model_kwargs={}
            )

            end.record()
            th.cuda.synchronize()
            print(f"[GPU {local_rank}] Time for 1 sample: {start.elapsed_time(end):.2f} ms")

            softmax_probs = th.softmax(sample, dim=1)
            for j in range(b.shape[0]):
                slice_ID = os.path.splitext(os.path.basename(paths[j]))[0]
                crack_prob = (softmax_probs[j, 1].cpu().numpy() * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(args.out_dir, f"{slice_ID}_sample{i}_crack_prob.png"), crack_prob)

                argmax_mask = (softmax_probs[j].argmax(0).cpu().numpy() * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(args.out_dir, f"{slice_ID}_sample{i}_argmax_mask.png"), argmax_mask)

                save_clean_mask(
                    sample=softmax_probs[j:j+1],
                    output_path=os.path.join(args.out_dir, f"{slice_ID}_sample{i}_binary_mask.png"),
                    input_image=paths[j],
                    overlay_path=os.path.join(args.out_dir, f"{slice_ID}_sample{i}_overlay.png"),
                    threshold=0.5,
                    min_area=80
                )

    dist.destroy_process_group()

def create_argparser():
    defaults = dict(
        data_name='folder',
        data_dir="../dataset/image_folder",
        clip_denoised=True,
        num_samples=1,
        batch_size=8,
        use_ddim=False,
        model_path="",
        num_ensemble=1,
        gpu_dev="0",
        out_dir='./results/',
        debug=False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
