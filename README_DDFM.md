# DDFM for Segmentation

This example extends **MedSegDiff** with a minimal implementation of
Denoising Diffusion Flow Matching (DDFM). The new diffusion class is
`guided_diffusion.ddfm.DDFMDiffusion` which currently inherits from
`SpacedDiffusion`.

The training and sampling scripts are:

- `scripts/ddfm_segmentation_train.py`
- `scripts/ddfm_segmentation_sample.py`

Both scripts mirror the original `segmentation_*` tools but create a
DDFM diffusion process using `create_model_and_ddfm` from
`guided_diffusion.script_util`.

Run training with for example:

```bash
python scripts/ddfm_segmentation_train.py --data_dir <data> --out_dir <results>
```

Sampling follows the same pattern:

```bash
python scripts/ddfm_segmentation_sample.py --data_dir <data> --model_path <ckpt>
```
