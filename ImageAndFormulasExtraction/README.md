# DocLayout-YOLO v2: local inference setup

## WARNING: GPT Generated from a chat history and not verified XD

## This is a vibecoded proof-of-concept, entirely relying on the amazing work DocLayout-YOLO

```BibTeX
@misc{zhao2024doclayoutyoloenhancingdocumentlayout,
      title={DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception},
      author={Zhiyuan Zhao and Hengrui Kang and Bin Wang and Conghui He},
      year={2024},
      eprint={2410.12628},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.12628},
}

@article{wang2024mineru,
  title={MinerU: An Open-Source Solution for Precise Document Content Extraction},
  author={Wang, Bin and Xu, Chao and Zhao, Xiaomeng and Ouyang, Linke and Wu, Fan and Zhao, Zhiyuan and Xu, Rui and Liu, Kaiwen and Qu, Yuan and Shang, Fukai and others},
  journal={arXiv preprint arXiv:2409.18839},
  year={2024}
}
```

This README reproduces the working setup for running `ImageAndFormulasExtraction/my_demo/test.py` with the **DocLayout-YOLO v2** universal checkpoint on Linux, using Conda and an NVIDIA GPU.

> **Important:** The v2 checkpoint needs the custom Ultralytics implementation from the upstream `release-v2` branch. The regular `ultralytics` package from PyPI alone **will not load the checkpoint**.

## Prerequisites

- Linux with an NVIDIA GPU and a working NVIDIA driver
- Conda and Git
- Python 3.11 (installed in the environment below)
- Project layout:

```text
DocumentParsing/
└── ImageAndFormulasExtraction/
    ├── my_demo/
    │   └── test.py
    └── DocLayout-YOLO/          # cloned in the steps below
```

Run the following commands from the `DocumentParsing` directory. Replace `/path/to/DocumentParsing` with your project directory.

## 1. Create and activate the environment

```bash
cd /path/to/DocumentParsing
conda create -n doclayout-v2 python=3.11 -y
conda activate doclayout-v2
python -m pip install --upgrade pip
```

## 2. Install PyTorch and the supporting packages

For the **CUDA 12.8** (`cu128`) PyTorch build:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install ultralytics huggingface_hub pymupdf
```

The system's CUDA 12.8.1 installation is compatible with a `cu128` PyTorch build, assuming the NVIDIA driver is sufficiently recent. PyTorch pip wheels include CUDA runtime dependencies; having the CUDA toolkit installed system-wide does **not** eliminate these downloads.

**To avoid downloading PyTorch/CUDA again:** instead of creating a fresh environment and running the PyTorch installation command, clone an existing Conda environment that already has CUDA-enabled PyTorch:

```bash
conda create -n doclayout-v2 --clone YOUR_EXISTING_TORCH_ENV -y
conda activate doclayout-v2
python -m pip install ultralytics huggingface_hub pymupdf
```

Use either the fresh-environment path or the cloned-environment path, not both.

## 3. Install the *v2* code, replacing the standard Ultralytics package

```bash
git clone --depth 1 --branch release-v2 https://github.com/opendatalab/DocLayout-YOLO.git ImageAndFormulasExtraction/DocLayout-YOLO
python -m pip uninstall -y ultralytics
python -m pip install --no-deps -e ImageAndFormulasExtraction/DocLayout-YOLO
```

The `--no-deps` flag prevents the fork's declared dependencies from replacing your existing PyTorch/CUDA installation. Installing the ordinary `ultralytics` package first supplies the usual Ultralytics dependencies, after which the fork replaces its code.

If you already cloned the wrong branch, switch the existing checkout instead of cloning again:

```bash
git -C ImageAndFormulasExtraction/DocLayout-YOLO fetch --depth 1 origin release-v2
git -C ImageAndFormulasExtraction/DocLayout-YOLO switch -C release-v2 FETCH_HEAD
python -m pip uninstall -y ultralytics doclayout-yolo
python -m pip install --no-deps -e ImageAndFormulasExtraction/DocLayout-YOLO
```

## 4. Verify the installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'GPU available:', torch.cuda.is_available())"
python -c "from ultralytics.nn.modules.block_bf16_autocast import HWAttentionBF16Autocast; print('DocLayout-YOLO v2 import OK')"
```

The first command should report `GPU available: True`, and the second should print `DocLayout-YOLO v2 import OK`.

## 5. Run the demo

From the `DocumentParsing` directory:

```bash
python ImageAndFormulasExtraction/my_demo/test.py
```

The demo downloads the universal checkpoint on its first run and reuses the Hugging Face cache afterward:

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

weights = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO",
    filename="universal/DocLayout-YOLO-universal.pt",
)
model = YOLO(weights)
```

If your script uses a relative image path such as `page.png`, that path is resolved relative to the directory **where you run Python**, not relative to `test.py`. Put the image in `DocumentParsing/` or update the script's path. If inference uses `save=True`, Ultralytics saves annotated output under its configured runs directory (typically `runs/detect/predict*`).

## Troubleshooting

- **`No module named 'ultralytics.nn.modules.block_bf16_autocast'`:** You have the standard Ultralytics package, or the wrong upstream branch. Repeat step 3 using `release-v2`.
- **`No module named 'ultralytics'`:** You probably installed the original `doclayout_yolo` package from the older branch. Check out `release-v2` and reinstall as in step 3.
- **`torch.cuda.is_available()` returns `False`:** Check your NVIDIA driver with `nvidia-smi` and confirm that your installed PyTorch build supports CUDA.
- **Hugging Face unauthenticated-request warning:** Usually harmless for a one-off checkpoint download. `HF_TOKEN` is optional unless you hit rate limits.
- **Disk space:** To delete the environment, run `conda deactivate && conda env remove -n doclayout-v2 -y`. `conda clean --all -y` and `python -m pip cache purge` additionally remove *shared* installation caches; they may force later redownloads for other environments.

## Recording the *exact* working versions

The setup above describes the working installation **procedure**. To preserve the exact package builds and v2 source revision from a successfully working machine, record:

```bash
conda activate doclayout-v2
python -m pip freeze > pip-freeze-working.txt
conda list --explicit > conda-linux-64-spec.txt
git -C ImageAndFormulasExtraction/DocLayout-YOLO rev-parse HEAD > doclayout-v2-commit.txt
```

Keep these files with the project. A `pip freeze` file containing `-e /absolute/local/path` is **not portable as-is**; install the v2 checkout from Git at the recorded commit on the new machine. Exact CUDA/PyTorch binaries can also depend on the host platform and NVIDIA driver.

Upstream: https://github.com/opendatalab/DocLayout-YOLO/tree/release-v2

Checkpoint: https://huggingface.co/juliozhao/DocLayout-YOLO
