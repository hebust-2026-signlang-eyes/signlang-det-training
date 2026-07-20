# signlang-det-training

[English](README.md) | [简体中文](README.zh-CN.md)

Training workflow for a reusable dynamic sign language feature encoder, packaged as a self-contained notebook designed to run on Kaggle, 
The training results can be seamlessly applied to [SignLang Eyes](https://github.com/hebust-2026-signlang-eyes/signlang-eyes).

## Overview 🚀

This project trains `signlang_det`, a lightweight hand-motion encoder for prototype-based sign retrieval. It is designed for end-to-end training in a Kaggle GPU session. The encoder learns general temporal representations from the Google Isolated Sign Language Recognition corpus and then adapts them to a target landmark collection.

The temporary classification heads are used only for optimization. They are not included in the exported encoder and do not restrict the vocabulary used at inference time.

The recognition approach used in this project was heavily inspired by the 1st solution of the [Google - Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs) competition: [https://www.kaggle.com/competitions/asl-signs/writeups/hoyeol-sohn-1st-place-solution-1dcnn-combined-with](https://www.kaggle.com/competitions/asl-signs/writeups/hoyeol-sohn-1st-place-solution-1dcnn-combined-with)

The complete workflow is available in [signlang_det_kaggle_training.ipynb](signlang_det_kaggle_training.ipynb).

Tagged revisions can also be delivered through Kaggle CD: GitHub submits each queued tag to one stable Kaggle notebook, then a repository writer uploads the completed ZIP through a locked Issue. The Issue workflow validates and converts candidates in order before publishing the matching GitHub Release. See [Kaggle Continuous Delivery](docs/KAGGLE_CD.md) for setup and recovery instructions.

## Highlights ✨

- Shared `hand168-temporal` preprocessing across training, evaluation, and inference
- Wrist-relative normalization for two hands with explicit missing-hand handling
- Shared hand TCN branches followed by masked Transformer fusion
- Cross-entropy, supervised contrastive, and batch-hard triplet objectives
- Class-disjoint validation and test splits for open-vocabulary evaluation
- Pooled cosine retrieval with constrained frame-level DTW reranking
- Resumable memory-mapped preprocessing and atomic checkpoints
- Standalone encoder export with a SHA-256 fingerprint

## Training Workflow 🧭

The notebook runs the complete pipeline from top to bottom:

1. Validate the Kaggle CUDA runtime and selected GPU architecture.
2. Convert Google ASL parquet sequences into a resumable `hand168-temporal` cache.
3. Train general hand-motion representations with P-by-K sampling.
4. Select and evaluate the encoder on classes excluded from training.
5. Audit and preprocess the target object-NPY landmark collection.
6. Load only the selected encoder, create a new auxiliary classifier, and adapt the representation.
7. Evaluate pooled retrieval and Top-K DTW reranking.
8. Export the final encoder, configuration, metrics, splits, logs, and fingerprint.

### Feature contract

| Item | Contract |
|---|---|
| Input | `float32`, `B × 64 × 168` |
| Per-frame features | 2 hands × 21 landmarks × `(x, y, z, velocity)` |
| Padding value | `-100.0` |
| Encoder output | `float32`, `B × 64 × 128` |
| Valid output frames | L2-normalized |
| Padding output frames | All zeros |

## Data Requirements 📚

Attach both datasets to the Kaggle notebook before running it.

### Source corpus

The [Google - Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs/data) competition dataset must contain:

- `train.csv`
- `sign_to_prediction_index_map.json`
- the parquet files referenced by the `path` column in `train.csv`

Only `left_hand` and `right_hand` landmarks are used. Face and pose landmarks are ignored.

### Target landmarks

The target data comes from the [ASL-preprocessing 7 output](https://www.kaggle.com/code/abdelrhmankaram/asl-preprocessing-7/output). It must expose a `landmarks` directory containing object-NPY files. Each valid file must provide:

- a string label;
- a `float32` landmark array with shape `T × 100 × 3`.

The default target layout expects right-hand landmarks at indices `0:21` and left-hand landmarks at `21:42`. Adjust `target_right_slice` and `target_left_slice` if the attached dataset uses a different documented layout. Object files are loaded through a restricted pickle allow-list and invalid files are recorded in a rejection manifest.

## Run on Kaggle ▶️

1. Import or upload [signlang_det_kaggle_training.ipynb](signlang_det_kaggle_training.ipynb) to Kaggle.
2. Attach the source corpus and target landmark dataset.
3. Select a **single T4 GPU** accelerator.
4. Review the configuration cell. The notebook validates the standard Competition and Notebook Output mount paths; set explicit paths when using a non-standard mount.
5. Set `SMOKE_TEST = True` for a short integration run, or keep it `False` for the documented training budget.
6. Choose **Run all** and retain the complete `/kaggle/working/signlang-det` output directory.

The notebook defaults to `num_workers = 0` and disables DataParallel for stable notebook restarts and efficient use of the lightweight encoder. Compatible checkpoints and an incomplete source cache are resumed automatically.

## Configuration ⚙️

The main settings are defined in the configuration cells near the top of the notebook.

| Setting | Default | Purpose |
|---|---:|---|
| `source_root` | standard Kaggle mount | Optional override for the Google ASL dataset root |
| `target_landmarks` | standard Kaggle mount | Optional override for the target `landmarks` directory |
| `work_dir` | `/kaggle/working/signlang-det` | Cache, checkpoint, metric, and export directory |
| `max_frames` | `64` | Maximum encoded sequence length |
| `min_frames` | `12` | Minimum accepted action length, matching the runtime segmenter |
| `max_input_frames` | `120` | Maximum accepted action length before rejection |
| `embedding_dim` | `128` | Frame and pooled feature dimension |
| `top_k` | `20` | Number of pooled candidates reranked with DTW |
| `dtw_window` | `12` | Constrained DTW window |
| `resume` | `True` | Restore compatible training state |
| `use_data_parallel` | `False` | Optional dual-GPU execution |

Production defaults use 30 epochs × 300 batches for representation training and 40 epochs × 200 batches for adaptation. These epochs contain a fixed number of P-by-K batches and do not represent full dataset passes.

## Outputs 📦

After verifying the exported encoder, the notebook removes caches, checkpoints, state files, and other intermediate data. `/kaggle/working/signlang-det` then contains only:

- `signlang_det_encoder.pt`, containing the final encoder contract, weights, and fingerprint;
- `figures/training_curves.png` and `figures/retrieval_summary.png`;
- `representation_training/metrics.csv` and `domain_adaptation/metrics.csv`;
- `representation_training/train.log` and `domain_adaptation/train.log`.

Kaggle CD converts the PT model to ONNX and RKNN. The three models, `model-manifest.json`, and original tagged notebook are uploaded separately; all remaining notebook outputs are placed in `notebook-output.tar.gz`.

Keep the encoder fingerprint and `hand168-temporal` preprocessing identifier with every dynamic prototype. A prototype produced by a different encoder or preprocessing contract must not be mixed with the exported model.

## Limitations and Calibration ⚠️

- Full training requires Kaggle-mounted datasets and a compatible CUDA runtime; repository-side checks cannot reproduce the production run without them.
- The current PyTorch CUDA build described by the workflow targets T4. CUDA availability alone does not guarantee architecture compatibility.
- The notebook does not provide action segmentation, a serving API, or a prototype management interface.
- Unknown-sign distance and margin thresholds cannot be inferred from training loss. They require separate calibration data containing known queries, difficult negatives, and unknown actions.
- Dataset licenses and usage restrictions remain the responsibility of their respective owners. This repository does not redistribute either training dataset.

## Acknowledgements 🙏

- Thanks to [hoyso48](https://www.kaggle.com/hoyso48) for the valuable insights shared at [https://www.kaggle.com/competitions/asl-signs/writeups/hoyeol-sohn-1st-place-solution-1dcnn-combined-with](https://www.kaggle.com/competitions/asl-signs/writeups/hoyeol-sohn-1st-place-solution-1dcnn-combined-with).
- Thanks to the organizers and contributors of the [Google - Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs/data) competition for making the source corpus available through Kaggle.
- Thanks to Microsoft for providing the [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) dataset.
- Thanks to [Abdelrhman Karam](https://www.kaggle.com/abdelrhmankaram) for publishing the [ASL-preprocessing 7 output](https://www.kaggle.com/code/abdelrhmankaram/asl-preprocessing-7/output) used for target-domain adaptation.

The datasets remain subject to their original licenses, competition rules, and usage terms.

## Project Files 🗂️

| File | Description |
|---|---|
| [signlang_det_kaggle_training.ipynb](signlang_det_kaggle_training.ipynb) | Self-contained Kaggle training and export workflow |
| [README.zh-CN.md](README.zh-CN.md) | Simplified Chinese documentation |
| [docs/KAGGLE_CD.md](docs/KAGGLE_CD.md) | Tag-driven Kaggle delivery setup and operations |
| [LICENSE](LICENSE) | Project license |

## License 📄

This project is released under the [Apache License 2.0](LICENSE).
