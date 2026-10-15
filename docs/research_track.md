# Annotation-Efficient Primate Detection

> **Current research track:** development has moved to the integrity-first
> **WildPrimate-DG** cross-site detector protocol. See
> [`research/WILDPRIMATE_DG_PROTOCOL.md`](research/WILDPRIMATE_DG_PROTOCOL.md),
> [`research/STATUS.md`](research/STATUS.md) for the live gate status, and the
> frozen generated split in
> [`research/protocols/nkhotakota_v2_split_manifest.json`](research/protocols/nkhotakota_v2_split_manifest.json).
> Earlier FastKAN and active-learning results are retained as prior evidence,
> not presented as state of the art.

> Developing efficient computer-vision detectors for wildlife and primate monitoring under limited annotation budgets — built on the **YOLO-KAN** architecture.

## Overview

Camera-trap imagery is one of the primary tools wildlife conservationists use to estimate primate populations, detect poaching activity, and monitor biodiversity. Manually reviewing millions of images is infeasible, so automated detection is essential. However, building accurate detectors is bottlenecked by the cost of labeling.

This project investigates **annotation-efficient object detection** for wildlife imagery. The current technical foundation is **YOLO-KAN** — *"YOLO-KAN: Exploring the Adaptability of Kolmorov-Arnold Networks in the YOLO Model"* — which replaces selected convolutional blocks in the YOLO backbone with Kolmogorov-Arnold Network (KAN) layers whose activation functions are learned rather than fixed. KAN's stronger function-fitting capacity per parameter is the property we exploit in the low-label regime.

The repository contains (1) a faithful reproduction of YOLO-KAN's checkpoint evaluation on COCO, and (2) a research plan and tooling for the follow-on primate-detection study.

## Research Motivation

Wildlife computer-vision datasets suffer from three structural problems that standard detection pipelines are not designed for:

- **Expensive annotation.** Expert primatologists must hand-draw bounding boxes over millions of field images; labels are the dominant cost, not compute.
- **Limited expert labels.** Many species are rare, so even with effort the labeled set stays small and class-imbalanced.
- **Domain differences.** Camera-trap images differ sharply from the web/COCO domain models are pretrained on — night-vision, motion blur, dense foliage, partial occlusion, and severe class imbalance.

The central research question is whether KAN's learnable-activation design **reduces the number of labeled images required** to reach a given detection quality — the regime where its theoretical parameter-efficiency advantage should matter, and which the original YOLO-KAN paper never tested.

## Current Achievement — YOLO-KAN Reproduction

The YOLO-KAN source repository was reproduced at the **checkpoint-evaluation level** on COCO val2017:

- ✅ Environment setup (Python 3.10, PyTorch cu128 for Blackwell `sm_120`)
- ✅ YOLO-KAN source reproduction (git submodule, pinned commit `0f5e5a1`)
- ✅ Checkpoint-compatibility resolution (Nov-2024 checkpoints load & forward via a compatibility shim)
- ✅ COCO val2017 validation pipeline (`pycocotools`, `save-json`)
- ✅ Evaluation on NVIDIA RTX PRO 6000 Blackwell

**Reproduction status: PARTIALLY VERIFIED.** This is a checkpoint-evaluation reproduction (provided weights evaluated on val2017), not a from-scratch training reproduction. The pipeline, environment, checkpoint compatibility, dataset, and the paper's *directional* precision claim (KAN-2 > baseline) are all verified; exact numeric equality with the paper is neither achievable (no per-model paper numbers; the repo validates on `train2017` while we use `val2017`) nor expected. Full details: [`reproduction/results/core_checkpoint_results.md`](reproduction/results/core_checkpoint_results.md).

## Experimental Environment

**Hardware**
- NVIDIA RTX PRO 6000 Blackwell (Blackwell architecture, compute capability `sm_120`, ~96 GB VRAM)
- Evaluation runs on a single GPU

**Software**
- Python 3.10.20
- PyTorch 2.11.0+cu128 (CUDA 12.8 runtime — required for Blackwell `sm_120`)
- ultralytics 8.3.3 (YOLO-KAN modified fork, editable install)
- pycocotools 2.0.11 (COCO evaluation)

Full locked environment: [`reproduction/configs/requirements-lock.txt`](reproduction/configs/requirements-lock.txt). Setup rationale: [`reproduction/ENVIRONMENT.md`](reproduction/ENVIRONMENT.md).

## Results

Checkpoint evaluation on **COCO val2017** (imgsz=640, batch=64, pycocotools). Only verified numbers are reported.

| Model | Checkpoint | Precision | Recall | mAP50 | mAP50-95 | Params | GFLOPs | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| YOLO11 (baseline) | `11-official.pt` | 0.652 | 0.493 | 0.541 | 0.386 | 2.62 M | 6.5 | ✅ Verified |
| YOLO-KAN-1 | `11-1-7.pt` | 0.647 | 0.504 | 0.545 | 0.389 | 3.69 M | 7.2 | ✅ Verified |
| YOLO-KAN-2 | `11-2-5.pt` | 0.659 | 0.492 | 0.544 | 0.389 | 3.46 M | 7.3 | ◑ Partially verified |

**KAN-vs-baseline deltas (val2017):**

| Model | ΔPrecision | ΔRecall | ΔmAP50 | ΔmAP50-95 |
|---|---:|---:|---:|---:|
| YOLO-KAN-1 | −0.005 | +0.011 | +0.004 | +0.003 |
| YOLO-KAN-2 | **+0.007** | −0.001 | +0.003 | +0.003 |

The paper's abstract claims *"one of the proposed YOLO-KAN models achieves a 1.84% improvement in precision over YOLOv11n on COCO."* The **direction** of this claim is confirmed on val2017 (YOLO-KAN-2 precision 0.659 > baseline 0.652). The exact **magnitude** (+1.84%) is inconclusive here because (a) the paper validates on `train2017` (a repo bug) while we use `val2017`, and (b) only 2 of the 8 KAN variants were evaluated — the +1.84% model may be among the unevaluated variants. This is reported transparently rather than silently overstated.

Per-checkpoint raw evidence logs: [`reproduction/results/val/`](reproduction/results/val/).

## Repository Structure

```
AE_Primate_Detection/
├── README.md                      # this file
├── .gitignore
├── .gitmodules                    # YOLO-KAN submodule
│
├── YOLO-KAN/                      # submodule: shankswhite/YOLO-KAN @ 0f5e5a1
│                                  # (the reproduced upstream model code)
│
├── reproduction/                  # YOLO-KAN reproduction workspace
│   ├── README.md                  # workspace orientation
│   ├── STATUS.md, BLOCKERS.md, RUNBOOK.md
│   ├── ENVIRONMENT.md, EXPERIMENT_MATRIX.md, RESULTS.md
│   ├── PAPER_METHOD.md, CHECKPOINT_AUDIT.md, DATASET_AUDIT.md
│   ├── REPOSITORY_AUDIT.md, RESEARCH_IDEAS.md
│   ├── PROFESSOR_UPDATE.md, RESEARCH_MATE_UPDATE.md
│   ├── NEXT_RESEARCH_PLAN.md
│   ├── configs/                   # coco_local.yaml, requirements-lock.txt
│   ├── scripts/                   # env check, build, checkpoint, eval, train drivers
│   ├── patches/                   # checkpoint_compatibility.patch
│   └── results/                   # core_checkpoint_results.{md,csv}, val/ evidence
│
├── research/                      # follow-on primate-detection research plan
│   ├── final_research_plan.md     # hypothesis, baselines, ablations, metrics
│   ├── final_report.md            # one-screen summary
│   ├── gap_analysis.md, contribution_candidates.md
│   ├── dataset_strategy.md, implementation_plan.md
│   └── reproduction_analysis/     # deep architecture & paper-claim reviews
│
├── experiments/                   # follow-on experiment tooling (WIP)
│   ├── README.md
│   └── tools/                     # dataset prep, fine-tune, audit scripts
│
└── communication/                 # milestone updates (professor & research-mate)
    ├── professor_updates.md
    └── research_mate_updates.md
```

Datasets, checkpoints, weights, caches, and logs are **not** tracked — see `.gitignore`. They remain on the local workstation.

## Reproducing the Evaluation

```bash
# 1. Clone with the YOLO-KAN submodule
git clone --recurse-submodules https://github.com/mobashirsifat123/AE_Primate_Detection.git
cd AE_Primate_Detection
git submodule update --init --recursive   # if already cloned without --recurse

# 2. Create the environment (Python 3.10, PyTorch cu128 for Blackwell)
#    See reproduction/ENVIRONMENT.md and reproduction/scripts/install_env.sh

# 3. Evaluate a checkpoint on COCO val2017
#    See reproduction/RUNBOOK.md and reproduction/scripts/09_evaluate_all.sh
```

Full step-by-step instructions: [`reproduction/RUNBOOK.md`](reproduction/RUNBOOK.md). Method/audit notes: [`reproduction/PAPER_METHOD.md`](reproduction/PAPER_METHOD.md), [`reproduction/CHECKPOINT_AUDIT.md`](reproduction/CHECKPOINT_AUDIT.md), [`reproduction/DATASET_AUDIT.md`](reproduction/DATASET_AUDIT.md).

## Follow-on Research

The reproduction surfaces a clear research gap: the original YOLO-KAN work never tested KAN in the **low-annotation regime** where its parameter-efficiency should matter most. The planned contribution is a parameter-controlled test of KAN sample-complexity on primate camera-trap detection (Nkhotakota primary, WCS for generalization), with parameter-matched controls and 3 seeds so that a positive *or* negative result is publishable. Plan: [`research/final_research_plan.md`](research/final_research_plan.md).
