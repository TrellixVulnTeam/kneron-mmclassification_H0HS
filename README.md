# Kneron AI Training/Deployment Platform (mmClassification-based)


## Introduction

---
  [kneron-mmclassification](https://github.com/kneron/kneron-mmclassification) is a platform built upon the well-known [mmclassification](https://github.com/open-mmlab/mmclassification) for classification. We encourage that you may start with the [RegNet: Step-by-Step](https://github.com/kneron/kneron-mmclassification/blob/main/docs_kneron/regnet_step_by_step.md) to build basic knowledge of Kneron-Edition mmclassification, and read [mmclassification docs](https://mmclassification.readthedocs.io/en/latest/) for detailed mmclassification usage.  

  In this repository, we provide an end-to-end training/deployment flow to realize on Kneron's AI accelerators:

  1. **Training/Evalulation:**
      - Modified model configuration and verified for Kneron hardware platform
      - Please see [Overview of Benchmark and Model Zoo](#Overview-of-Benchmark-and-Model-Zoo) for the model list
  2. **Converting to onnx:**
      - pytorch2onnx_kneron.py (beta)
      - Export *optimized* and *Kneron-toolchain supported* onnx
          - Automatically modify model for arbitrary data normalization preprocess
  3. **General Evaluation**
      - test_kneron.py (beta)
      - Evaluate the model with pytorch checkpoint, onnx, and kneron-nef
  4. **Testing**
      - inference_kn (beta)
      - Verify the converted [NEF](http://doc.kneron.com/docs/#toolchain/manual/#5-nef-workflow) model on Kneron USB accelerator with this API
  5. **Converting Kneron-NEF:** (toolchain feature)
     - Convert the trained pytorch model to [Kneron-NEF](http://doc.kneron.com/docs/#toolchain/manual/#5-nef-workflow) model, which could be used on Kneron hardware platform.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

N/A

## Overview of Benchmark and Kneron Model Zoo
| Backbone  | size   | Mem (GB) |   Top-1 (%) |   Top-5 (%) | Config | Download |
|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|:------:|
| RegNet | 224 |   0.1    |  70.75  |   90.0  | [config](https://github.com/kneron/kneron-mmclassification/tree/main/configs/regnet/regnetx-400mf_8xb128_in1k_normalize.py) |[model](https://github.com/kneron/Model_Zoo/raw/main/mmclassification/regnet/latest.zip)

## Installation
- Please refer to [RegNet: Step-by-Step, Step 0. Environment](https://github.com/kneron/kneron-mmclassification/blob/main/docs_kneron/regnet_step_by_step.md) for installation.
- Please refer to [Kneron PLUS - Python: Installation](http://doc.kneron.com/docs/#plus_python/introduction/install_dependency/) for the environment setup for Kneron USB accelerator.

## Getting Started
### Tutorial - Kneron Edition
- [RegNet: Step-by-Step](https://github.com/kneron/kneron-mmclassification/blob/main/docs_kneron/regnet_step_by_step.md): A tutorial for users to get started easily. To see detailed documents, please see below.

### Documents - Kneron Edition
- [Kneron ONNX Export] (under development)
- [Kneron Inference] (under development)
- [Kneron toolchain step-by-step (YOLOv3)](http://doc.kneron.com/docs/#toolchain/yolo_example/)
- [Kneron toolchain manual](http://doc.kneron.com/docs/#toolchain/manual/#0-overview)

### Original mmclassification Documents
- [original mmclassification getting started](https://github.com/open-mmlab/mmclassification/blob/master/docs/en/getting_started.md): It is recommended to read original mmclassification getting started documents for other mmclassification operations.
- [original mmclassification readthedoc](https://mmclassification.readthedocs.io/en/latest/): Original mmclassification documents.

## Contributing
---
[kneron-mmclassification](https://github.com/kneron/kneron-mmclassification) a platform built upon [OpenMMLab-mmclassification](https://github.com/open-mmlab/mmclassification)

- For issues regarding to the original [mmclassification](https://github.com/open-mmlab/mmclassification):
We appreciate all contributions to improve [OpenMMLab-mmclassification](https://github.com/open-mmlab/mmclassification). Welcome community users to participate in these projects. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

- For issues regarding to this repository [kneron-mmclassification](https://github.com/kneron/kneron-mmclassification): Welcome to leave the comment or submit the pull request here to improve kneron-mmclassification


## Related Projects
- kneron-mmdetection: Kneron training/deployment platform on [OpenMMLab -mmDetection](https://github.com/open-mmlab/mmdetection) detection toolbox
