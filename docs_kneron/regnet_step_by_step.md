# Step 0. Environment

## Prerequisites
- Python 3.6+
- PyTorch 1.3+ (We recommend you installing PyTorch using Conda following the [Official PyTorch Installation Instruction](https://pytorch.org/))
- (Optional) CUDA 9.2+ (If you installed PyTorch with cuda using Conda following the [Official PyTorch Installation Instruction](https://pytorch.org/), you can skip CUDA installation)
- (Optional, used to build from source) GCC 5+
- [mmcv-full](https://mmcv.readthedocs.io/en/latest/#installation) (Note: not `mmcv`!)

### Install kneron-mmclassification

1. We recommend you installing mmcv-full with pip:

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
    ```
    Please replace {cu_version} and {torch_version} in the url to your desired one. For example, to install the latest mmcv-full with CUDA 11.0 and PyTorch 1.7.0, use the following command:
    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
    ```    
    See [here](https://github.com/open-mmlab/mmcv#installation) for different versions of MMCV compatible to different PyTorch and CUDA versions.

2. Clone the Kneron-version MMClassification(kneron-mmclassification) repository.

    ```bash
    git clone https://github.com/kneron/kneron-mmclassification.git
    cd kneron-mmclassification
    ```

3. Install required python packages for building kneron-mmclassification and then install kneron-mmclassification.

    ```shell
    pip3 install -e .
    ```

# Step 1: Training models on standard datasets

MMClassification provides hundreds of existing and existing classification models in [Model Zoo](https://mmclassification.readthedocs.io/en/latest/model_zoo.html), and supports several standard datasets like ImageNet, MNIST, CIFAR10, CIFAR100, etc. This note demonstrates how to perform common classification tasks with these existing models and standard datasets, including:

- Use existing models to inference on given images.
- Test existing models on standard datasets.
- Train models on standard datasets.

## Train models on standard datasets

MMClassification also provides out-of-the-box tools for training classification models.
This section will show how to train models (under [configs](https://github.com/open-mmlab/mmclassification/tree/master/configs)) on standard datasets i.e. COCO.

**Important**: You might need to modify the [config file](https://github.com/open-mmlab/mmclassification/blob/master/docs/en/tutorials/config.md) according your GPUs resource (such as "samples_per_gpu","workers_per_gpu" ...etc due to your GPUs RAM limitation).
The default learning rate in config files is for 8 GPUs and 2 img/gpu (batch size = 8\*2 = 16).

### Step 1-1: Prepare datasets

Public datasets such as ImageNet or mnist.
We suggest that you download and extract the dataset to somewhere outside the project directory and symlink (`ln`) the dataset root to `kneron-mmclassification/data`(`ln -s realpath/to/dataset kneron-mmclassification/data`), as shown below:

```plain
kneron-mmclassification
├── mmcls
├── tools
├── configs
├── data
│   ├── imagenet
│   │   ├── meta
│   │   ├── train
│   │   ├── val
│   ├── cifar
│   │   ├── cifar-10-batches-py
│   ├── mnist
│   │   ├── train-images-idx3-ubyte
│   │   ├── train-labels-idx1-ubyte
│   │   ├── t10k-images-idx3-ubyte
│   │   ├── t10k-labels-idx1-ubyte
...
```

It's recommended to *symlink* the dataset folder to mmdetection folder. However, if you place your dataset folder at different place and do not want to symlink, you have to change the corresponding paths in config files (absolute path is recommended).

### Step 1-2: Training Example with RegNet:

[Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)

We only need the configuration file (which is provided in `configs/regnet/regnetx-400mf_8xb128_in1k.py`) to train RegNet:
```python
python tools/train.py configs/regnet/regnetx-400mf_8xb128_in1k.py
```
* (Note) you might need to create a folder name 'work_dir' in MMClassification root folder because we set 'work_dir' as default folder in 'regnetx-400mf_8xb128_in1k.py'
* (Note 2) The whole training process might take several days, depending on your computational resource (number of GPUs, etc). If you just want to take a quick look at the deployment flow, we suggest that you download our trained model so you can skip the training process:
```bash
mkdir work_dirs
cd work_dirs
wget https://github.com/kneron/Model_Zoo/mmclassification/regnet/latest.zip
unzip latest.zip
cd ..
```
* (Note 3) This is a "training from scratch" tutorial, which might need lots of time and gpu resource. If you want to train a model on your custom dataset, it is recommended that you read [finetune.md](https://github.com/open-mmlab/mmclassification/blob/master/docs/en/tutorials/finetune.md), [customize_dataset.md](https://github.com/open-mmlab/mmclassification/blob/master/docs/en/tutorials/new_dataset.md), [colab tutorial: Fine-tune a model](https://github.com/open-mmlab/mmclassification/blob/master/docs/en/tutorials/MMClassification_python.ipynb)

# Step 2: Test trained model
`tools/test_kneron.py` is a script that generates inference results from test set with our pytorch model and evaluates the results to see if our pytorch model is well trained (if `--eval` argument is given). Note that it's always good to evluate our pytorch model before deploying it.

```python
python tools/test_kneron.py \
    configs/regnet/regnetx-400mf_8xb128_in1k_normalize.py \
    work_dirs/regnetx-400mf_8xb128_in1k_normalize/latest.pth \
    --metrics accuracy \
```
* `configs/regnet/regnetx-400mf_8xb128_in1k_normalize.py` is your Regnet training config
* `work_dirs/regnetx-400mf_8xb128_in1k_normalize/latest.pth` is your trained Regnet model

The expected result of the command above will be something similar to the following text (the numbers may slightly differ):
```
...
accuracy_top-1 : 70.75

accuracy_top-5 : 90.01
...
```

# Step 3: Export ONNX and Verify
### Step 3-1: Export ONNX:
`tools/deployment/pytorch2onnx_kneron.py` is a script provided by Kneron to help user to convert our trained pth model to onnx:
```python
python tools/deployment/pytorch2onnx_kneron.py \
    configs/regnet/regnetx-400mf_8xb128_in1k_normalize.py \
    --checkpoint work_dirs/regnetx-400mf_8xb128_in1k_normalize/latest.pth \
    --output-file work_dirs/regnetx-400mf_8xb128_in1k_normalize/latest.onnx \
```
* `configs/regnet/regnetx-400mf_8xb128_in1k_normalize.py` is your Regnet training config
* `work_dirs/regnetx-400mf_8xb128_in1k_normalize/latest.pth` is your trained Regnet model

The output onnx should be the same name as 'work_dirs/regnetx-400mf_8xb128_in1k_normalize/latest.pth' with '.onnx' post-fix in the same folder.

### Step 3-2: Verify ONNX:
tools/test_kneron.py is a script provided by kneron-mmclassification to help users to verify if our exported ONNX generates similar outputs with what our PyTorch model does:
```python
python tools/test_kneron.py \
    configs/regnet/regnetx-400mf_8xb128_in1k_normalize.py \
    work_dirs/regnetx-400mf_8xb128_in1k_normalize/latest.onnx \
    --metrics accuracy \
```
The expected result of the command above should be something similar to the following text (the numbers may slightly differ):
```
...
accuracy_top-1 : 70.74

accuracy_top-5 : 90.01
...
```
Note that the ONNX results may differ from the PyTorch results due to some implementation differences between PyTorch and ONNXRuntime.
# Step 4: Convert onnx to [NEF](http://doc.kneron.com/docs/#toolchain/manual/#5-nef-workflow) model for Kneron platform

### Step 4-1: Install Kneron toolchain docker:
* check [document](http://doc.kneron.com/docs/#toolchain/manual/#1-installation)

### Step 4-2: Mout Kneron toolchain docker
* Mount a folder (e.g. '/mnt/hgfs/Competition') to toolchain docker container as `/data1`. The converted onnx in Step 3 should be put here. All the toolchain operation should happen in this folder.
```
sudo docker run --rm -it -v /mnt/hgfs/Competition:/data1 kneron/toolchain:latest
```

### Step 4-3: Import KTC and required lib in python shell
* Here we demonstrate how to go through all Kneron Toolchain (KTC) flow through Python API:
```python
import ktc
import numpy as np
import os
import onnx
from PIL import Image
```

### Step 4-4: Optimize the onnx model
```python
onnx_path = '/data1/latest.onnx'
m = onnx.load(onnx_path)
m = ktc.onnx_optimizer.onnx2onnx_flow(m)
onnx.save(m,'latest.opt.onnx')
```

### Step 4-5: Configure and load data necessary for ktc, and check if onnx is ok for toolchain
```python
# npu (only) performance simulation
km = ktc.ModelConfig((&)model_id_on_public_field, "0001", "720", onnx_model=m)
eval_result = km.evaluate()
print("\nNpu performance evaluation result:\n" + str(eval_result))
```

### Step 4-6: Quantize the onnx model
We [random sampled 50 images from voc dataset](https://www.kneron.com/forum/uploads/112/SMZ3HLBK3DXJ.7z) (50 images) as quantization data , we have to
1. Download the data
2. Uncompression the data as folder named `voc_data50"`
3. Put the `voc_data50` into docker mounted folder (the path in docker container should be `/data1/voc_data50`)

The following script will do some preprocess(should be the same as training code) on our quantization data, and put it in a list:
```python
import os
from os import walk

img_list = []
for (dirpath, dirnames, filenames) in walk("/data1/voc_data50"):
    for f in filenames:
        fullpath = os.path.join(dirpath, f)

        image = Image.open(fullpath)
        image = image.convert("RGB")
        image = Image.fromarray(np.array(image)[...,::-1])
        img_data = np.array(image.resize((224, 224), Image.BILINEAR)) / 256 - 0.5
        print(fullpath)
        img_list.append(img_data)
```

Then perform quantization. The BIE model will be generated at `/data1/output.bie`.

```python
# fixed-point analysis
bie_model_path = km.analysis({"input": img_list})
print("\nFixed-point analysis done. Save bie model to '" + str(bie_model_path) + "'")
```

### Step 4-7: Compile
The final step is compile the BIE model into an NEF model.
```python
# compile
nef_model_path = ktc.compile([km])
print("\nCompile done. Save Nef file to '" + str(nef_model_path) + "'")
```

You can find the NEF file at `/data1/batch_compile/models_720.nef`. `models_720.nef` is the final compiled model.

# Step 5: Run [NEF](http://doc.kneron.com/docs/#toolchain/manual/#5-nef-workflow) model on [KL720 USB accelerator](https://www.kneo.ai/products/hardwares/HW2020122500000007/1)

* N/A

# Step 6 (For Kneron AI Competition 2022): Run [NEF](http://doc.kneron.com/docs/#toolchain/manual/#5-nef-workflow) model on [KL720 USB accelerator](https://www.kneo.ai/products/hardwares/HW2020122500000007/1)

[WARNING] Don't do this step in toolchain docker enviroment mentioned in Step 4

Recommend you read [Kneron PLUS official document](http://doc.kneron.com/docs/#plus_python/#_top) first.

### Step 6-1: Download and Install PLUS python library(.whl)
* Go to [Kneron Education Center](https://www.kneron.com/tw/support/education-center/)
* Scroll down to `OpenMMLab Kneron Edition` table
* Select `Kneron Plus v1.3.0 (pre-built python library, firmware)`
* Select `python library`
* Select Your OS version (Ubuntu, Windows, MacOS, Raspberry pi)
* Download `KneronPLUS-1.3.0-py3-none-any_{your_os}.whl`
* Unzip downloaded `KneronPLUS-1.3.0-py3-none-any.whl.zip`
* `pip install KneronPLUS-1.3.0-py3-none-any.whl`

### Step 6-2: Download and upgrade KL720 USB accelerator firmware
* Go to [Kneron education center](https://www.kneron.com/tw/support/education-center/)
* Scroll down to `OpenMMLab Kneron Edition table`
* Select `Kneron Plus v1.3.0 (pre-built python library, firmware)`
* Select `firmware`
* Download `kl720_frimware.zip (fw_ncpu.bin、fw_scpu.bin)`
* unzip downloaded `kl720_frimware.zip`
* upgrade KL720 USB accelerator firmware(fw_ncpu.bin、fw_scpu.bin) by following [document](http://doc.kneron.com/docs/#plus_python/getting_start/), `Sec. 2. Update AI Device to KDP2 Firmware`, `Sec. 2.2 KL720`

### Step 6-3: Download RegNetX example code
* Go to [Kneron education center](https://www.kneron.com/tw/support/education-center/)
* Scroll down to **OpenMMLab Kneron Edition** table
* Select **kneron-mmclassification**
* Select **RegNetX**
* Download **regnetx_plus_demo.zip**
* unzip downloaded **regnetx_plus_demo**

### Step 6-4: Test enviroment is ready (require [KL720 USB accelerator](https://www.kneo.ai/products/hardwares/HW2020122500000007/1))
In `regnetx_plus_demo`, we provide a RegNetX-Cls example model and image for quick test. 
* Plug in [KL720 USB accelerator](https://www.kneo.ai/products/hardwares/HW2020122500000007/1) into your computer USB port
* Go to the regnetx_plus_demo folder
```bash
cd /PATH/TO/regnetx_plus_demo
```

* Install required python libraries
```bash
pip install -r requirements.txt
```

* Run example on [KL720 USB accelerator](https://www.kneo.ai/products/hardwares/HW2020122500000007/1)
```python
python KL720DemoGenericInferenceRegNetX_BypassHwPreProc.py -nef ./example_RegNetX_720.nef -img shark.jpg
```

Then you can see the inference result shown on your console window.
The expected result of the command above will be something similar to the following text:
```plain
...
[Connect Device]
 - Success
[Set Device Timeout]
 - Success
[Upload Model]
 - Success
======== NEF Info =========

Toolchain ver= kneron/toolchain:v0.17.2

Schema ver   = v0.9.1

============================
[Read Image]
 - Success
[Starting Inference Work]
 - Starting inference loop 1 times
 - .
[Retrieve Inference Node Output ]
 - Success
[Result]
 - Top-1 class id : 2  (imagenet 1000 classes idx)
...
```

### Step 6-4: Run your NEF model and your image on [KL720 USB accelerator](https://www.kneo.ai/products/hardwares/HW2020122500000007/1)
Use the same script in previous step, but now we change the input NEF model path and image to yours
```bash
python KL720DemoGenericInferenceRegNetX_BypassHwPreProc.py -img /PATH/TO/YOUR_IMAGE.bmp -nef /PATH/TO/YOUR/720_NEF_MODEL.nef
```
