# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import warnings
from functools import partial

import mmcv
import numpy as np
import onnxruntime as rt
import torch
from mmcv.onnx import register_extra_symbolics
from mmcv.runner import load_checkpoint

from mmcls.models import build_classifier

#from onnx import optimizer
import onnx
from optimizer_scripts.tools import eliminating
from optimizer_scripts.tools import fusing
from optimizer_scripts.tools import replacing
from optimizer_scripts.tools import other
from optimizer_scripts.tools import combo
from optimizer_scripts.tools import special

torch.manual_seed(3)


def torch_exported_onnx_flow(m: onnx.ModelProto,
                             disable_fuse_bn=False) -> onnx.ModelProto:
    """Optimize the Pytorch exported onnx.

    Args:
        m (ModelProto): the input onnx model
        disable_fuse_bn (bool, optional): do not fuse BN into Conv. Defaults to False.

    Returns:
        ModelProto: the optimized onnx model
    """
    m = combo.preprocess(m, disable_fuse_bn)
    m = combo.pytorch_constant_folding(m)

    m = combo.common_optimization(m)

    m = combo.postprocess(m)

    return m


def _demo_mm_inputs(input_shape, num_classes):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)
    gt_labels = rng.randint(
        low=0, high=num_classes, size=(N, 1)).astype(np.uint8)
    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'gt_labels': torch.LongTensor(gt_labels),
    }
    return mm_inputs


def pytorch2onnx(model,
                 input_shape,
                 normalize_cfg,
                 opset_version=11,
                 dynamic_export=False,
                 show=False,
                 output_file='tmp.onnx',
                 do_simplify=False,
                 verify=False,
                 is_original_forward=False,
                 in_model_preprocess=False):
    """Export Pytorch model to ONNX model and verify the outputs are same
    between Pytorch and ONNX.

    Args:
        model (nn.Module): Pytorch model we want to export.
        input_shape (tuple): Use this input shape to construct
            the corresponding dummy input and execute the model.
        opset_version (int): The onnx op version. Default: 11.
        show (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the output ONNX model.
            Default: `tmp.onnx`.
        verify (bool): Whether compare the outputs between Pytorch and ONNX.
            Default: False.
    """
    model.cpu().eval()

    if hasattr(model.head, 'num_classes'):
        num_classes = model.head.num_classes
    # Some backbones use `num_classes=-1` to disable top classifier.
    elif getattr(model.backbone, 'num_classes', -1) > 0:
        num_classes = model.backbone.num_classes
    else:
        raise AttributeError('Cannot find "num_classes" in both head and '
                             'backbone, please check the config file.')

    mm_inputs = _demo_mm_inputs(input_shape, num_classes)

    imgs = mm_inputs.pop('imgs')
    img_list = [img[None, :] for img in imgs]

    # replace original forward function
    origin_forward = model.forward
    #model.forward = partial(model.forward, img_metas={}, return_loss=False)
    # is_original_forward or if postprocess
    if not is_original_forward:
        model.forward = partial(
            model.forward,
            img_metas={},
            return_loss=False,
            softmax=False,
            post_process=False)
    else:
        model.forward = partial(
            model.forward, img_metas={}, return_loss=False, post_process=False)
    register_extra_symbolics(opset_version)
    # support dynamic shape export
    if dynamic_export:
        dynamic_axes = {
            'input': {
                0: 'batch',
                2: 'width',
                3: 'height'
            },
            'probs': {
                0: 'batch'
            }
        }
    else:
        dynamic_axes = {}
    with torch.no_grad():
        torch.onnx.export(
            model, (img_list, ),
            output_file,
            input_names=['input'],
            output_names=['probs'],
            export_params=True,
            keep_initializers_as_inputs=True,
            dynamic_axes=dynamic_axes,
            verbose=show,
            opset_version=opset_version)
        print(f'Successfully exported ONNX model: {output_file}')

    print("#####  add BN for doing input data normalization  #####")
    import onnxsim
    import onnx
    from mmdet import digit_version

    min_required_version = '0.3.0'
    assert digit_version(onnxsim.__version__) >= digit_version(
        min_required_version
    ), f'Requires to install onnx-simplify>={min_required_version}'

    input_dic = {'input': img_list[0].detach().cpu().numpy()}
    model_opt, check_ok = onnxsim.simplify(
        output_file,
        input_data=input_dic,
        dynamic_input_shape=dynamic_export)
    if check_ok:
        onnx.save(model_opt, output_file)
        print(f'Successfully simplified ONNX model: {output_file}')
    else:
        warnings.warn('Failed to simplify ONNX model.')
    print(f'Successfully exported ONNX model: {output_file}')
    # print(normalize_cfg)
    m = onnx.load(output_file)
    #print(len(m.graph.input))
    m = torch_exported_onnx_flow(m, disable_fuse_bn=False)

    if len(m.graph.input) > 1:
        raise ValueError(
            " '--pixel-bias-value' and '--pixel-scale-value' only support one input node model currently"
        )

    if in_model_preprocess is True:
        mean = normalize_cfg['mean']
        std = normalize_cfg['std']

        i_n = m.graph.input[0]
        if i_n.type.tensor_type.shape.dim[1].dim_value != len(
                mean
        ) or i_n.type.tensor_type.shape.dim[1].dim_value != len(std):
            raise ValueError(
                "--pixel-bias-value (" + str(mean) +
                ") and --pixel-scale-value (" + str(std) +
                ") should be same as input dimension:" +
                str(i_n.type.tensor_type.shape.dim[1].dim_value))

        # add 128 for changing input range from 0~255 to -128~127 (int8) due to quantization due to quantization limitation
        normalize_bn_bias = [
            -1 * mean[0] / std[0] + 128.0 / std[0],
            -1 * mean[1] / std[1] + 128.0 / std[1],
            -1 * mean[2] / std[2] + 128.0 / std[2]
        ]
        normalize_bn_scale = [1 / std[0], 1 / std[1], 1 / std[2]]

        other.add_shift_scale_bn_after(m.graph, i_n.name,
                                       normalize_bn_bias,
                                       normalize_bn_scale)
        m = onnx.utils.polish_model(m)

    onnx_out = output_file[:-5] + '_kneron_optimized.onnx'
    onnx.save(m, onnx_out)
    print("exported success: ", onnx_out)

    return

    model.forward = origin_forward

    if do_simplify:
        import onnx
        import onnxsim

        if dynamic_axes:
            input_shape = (input_shape[0], input_shape[1], input_shape[2] * 2,
                           input_shape[3] * 2)
        else:
            input_shape = (input_shape[0], input_shape[1], input_shape[2],
                           input_shape[3])
        imgs = _demo_mm_inputs(input_shape, model.head.num_classes).pop('imgs')
        input_dic = {'input': imgs.detach().cpu().numpy()}
        input_shape_dic = {'input': list(input_shape)}

        model_opt, check_ok = onnxsim.simplify(
            output_file,
            input_shapes=input_shape_dic,
            input_data=input_dic,
            dynamic_input_shape=dynamic_export)
        if check_ok:
            onnx.save(model_opt, output_file)
            print(f'Successfully simplified ONNX model: {output_file}')
        else:
            print('Failed to simplify ONNX model.')
    if verify:
        # check by onnx
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # test the dynamic model
        if dynamic_export:
            dynamic_test_inputs = _demo_mm_inputs(
                (input_shape[0], input_shape[1], input_shape[2] * 2,
                 input_shape[3] * 2), model.head.num_classes)
            imgs = dynamic_test_inputs.pop('imgs')
            img_list = [img[None, :] for img in imgs]

        # check the numerical value
        # get pytorch output
        pytorch_result = model(img_list, img_metas={}, return_loss=False)[0]

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 1)
        sess = rt.InferenceSession(output_file)
        onnx_result = sess.run(
            None, {net_feed_input[0]: img_list[0].detach().numpy()})[0]
        if not np.allclose(pytorch_result, onnx_result):
            raise ValueError(
                'The outputs are different between Pytorch and ONNX')
        print('The outputs are same between Pytorch and ONNX')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMCls to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument(
        '--is-original-forward', action='store_true', help='use original forward')
    parser.add_argument(
        '--verify', action='store_true', help='verify the onnx model')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Whether to simplify onnx model.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    parser.add_argument(
        '--dynamic-export',
        action='store_true',
        help='Whether to export ONNX with dynamic input shape. \
            Defaults to False.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (
            1,
            3,
        ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None

    # build the model and load checkpoint
    classifier = build_classifier(cfg.model)

    if args.checkpoint:
        load_checkpoint(classifier, args.checkpoint, map_location='cpu')
    normalize_cfg = {'mean': cfg['NORM_MEAN'], 'std': cfg['NORM_STD']}
    # convert model to onnx file
    pytorch2onnx(
        classifier,
        input_shape,
        normalize_cfg,
        opset_version=args.opset_version,
        show=args.show,
        dynamic_export=args.dynamic_export,
        output_file=args.output_file,
        do_simplify=args.simplify,
        verify=args.verify,
        is_original_forward=args.is_original_forward)

    # Following strings of text style are from colorama package
    bright_style, reset_style = '\x1b[1m', '\x1b[0m'
    red_text, blue_text = '\x1b[31m', '\x1b[34m'
    white_background = '\x1b[107m'

    msg = white_background + bright_style + red_text
    msg += 'DeprecationWarning: This tool will be deprecated in future. '
    msg += blue_text + 'Welcome to use the unified model deployment toolbox '
    msg += 'MMDeploy: https://github.com/open-mmlab/mmdeploy'
    msg += reset_style
    warnings.warn(msg)
