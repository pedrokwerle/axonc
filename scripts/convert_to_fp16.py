#!/usr/bin/env python3
"""
Convert an FP32 ONNX model to FP16 — ALL nodes and tensors.

This does an aggressive conversion: every float32 initializer, I/O,
intermediate value_info, and constant node is cast to float16.
No op types are exempted.

Install dependencies:
    pip install onnx onnxconverter-common onnxruntime numpy

Usage:
    python convert_to_fp16.py squeezenet1.0-12.onnx squeezenet1.0-12-fp16.onnx
    python convert_to_fp16.py squeezenet1.0-12.onnx squeezenet1.0-12-fp16.onnx --verify
"""

import argparse
import sys
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
from onnxconverter_common import float16


def convert_fp32_to_fp16(input_path: str, output_path: str):
    """Convert ALL FP32 tensors and nodes in an ONNX model to FP16."""
    print(f"Loading: {input_path}")
    model_fp32 = onnx.load(input_path)

    # --- Pass 1: onnxconverter-common with no op block list ---
    print("Pass 1: onnxconverter-common (no op exclusions)...")
    model_fp16 = float16.convert_float_to_float16(
        model_fp32,
        min_positive_val=1e-7,
        max_finite_val=65504.0,
        keep_io_types=False,
        disable_shape_infer=False,
        op_block_list=[],           # Don't skip any op types
        node_block_list=[],         # Don't skip any specific nodes
    )

    # --- Pass 2: Force-convert anything the library missed ---
    print("Pass 2: Force-converting remaining FP32 tensors...")
    graph = model_fp16.graph

    # 2a: Initializers
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init).astype(np.float16)
            new_init = numpy_helper.from_array(arr, name=init.name)
            init.CopyFrom(new_init)
            print(f"  Fixed initializer: {init.name}")

    # 2b: Graph inputs
    for inp in graph.input:
        if inp.type.tensor_type.elem_type == TensorProto.FLOAT:
            inp.type.tensor_type.elem_type = TensorProto.FLOAT16
            print(f"  Fixed input: {inp.name}")

    # 2c: Graph outputs
    for out in graph.output:
        if out.type.tensor_type.elem_type == TensorProto.FLOAT:
            out.type.tensor_type.elem_type = TensorProto.FLOAT16
            print(f"  Fixed output: {out.name}")

    # 2d: Intermediate value_info
    for vi in graph.value_info:
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT16
            print(f"  Fixed value_info: {vi.name}")

    # 2e: Constant nodes with float tensors
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT:
                    arr = numpy_helper.to_array(attr.t).astype(np.float16)
                    new_t = numpy_helper.from_array(arr)
                    attr.t.CopyFrom(new_t)
                    print(f"  Fixed Constant node: {node.output[0]}")

    # 2f: Remove any Cast(fp16->fp32) or Cast(fp32->fp16) nodes the converter inserted
    nodes_to_remove = []
    for node in graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to":
                    if attr.i == TensorProto.FLOAT:
                        # Cast to fp32 — change to fp16
                        attr.i = TensorProto.FLOAT16
                        print(f"  Changed Cast->FP32 to Cast->FP16: {node.output[0]}")
                    if attr.i == TensorProto.FLOAT16:
                        # Now fp16->fp16, mark for removal
                        nodes_to_remove.append(node)

    # Remove redundant fp16->fp16 casts
    for node in nodes_to_remove:
        if len(node.input) == 1 and len(node.output) == 1:
            out_name = node.output[0]
            in_name = node.input[0]
            for other_node in graph.node:
                for i, inp in enumerate(other_node.input):
                    if inp == out_name:
                        other_node.input[i] = in_name
            for out in graph.output:
                if out.name == out_name:
                    out.name = in_name
            graph.node.remove(node)
            print(f"  Removed redundant Cast: {out_name} -> {in_name}")

    # --- Save ---
    onnx.save(model_fp16, output_path)
    print(f"\nSaved FP16 model: {output_path}")

    # --- Summary ---
    n_fp16 = 0
    n_fp32 = 0
    n_other = 0
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            n_fp16 += 1
        elif init.data_type == TensorProto.FLOAT:
            n_fp32 += 1
        else:
            n_other += 1

    print(f"\nSummary:")
    print(f"  FP16 initializers: {n_fp16}")
    if n_fp32 > 0:
        print(f"  WARNING: {n_fp32} initializers still FP32!")
    if n_other > 0:
        print(f"  Non-float initializers: {n_other} (int64 shape tensors etc.)")

    all_fp16 = True
    for inp in graph.input:
        dt = inp.type.tensor_type.elem_type
        name = onnx.TensorProto.DataType.Name(dt)
        print(f"  Input  '{inp.name}': {name}")
        if dt != TensorProto.FLOAT16:
            all_fp16 = False
    for out in graph.output:
        dt = out.type.tensor_type.elem_type
        name = onnx.TensorProto.DataType.Name(dt)
        print(f"  Output '{out.name}': {name}")
        if dt != TensorProto.FLOAT16:
            all_fp16 = False

    n_casts = sum(1 for n in graph.node if n.op_type == "Cast")
    if n_casts > 0:
        print(f"  WARNING: {n_casts} Cast nodes remain in graph")

    if all_fp16 and n_fp32 == 0:
        print("\n  ALL tensors and I/O are FP16.")
    else:
        print("\n  WARNING: Some tensors are not FP16. Check above.")


def verify_accuracy(fp32_path: str, fp16_path: str, n_samples: int = 5):
    """Run both models on random inputs and compare Top-1/Top-5 predictions."""
    import onnxruntime as ort

    print(f"\nVerifying accuracy ({n_samples} random samples)...")

    sess_fp32 = ort.InferenceSession(fp32_path)
    sess_fp16 = ort.InferenceSession(fp16_path)

    inp_name_32 = sess_fp32.get_inputs()[0].name
    inp_name_16 = sess_fp16.get_inputs()[0].name

    top1_match = 0
    top5_match = 0
    max_abs_diff = 0.0

    for i in range(n_samples):
        x_fp32 = np.random.randn(1, 3, 224, 224).astype(np.float32) * 0.5

        out_fp32 = sess_fp32.run(None, {inp_name_32: x_fp32})[0]

        x_fp16 = x_fp32.astype(np.float16)
        out_fp16 = sess_fp16.run(None, {inp_name_16: x_fp16})[0]

        out_fp16_as_32 = out_fp16.astype(np.float32)
        diff = np.abs(out_fp32 - out_fp16_as_32).max()
        max_abs_diff = max(max_abs_diff, diff)

        pred32 = np.argsort(out_fp32.flatten())[::-1]
        pred16 = np.argsort(out_fp16_as_32.flatten())[::-1]

        if pred32[0] == pred16[0]:
            top1_match += 1
        if pred32[0] in pred16[:5]:
            top5_match += 1

        print(f"  Sample {i}: FP32 top1={pred32[0]}, FP16 top1={pred16[0]}, "
              f"max_diff={diff:.6f}")

    print(f"\n  Top-1 agreement: {top1_match}/{n_samples}")
    print(f"  Top-5 agreement: {top5_match}/{n_samples}")
    print(f"  Max absolute output difference: {max_abs_diff:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX FP32 model to FP16 (all nodes)")
    parser.add_argument("input", help="Input FP32 ONNX model path")
    parser.add_argument("output", help="Output FP16 ONNX model path")
    parser.add_argument("--verify", action="store_true",
                        help="Run accuracy comparison between FP32 and FP16")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of random samples for verification (default: 5)")
    args = parser.parse_args()

    convert_fp32_to_fp16(args.input, args.output)

    if args.verify:
        verify_accuracy(args.input, args.output, args.n_samples)