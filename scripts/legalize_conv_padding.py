#!/usr/bin/env python3

import sys
import onnx
from onnx import helper, numpy_helper, shape_inference
import numpy as np
from onnx import helper, numpy_helper, TensorProto


def get_attribute(node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            if attr.type == onnx.AttributeProto.STRING:
                return attr.s.decode("utf-8")
            if attr.type == onnx.AttributeProto.INT:
                return attr.i
    return default


def remove_attribute(node, name):
    keep = [a for a in node.attribute if a.name != name]
    del node.attribute[:]
    node.attribute.extend(keep)


def ensure_opset_minimum(model, min_version):
    """Bump the default-domain opset to at least min_version."""
    for oi in model.opset_import:
        if oi.domain == "" and oi.version < min_version:
            oi.version = min_version
            return
    # If no default domain entry exists, add one
    model.opset_import.append(helper.make_opsetid("", min_version))


def main(input_path, output_path):
    model = onnx.load(input_path)
    graph = model.graph

    # Ensure opset >= 11 so Pad uses tensor inputs
    ensure_opset_minimum(model, 11)

    new_nodes = []
    pad_count = 0

    for node in graph.node:
        if node.op_type != "Conv":
            new_nodes.append(node)
            continue

        pads = get_attribute(node, "pads", None)
        auto_pad = get_attribute(node, "auto_pad", None)

        if auto_pad in ["SAME_UPPER", "SAME_LOWER"]:
            kernel_shape = get_attribute(node, "kernel_shape", None)
            if kernel_shape is None:
                raise RuntimeError(f"Conv {node.name} has auto_pad but no kernel_shape")
            pad_h = kernel_shape[0] // 2
            pad_w = kernel_shape[1] // 2
            pads = [pad_h, pad_w, pad_h, pad_w]

        if pads is None:
            new_nodes.append(node)
            continue

        if all(p == 0 for p in pads):
            new_nodes.append(node)
            continue

        # ---- Build Pad node (opset 11+: pads as tensor input) ----
        input_name = node.input[0]
        padded_output = f"{input_name}_padded_{pad_count}"
        pad_node_name = f"Pad_{pad_count}"

        # ONNX Pad opset 11 pads format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        # For NCHW: [N_begin, C_begin, H_begin, W_begin, N_end, C_end, H_end, W_end]
        pad_values = np.array(
            [0, 0, pads[0], pads[1], 0, 0, pads[2], pads[3]],
            dtype=np.int64,
        )

        pads_initializer_name = f"{pad_node_name}_pads"
        pads_initializer = numpy_helper.from_array(pad_values, pads_initializer_name)
        graph.initializer.append(pads_initializer)
        graph.input.append(
            helper.make_tensor_value_info(pads_initializer_name, TensorProto.INT64, list(pad_values.shape))
        )

        # constant_value input (optional, defaults to 0 — but explicit is safer)
        const_val = np.array(0.0, dtype=np.float16)
        const_initializer_name = f"{pad_node_name}_value"
        const_initializer = numpy_helper.from_array(const_val, const_initializer_name)
        graph.initializer.append(const_initializer)
        graph.input.append(
            helper.make_tensor_value_info(const_initializer_name, TensorProto.FLOAT16, [])
        )

        pad_node = helper.make_node(
            "Pad",
            inputs=[input_name, pads_initializer_name, const_initializer_name],
            outputs=[padded_output],
            name=pad_node_name,
        )

        new_nodes.append(pad_node)

        # ---- Modify Conv node ----
        node.input[0] = padded_output
        remove_attribute(node, "pads")
        remove_attribute(node, "auto_pad")
        node.attribute.extend([helper.make_attribute("pads", [0, 0, 0, 0])])

        new_nodes.append(node)

        pad_count += 1
        print(f"Rewrote Conv '{node.name}' with explicit Pad")

    del graph.node[:]
    graph.node.extend(new_nodes)

    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)

    onnx.save(model, output_path)
    print(f"\nSaved rewritten model to: {output_path}")
    print(f"Total padded convs rewritten: {pad_count}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python legalize_conv_padding.py input.onnx output.onnx")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])