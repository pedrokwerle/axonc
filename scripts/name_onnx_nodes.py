#!/usr/bin/env python3
"""Add names to unnamed ONNX nodes based on output tensor names."""
import onnx
import sys

def name_nodes(model_path, output_path):
    model = onnx.load(model_path)
    
    for node in model.graph.node:
        if not node.name and node.output:
            # Use output tensor name, sanitized for C identifiers
            node.name = node.output[0].replace('/', '_').replace(':', '_')
    
    onnx.save(model, output_path)
    print(f"Saved model with named nodes to {output_path}")

if __name__ == "__main__":
    name_nodes(sys.argv[1], sys.argv[2])