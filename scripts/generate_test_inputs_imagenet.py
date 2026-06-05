#!/usr/bin/env python3
"""
Convert sample images to C header files for FPGA testing with SqueezeNet.

Takes images (from ImageNet validation set or custom) and generates C arrays
in fp16 format suitable for SAURIA testing.

Input shape: (1, 3, 224, 224) — standard SqueezeNet input.
Preprocessing: ImageNet normalization (mean/std per channel).
Labels: full ImageNet class indices (0–999); use -1 for unknown/synthetic.

Usage:
    # Generate from ImageNet validation set
    python generate_test_inputs_squeezenet.py --imagenet /path/to/imagenet/val -n 10 -o test_inputs.h

    # Generate from a folder of custom images with a known ImageNet label
    python generate_test_inputs_squeezenet.py --image-dir ./my_images --image-label 207 -o test_inputs.h

    # Generate a single custom image
    python generate_test_inputs_squeezenet.py --image dog.jpg --image-label 207 -o test_inputs.h

    # Generate synthetic test patterns (no external data needed)
    python generate_test_inputs_squeezenet.py --synthetic -o test_inputs.h

    # Generate all types
    python generate_test_inputs_squeezenet.py --all --imagenet /path/to/imagenet/val -o test_inputs.h
"""

import argparse
import struct
import numpy as np
import os

# ImageNet normalization constants (same as torchvision SqueezeNet preprocessing)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

INPUT_SHAPE = (1, 3, 224, 224)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def float_to_fp16_hex(value):
    """Convert a float to its fp16 bit pattern as a uint16."""
    fp16_val = np.float16(value)
    return np.frombuffer(fp16_val.tobytes(), dtype=np.uint16)[0]


def array_to_c_hex(arr, name, comment=""):
    """Convert a numpy array to a C uint16_t array declaration (fp16 encoded)."""
    flat = arr.flatten().astype(np.float16)
    hex_values = [float_to_fp16_hex(v) for v in flat]

    lines = []
    if comment:
        lines.append(f"/* {comment} */")
    shape_str = "x".join(map(str, arr.shape))
    lines.append(f"/* Shape: [{shape_str}], {len(hex_values)} elements */")
    lines.append(f"__attribute__((section(\".dram_section_data\")))")
    lines.append(f"static const uint16_t {name}[{len(hex_values)}] = {{")

    for i in range(0, len(hex_values), 8):
        row = hex_values[i:i+8]
        row_str = ", ".join(f"0x{v:04x}" for v in row)
        if i + 8 < len(hex_values):
            row_str += ","
        lines.append(f"    {row_str}")

    lines.append("};")
    lines.append("")
    return "\n".join(lines)


def normalize_imagenet(img_chw_float32):
    """Apply ImageNet mean/std normalisation to a (3, H, W) float32 array in [0,1]."""
    return ((img_chw_float32 - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float16)


# ---------------------------------------------------------------------------
# Header generation
# ---------------------------------------------------------------------------

def generate_header(samples, output_path, input_shape=INPUT_SHAPE):
    """Write a complete C header file containing all test samples."""
    N, C, H, W = input_shape
    lines = []

    lines += [
        "/* AUTO-GENERATED TEST INPUTS FOR SQUEEZENET - DO NOT EDIT */",
        f"/* Input shape: {input_shape}  (N x C x H x W) */",
        "/* Preprocessing: ImageNet mean/std normalisation, fp16 */",
        "",
        "#ifndef TEST_INPUTS_H",
        "#define TEST_INPUTS_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define INPUT_N {N}",
        f"#define INPUT_C {C}",
        f"#define INPUT_H {H}",
        f"#define INPUT_W {W}",
        f"#define INPUT_SIZE ({N} * {C} * {H} * {W})",
        f"#define NUM_TEST_SAMPLES {len(samples)}",
        "",
    ]

    for i, (name, arr, label, description) in enumerate(samples):
        var_name = f"test_input_{i}_{name}"
        comment = f"Sample {i}: {description}, Label: {label}"
        lines.append(array_to_c_hex(arr, var_name, comment))

    # Lookup table
    lines += [
        "/* Test sample lookup table */",
        "typedef struct {",
        "    const char*     name;",
        "    const uint16_t* data;",
        "    int             expected_label;  /* ImageNet class index (0-999), or -1 if unknown */",
        "    const char*     description;",
        "} test_sample_t;",
        "",
        f"static const test_sample_t test_samples[{len(samples)}] = {{",
    ]
    for i, (name, arr, label, description) in enumerate(samples):
        var_name = f"test_input_{i}_{name}"
        lines.append(f'    {{ "{name}", {var_name}, {label}, "{description}" }},')
    lines += ["};", ""]

    # Helper
    lines += [
        "/* Helper: get input as _Float16 pointer */",
        "static inline const _Float16* get_test_input(int sample_idx) {",
        "    return (const _Float16*)test_samples[sample_idx].data;",
        "}",
        "",
        "#endif /* TEST_INPUTS_H */",
    ]

    with open(output_path, 'w') as f:
        f.write("\n".join(lines))

    print(f"Generated: {output_path}")
    print(f"  Samples    : {len(samples)}")
    print(f"  Input shape: {input_shape}")


# ---------------------------------------------------------------------------
# Sample sources
# ---------------------------------------------------------------------------

def load_imagenet_samples(imagenet_val_dir, num_samples=10):
    """
    Load samples from an ImageNet validation directory.

    Expected layout (standard ImageNet):
        imagenet_val_dir/
            n01440764/   <- synset folder
                ILSVRC2012_val_00000293.JPEG
                ...
            n01443537/
                ...

    Alternatively a flat folder of JPEG/PNG files is also supported
    (labels will be set to -1 / unknown in that case).

    Requires: torchvision, Pillow
    """
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),                              # -> [0,1] float32 CHW
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std= [0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(root=imagenet_val_dir, transform=transform)
    print(f"  Found {len(dataset)} images across {len(dataset.classes)} classes")

    # Build a mapping from ImageFolder's internal label index -> true ImageNet class index.
    #
    # ImageFolder assigns labels by sorting folder names alphabetically.
    # • Synset layout  (n01440764/): folder name is not a class index; ImageFolder's
    #   label order matches the standard torchvision convention, so use as-is.
    # • Integer layout (0/, 1/, 2/, ...): folder name IS the class index, but
    #   alphabetical sort of strings ("0","1","10","100",...) differs from numeric
    #   order, so we must remap alphabetical label -> int(folder_name).
    sample_folder = dataset.classes[0]
    if sample_folder.isdigit():
        label_remap = {
            folder_label: int(dataset.classes[folder_label])
            for folder_label in range(len(dataset.classes))
        }
        print("  Detected integer-named class folders — remapping labels to numeric order.")
    else:
        label_remap = None
        print("  Detected synset-named class folders — using ImageFolder label order.")

    samples = []
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    for idx in indices:
        img_tensor, folder_label = dataset[idx]
        true_label = label_remap[folder_label] if label_remap else folder_label
        img_np = img_tensor.numpy().astype(np.float16).reshape(INPUT_SHAPE)
        class_name = dataset.classes[folder_label]
        name = f"imagenet_{true_label}_{idx}"
        desc = f"ImageNet val class {true_label} folder '{class_name}' (dataset idx {idx})"
        samples.append((name, img_np, true_label, desc))

    return samples


def load_custom_image(image_path, label):
    """Load, resize to 224x224, and normalise a single custom image."""
    from PIL import Image

    img = Image.open(image_path).convert('RGB')
    # Resize shortest side to 256, then centre-crop to 224
    w, h = img.size
    scale = 256 / min(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    left = (img.width  - 224) // 2
    top  = (img.height - 224) // 2
    img  = img.crop((left, top, left + 224, top + 224))

    img_np = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0  # CHW
    img_np = normalize_imagenet(img_np).reshape(INPUT_SHAPE)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    name = stem.replace(' ', '_')
    desc = f"Custom image: {image_path}"
    return [(name, img_np, label, desc)]


def load_image_dir(image_dir, label):
    """Load all images from a directory."""
    from PIL import Image

    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    samples = []
    for fname in sorted(os.listdir(image_dir)):
        if os.path.splitext(fname)[1].lower() not in exts:
            continue
        path = os.path.join(image_dir, fname)
        samples.extend(load_custom_image(path, label))
    return samples


def generate_synthetic_samples():
    """
    Generate synthetic 3×224×224 test patterns (no external data needed).
    All patterns are ImageNet-normalised so value ranges match real inputs.
    """
    samples = []
    H, W = 224, 224

    def norm(arr_f32):
        """Normalise a (1,3,H,W) float32 array and return float16."""
        return normalize_imagenet(arr_f32[0]).reshape(INPUT_SHAPE)

    # 1. Black (all zeros)
    black = np.zeros((1, 3, H, W), dtype=np.float32)
    samples.append(("black", norm(black), -1, "Solid black image"))

    # 2. White (all ones)
    white = np.ones((1, 3, H, W), dtype=np.float32)
    samples.append(("white", norm(white), -1, "Solid white image"))

    # 3. 50% grey
    grey = np.full((1, 3, H, W), 0.5, dtype=np.float32)
    samples.append(("grey50", norm(grey), -1, "50% grey image"))

    # 4. Vertical gradient (dark top → bright bottom)
    grad_v = np.zeros((1, 3, H, W), dtype=np.float32)
    for i in range(H):
        grad_v[0, :, i, :] = i / (H - 1)
    samples.append(("gradient_v", norm(grad_v), -1, "Vertical gradient"))

    # 5. Horizontal gradient
    grad_h = np.zeros((1, 3, H, W), dtype=np.float32)
    for j in range(W):
        grad_h[0, :, :, j] = j / (W - 1)
    samples.append(("gradient_h", norm(grad_h), -1, "Horizontal gradient"))

    # 6. Checkerboard (16×16 cells)
    checker = np.zeros((1, 3, H, W), dtype=np.float32)
    cell = 16
    for i in range(H):
        for j in range(W):
            if (i // cell + j // cell) % 2 == 0:
                checker[0, :, i, j] = 1.0
    samples.append(("checker", norm(checker), -1, "Checkerboard 16px"))

    # 7. Horizontal stripes (alternating every 8 rows)
    stripes_h = np.zeros((1, 3, H, W), dtype=np.float32)
    for i in range(0, H, 16):
        stripes_h[0, :, i:i+8, :] = 1.0
    samples.append(("stripes_h", norm(stripes_h), -1, "Horizontal stripes"))

    # 8. Vertical stripes
    stripes_v = np.zeros((1, 3, H, W), dtype=np.float32)
    for j in range(0, W, 16):
        stripes_v[0, :, :, j:j+8] = 1.0
    samples.append(("stripes_v", norm(stripes_v), -1, "Vertical stripes"))

    # 9. Random uniform noise (seeded)
    np.random.seed(42)
    noise = np.random.rand(1, 3, H, W).astype(np.float32)
    samples.append(("noise_uniform", norm(noise), -1, "Uniform random noise (seed 42)"))

    # 10. Gaussian noise centred at 0.5
    np.random.seed(7)
    gnoise = np.clip(np.random.randn(1, 3, H, W).astype(np.float32) * 0.2 + 0.5, 0, 1)
    samples.append(("noise_gaussian", norm(gnoise), -1, "Gaussian noise (mu=0.5, sigma=0.2)"))

    # 11. Centred bright rectangle
    rect = np.zeros((1, 3, H, W), dtype=np.float32)
    rect[0, :, 56:168, 56:168] = 0.8   # 112×112 centred square
    samples.append(("centre_rect", norm(rect), -1, "Bright centre rectangle"))

    # 12. Radial gradient (bright centre → dark edges)
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    max_r = np.sqrt(cy**2 + cx**2)
    radial = 1.0 - np.sqrt((Y - cy)**2 + (X - cx)**2) / max_r
    radial = np.clip(radial, 0, 1).astype(np.float32)
    radial_img = np.stack([radial, radial, radial])[np.newaxis]   # (1,3,H,W)
    samples.append(("radial_gradient", norm(radial_img), -1, "Radial gradient (bright centre)"))

    return samples


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_sample_stats(samples):
    print("\nSample Statistics:")
    print("-" * 70)
    for i, (name, arr, label, desc) in enumerate(samples):
        label_str = str(label) if label >= 0 else "unknown"
        print(f"  [{i:2d}] {name}")
        print(f"        label={label_str}  shape={arr.shape}  "
              f"range=[{float(arr.min()):.3f}, {float(arr.max()):.3f}]")
        print(f"        {desc}")
    print("-" * 70)
    known = sum(1 for _, _, l, _ in samples if l >= 0)
    unknown = len(samples) - known
    print(f"Total: {len(samples)} samples  ({known} with known labels, {unknown} unknown/synthetic)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate C fp16 test inputs for SqueezeNet FPGA testing (3×224×224)')
    parser.add_argument('-o', '--output', default='test_inputs.h',
                        help='Output header file (default: test_inputs.h)')
    parser.add_argument('--imagenet', metavar='DIR',
                        help='Path to ImageNet validation directory (ImageFolder layout)')
    parser.add_argument('-n', '--num-samples', type=int, default=10,
                        help='Number of ImageNet samples to draw (default: 10)')
    parser.add_argument('--image', metavar='FILE',
                        help='Single custom image path')
    parser.add_argument('--image-label', type=int, default=-1,
                        help='ImageNet class index (0-999) for custom image(s); -1 = unknown (default: -1)')
    parser.add_argument('--image-dir', metavar='DIR',
                        help='Directory of custom images (all get --image-label)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Generate synthetic test patterns')
    parser.add_argument('--all', action='store_true',
                        help='Generate all available sample types')
    args = parser.parse_args()

    samples = []

    if args.all or args.synthetic:
        print("Generating synthetic samples...")
        samples.extend(generate_synthetic_samples())

    if args.all or args.imagenet:
        dir_ = args.imagenet or '.'
        print(f"Loading {args.num_samples} ImageNet samples from {dir_} ...")
        samples.extend(load_imagenet_samples(dir_, args.num_samples))

    if args.image_dir:
        print(f"Loading images from directory: {args.image_dir}")
        samples.extend(load_image_dir(args.image_dir, args.image_label))

    if args.image:
        print(f"Loading custom image: {args.image}")
        samples.extend(load_custom_image(args.image, args.image_label))

    if not samples:
        print("No source specified — generating synthetic samples by default.")
        samples.extend(generate_synthetic_samples())

    print_sample_stats(samples)
    generate_header(samples, args.output, input_shape=INPUT_SHAPE)

    print(f"\nUsage in C code:")
    print('  #include "test_inputs.h"')
    print('  ')
    print('  for (int i = 0; i < NUM_TEST_SAMPLES; i++) {')
    print('      const _Float16* input = get_test_input(i);')
    print('      printf("Testing: %s\\n", test_samples[i].name);')
    print('      entry(input, output);')
    print('  }')


if __name__ == '__main__':
    main()