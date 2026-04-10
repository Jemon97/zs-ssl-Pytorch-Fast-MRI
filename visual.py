import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from models import utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load data.mat, reconstruct zero-filled and reference images, and save them."
    )
    parser.add_argument(
        "--data_mat",
        type=str,
        default="data/data.mat",
        help="Path to data.mat containing kspace, sens_maps, and mask.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save reconstructed images.",
    )
    return parser.parse_args()


def load_data(mat_path):
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"data.mat not found: {mat_path}")

    data = sio.loadmat(mat_path)
    required_keys = ("kspace", "sens_maps", "mask")
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing keys in data.mat: {missing}")

    kspace = data["kspace"]
    sens_maps = data["sens_maps"]
    mask = data["mask"]
    full_kspace = data.get("full_kspace", None)
    return kspace, sens_maps, mask, full_kspace


def reconstruct(kspace, sens_maps, mask, full_kspace=None):
    # Match repository training/inference convention.
    kspace = kspace / np.max(np.abs(kspace))
    mask = np.asarray(mask).squeeze()
    ncoil = kspace.shape[-1]
    sampled_kspace = kspace * np.tile(mask[..., np.newaxis], (1, 1, ncoil))

    zero_filled = utils.sense1(sampled_kspace, sens_maps)
    if full_kspace is not None:
        full_kspace = full_kspace / np.max(np.abs(full_kspace))
        reference = utils.sense1(full_kspace, sens_maps)
    else:
        reference = utils.sense1(kspace, sens_maps)
    return zero_filled, reference


def normalize_pair_for_display(zero_filled, reference):
    reference_mag = np.abs(reference)
    zero_filled_mag = np.abs(zero_filled)
    factor = np.max(reference_mag)
    if factor == 0:
        return zero_filled_mag, reference_mag
    return zero_filled_mag / factor, reference_mag / factor


def save_image(img, path, title, vmax):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray", vmin=0.0, vmax=vmax)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    kspace, sens_maps, mask, full_kspace = load_data(args.data_mat)
    zero_filled, reference = reconstruct(kspace, sens_maps, mask, full_kspace=full_kspace)

    kspace_nonzero_ratio = np.count_nonzero(np.abs(kspace) > 0) / kspace.size
    diff_norm = np.linalg.norm(np.abs(reference) - np.abs(zero_filled)) / (np.linalg.norm(np.abs(reference)) + 1e-12)
    print(f"kspace_nonzero_ratio={kspace_nonzero_ratio:.4f}")
    print(f"relative_diff(reference, zero_filled)={diff_norm:.6f}")
    if full_kspace is None and diff_norm < 1e-3:
        print(
            "Warning: reference and zero-filled are almost identical. "
            "Current data.mat likely stores only undersampled k-space. "
            "Provide full_kspace in data.mat for a true reference image."
        )

    zero_filled_mag, reference_mag = normalize_pair_for_display(zero_filled, reference)
    display_vmax = 0.6 * np.max(reference_mag) if np.max(reference_mag) > 0 else 1.0

    zero_path = os.path.join(args.output_dir, "zero_filled.png")
    ref_path = os.path.join(args.output_dir, "reference.png")
    compare_path = os.path.join(args.output_dir, "zero_filled_vs_reference.png")

    save_image(zero_filled_mag, zero_path, "Zero-Filled Reconstruction", vmax=display_vmax)
    save_image(reference_mag, ref_path, "Reference Reconstruction", vmax=display_vmax)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(zero_filled_mag, cmap="gray", vmin=0.0, vmax=display_vmax)
    plt.title("Zero-Filled")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(reference_mag, cmap="gray", vmin=0.0, vmax=display_vmax)
    plt.title("Reference")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(compare_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"Saved: {zero_path}")
    print(f"Saved: {ref_path}")
    print(f"Saved: {compare_path}")


if __name__ == "__main__":
    main()
