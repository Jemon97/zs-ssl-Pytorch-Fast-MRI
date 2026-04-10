import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import time

from models import UnrollNet, parser_ops, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ZS-SSL inference from an existing checkpoint."
    )
    parser.add_argument(
        "--data_mat",
        type=str,
        default="data/data.mat",
        help="Path to .mat file containing kspace, sens_maps, mask and optionally full_kspace.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="saved_models/ZS_SSL_Model_300Epochs_Rate4_10Unrolls/best.pth",
        help="Path to model checkpoint (.pth).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save inference results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )
    parser.add_argument(
        "--save_mat",
        action="store_true",
        help="Save inference outputs to a MAT file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print step-by-step progress logs.",
    )
    return parser.parse_args()


def load_data(mat_path: str):
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"Input data not found: {mat_path}")

    data = sio.loadmat(mat_path)
    required = ("kspace", "sens_maps", "mask")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing keys in {mat_path}: {missing}")

    kspace = data["kspace"]
    sens_maps = data["sens_maps"]
    mask = np.asarray(data["mask"]).squeeze()
    full_kspace = data.get("full_kspace", None)

    if kspace.ndim != 3:
        raise ValueError(f"Expected kspace with shape (H, W, C), got {kspace.shape}")
    if sens_maps.shape != kspace.shape:
        raise ValueError(
            f"sens_maps shape {sens_maps.shape} must match kspace shape {kspace.shape}"
        )
    if mask.shape != kspace.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape} must match kspace spatial shape {kspace.shape[:2]}"
        )

    return kspace, sens_maps, mask, full_kspace


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model_for_kspace_shape(kspace_shape, device: torch.device) -> torch.nn.Module:
    parser = parser_ops.get_parser()
    model_args = parser.parse_args([])
    model_args.nrow_GLOB, model_args.ncol_GLOB, model_args.ncoil_GLOB = kspace_shape
    model = UnrollNet.UnrolledNet(model_args, device=device).to(device)
    return model


def run_inference(
    model: torch.nn.Module,
    kspace: np.ndarray,
    sens_maps: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
):
    # Match train.py normalization behavior.
    kspace = kspace / (np.max(np.abs(kspace)) + 1e-12)

    sampled_kspace = kspace * np.tile(mask[..., np.newaxis], (1, 1, kspace.shape[-1]))
    zero_filled = utils.sense1(sampled_kspace, sens_maps)

    nw_input = utils.complex2real(zero_filled[np.newaxis])  # [1, H, W, 2]
    nw_input = torch.from_numpy(nw_input).permute(0, 3, 1, 2).float().to(device)  # [1, 2, H, W]
    nw_mask = torch.from_numpy(mask[np.newaxis]).to(device)
    nw_sens_maps = torch.from_numpy(np.transpose(sens_maps[np.newaxis], (0, 3, 1, 2))).to(device)

    model.eval()
    with torch.no_grad():
        recon, _, _ = model(nw_input, nw_mask, nw_mask, nw_sens_maps)

    recon = recon.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy()
    recon = utils.real2complex(recon)

    return zero_filled, recon


def save_plot(zero_filled, recon, reference, output_path: str) -> None:
    zf = np.abs(zero_filled)
    rec = np.abs(recon)
    ref = np.abs(reference)

    factor = np.max(ref) + 1e-12
    zf, rec, ref = zf / factor, rec / factor, ref / factor
    vmax = 0.6 * np.max(ref) if np.max(ref) > 0 else 1.0

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(zf, cmap="gray", vmin=0.0, vmax=vmax)
    plt.title("Zero-filled")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(rec, cmap="gray", vmin=0.0, vmax=vmax)
    plt.title("ZS-SSL Recon")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(ref, cmap="gray", vmin=0.0, vmax=vmax)
    plt.title("Reference")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close()


def nmse(pred: np.ndarray, target: np.ndarray) -> float:
    pred_mag = np.abs(pred)
    tgt_mag = np.abs(target)
    return float((np.linalg.norm(pred_mag - tgt_mag) ** 2) / (np.linalg.norm(tgt_mag) ** 2 + 1e-12))


def main() -> None:
    args = parse_args()
    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.verbose:
        print("[1/5] Loading data...")

    kspace, sens_maps, mask, full_kspace = load_data(args.data_mat)
    device = get_device(args.device)
    if args.verbose:
        print(f"      kspace shape={kspace.shape}, device={device}")
        print("[2/5] Building model...")

    model = build_model_for_kspace_shape(kspace.shape, device)
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.verbose:
        print("[3/5] Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    if args.verbose:
        print("[4/5] Running forward inference...")
    t_infer = time.time()
    zero_filled, recon = run_inference(model, kspace, sens_maps, mask, device=device)
    if args.verbose:
        print(f"      forward done in {time.time() - t_infer:.2f}s")

    if full_kspace is not None:
        full_kspace = full_kspace / (np.max(np.abs(full_kspace)) + 1e-12)
        reference = utils.sense1(full_kspace, sens_maps)
    else:
        # Fallback reference if full_kspace is unavailable.
        reference = utils.sense1(kspace / (np.max(np.abs(kspace)) + 1e-12), sens_maps)

    fig_path = os.path.join(args.output_dir, "checkpoint_recon_compare.png")
    if args.verbose:
        print("[5/5] Saving outputs...")
    save_plot(zero_filled, recon, reference, fig_path)

    print(f"Saved figure: {fig_path}")
    print(f"Data shape: {kspace.shape}")
    print(f"Device: {device}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"NMSE(zero_filled vs ref): {nmse(zero_filled, reference):.6f}")
    print(f"NMSE(recon vs ref): {nmse(recon, reference):.6f}")
    print(f"Total elapsed: {time.time() - t0:.2f}s")

    if args.save_mat:
        mat_path = os.path.join(args.output_dir, "inference_output.mat")
        sio.savemat(
            mat_path,
            {
                "zero_filled": np.abs(zero_filled),
                "zs_ssl_recon": np.abs(recon),
                "reference": np.abs(reference),
                "mask": mask,
            },
        )
        print(f"Saved MAT: {mat_path}")


if __name__ == "__main__":
    main()
