"""
从 fastMRI 风格的多线圈 knee .h5 中随机取一个 slice，导出为 ZS-SSL 用的 data.mat。

data.mat 字段:
  - kspace:      complex64, (nrow, ncol, ncoil) 欠采样 k-space（用于 zero-filled）
  - full_kspace: complex64, (nrow, ncol, ncoil) 全量输入 k-space（用于 reference）
  - sens_maps:   complex64, (nrow, ncol, ncoil)
  - mask:        float32,   (nrow, ncol)，0/1 采样掩膜
"""
import argparse
import os
import sys
from typing import Optional, Tuple

import h5py
import numpy as np
import scipy.io as sio

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models import utils


def _center_crop_spatial(k_ncoil_first: np.ndarray, target_nrow: int, target_ncol: int) -> np.ndarray:
    """k_ncoil_first: (ncoil, nrow, ncol) -> cropped (ncoil, target_nrow, target_ncol)."""
    _, nr, nc = k_ncoil_first.shape
    r0 = (nr - target_nrow) // 2
    c0 = (nc - target_ncol) // 2
    return k_ncoil_first[:, r0 : r0 + target_nrow, c0 : c0 + target_ncol]


def _center_crop_mask_1d(mask_1d: np.ndarray, target_len: int) -> np.ndarray:
    L = int(mask_1d.shape[0])
    if L == target_len:
        return mask_1d
    c0 = (L - target_len) // 2
    return mask_1d[c0 : c0 + target_len]


def _infer_mask_1d_from_kspace(k_ncoil_first: np.ndarray) -> np.ndarray:
    """
    Infer a 1D phase-encode mask from k-space energy.
    k_ncoil_first: (ncoil, nrow, ncol)
    """
    line_energy = np.sum(np.abs(k_ncoil_first), axis=(0, 1))
    mask_1d = (line_energy > 0).astype(np.float32)
    if not np.any(mask_1d):
        # Fallback for unusual inputs: treat as fully sampled.
        mask_1d = np.ones((k_ncoil_first.shape[-1],), dtype=np.float32)
    return mask_1d


def _estimate_sens_maps(kspace: np.ndarray) -> np.ndarray:
    """kspace: (nrow, ncol, ncoil) complex —— 与 utils.sense1 一致的 RSS 归一化线圈灵敏度。"""
    coil_imgs = utils.ifft(kspace, axes=(0, 1), norm=None, unitary_opt=True)
    rss = np.sqrt(np.sum(np.abs(coil_imgs) ** 2, axis=2, keepdims=True))
    rss = np.maximum(rss, 1e-8)
    return (coil_imgs / rss).astype(np.complex64)


# To implement a Cartesian undersampling mask with a fully sampled Auto-Calibration Signal (ACS) region and an overall acceleration factor ($R$) of 4, you must override the existing 1D mask extraction and generate a synthetic mask prior to the 2D tiling step.This approach replaces the previous m1 logic. It forces the center of k-space to be densely sampled for sensitivity map estimation (e.g., ESPIRiT) and randomly samples the higher frequencies to achieve the target undersampling rate.Modified Code BlockPythonimport numpy as np
from typing import Optional, Tuple
import h5py

def build_mat_from_h5_slice(
    h5_path: str,
    slice_index: Optional[int],
    nrow: int,
    ncol: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    with h5py.File(h5_path, "r") as f:
        k_all = f["kspace"]
        ns,ncoil,nrow, ncol = k_all.shape
        idx = int(np.random.randint(0, ns)) if slice_index is None else int(slice_index)
        if not (0 <= idx < ns):
            raise IndexError(f"slice_index {idx} out of range [0, {ns})")
        k_sl = np.array(k_all[idx], dtype=np.complex64)

    # # Center crop the k-space data
    # k_sl = _center_crop_spatial(k_sl, nrow, ncol)

    # --- Custom Undersampling Logic (R=4, ACS=24) ---
    acs_lines = 24
    R = 4
    m1 = np.zeros(ncol, dtype=np.float32)

    # 1. Define and fully sample the central ACS region
    center_start = (ncol - acs_lines) // 2
    center_end = center_start + acs_lines
    m1[center_start:center_end] = 1.0

    # 2. Calculate remaining lines to sample in the outer high-frequency regions
    num_sampled_total = ncol // R
    num_outer_lines_to_sample = num_sampled_total - acs_lines

    if num_outer_lines_to_sample > 0:
        # Extract indices outside the ACS
        outer_indices = np.concatenate([
            np.arange(0, center_start),
            np.arange(center_end, ncol)
        ])
        
        # Randomly sample the outer phase-encoding lines without replacement
        sampled_outer = np.random.choice(outer_indices, size=num_outer_lines_to_sample, replace=False)
        m1[sampled_outer] = 1.0
    elif num_outer_lines_to_sample < 0:
        # Failsafe if the phase encoding dimension is too small for the requested ACS/R combination
        raise ValueError(f"ACS lines ({acs_lines}) exceed total allowed sampled lines ({num_sampled_total}) for R={R}.")
    # ------------------------------------------------

    full_kspace = np.transpose(k_sl, (1, 2, 0)).astype(np.complex64)
    
    # sens_maps estimation benefits from the contiguous 24 ACS lines
    sens_maps = _estimate_sens_maps(full_kspace) 
    
    # Tile the 1D phase-encoding mask along the readout dimension
    mask_2d = np.tile(m1[np.newaxis, :], (nrow, 1)).astype(np.float32)
    
    # Apply the mask to generate the undersampled measurement
    kspace = full_kspace * mask_2d[..., np.newaxis]
    
    return kspace.astype(np.complex64), full_kspace, sens_maps, mask_2d


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Export random fastMRI knee slice to data.mat for ZS-SSL.")
    ap.add_argument(
        "--h5_dir",
        default=os.path.join(here, "multicoil_train"),
        help="含 .h5 的目录（默认 data/multicoil_train）",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(here, "data_new.mat"),
        help="输出 .mat 路径",
    )
    ap.add_argument("--nrow", type=int, default=320, help="读方向中心裁剪尺寸（与 parser nrow_GLOB 一致）")
    ap.add_argument("--ncol", type=int, default=368, help="相位编码中心裁剪尺寸（与 parser ncol_GLOB 一致）")
    ap.add_argument("--h5_file", type=str, default=None, help="指定某个 .h5；不指定则随机选一个")
    ap.add_argument("--slice", type=int, default=None, help="指定 slice 索引；不指定则随机")
    ap.add_argument("--seed", type=int, default=None, help="随机种子（可复现）")
    args = ap.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.h5_file:
        h5_path = args.h5_file if os.path.isabs(args.h5_file) else os.path.join(args.h5_dir, args.h5_file)
    else:
        names = [n for n in os.listdir(args.h5_dir) if n.endswith(".h5")]
        if not names:
            raise FileNotFoundError(f"目录中无 .h5: {args.h5_dir}")
        h5_path = os.path.join(args.h5_dir, str(np.random.choice(names)))

    kspace, full_kspace, sens_maps, mask_2d = build_mat_from_h5_slice(
        h5_path, args.slice, args.nrow, args.ncol
    )

    sio.savemat(
        args.out,
        {
            "kspace": kspace,
            "full_kspace": full_kspace,
            "sens_maps": sens_maps,
            "mask": mask_2d,
        },
        format="5",
        do_compression=True,
    )
    print(
        f"Wrote {args.out}\n"
        f"  source: {h5_path}\n"
        f"  kspace {kspace.shape}, full_kspace {full_kspace.shape}, "
        f"sens_maps {sens_maps.shape}, mask {mask_2d.shape}"
    )


if __name__ == "__main__":
    main()
