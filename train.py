import os
import time
import argparse

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader

from models import utils, parser_ops, UnrollNet
from models.modules import MixL1L2Loss, Dataset, Dataset_Inference, train, validation, test


def build_parser() -> argparse.ArgumentParser:
    parser = parser_ops.get_parser()
    parser.add_argument(
        "--do_inference",
        action="store_true",
        help="Run inference after training using best checkpoint.",
    )
    parser.add_argument(
        "--plot_loss",
        action="store_true",
        help="Plot training/validation loss curves at the end (requires matplotlib).",
    )
    return parser


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = build_parser()
    args = parser.parse_args()

    # -------------------- Load data --------------------
    data = sio.loadmat(args.data_dir)
    kspace_train,full_kspace, sens_maps, original_mask = data["kspace"], data["full_kspace"], data["sens_maps"], data["mask"]
    args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB = kspace_train.shape

    # Normalize the kspace to 0-1 region
    kspace_train = kspace_train / np.max(np.abs(kspace_train[:]))

    # -------------------- Validation masks/data --------------------
    cv_trn_mask, cv_val_mask = utils.uniform_selection(kspace_train, original_mask, rho=args.rho_val)
    remainder_mask, cv_val_mask = np.copy(cv_trn_mask), np.copy(np.complex64(cv_val_mask))

    ref_kspace_val = np.empty(
        (args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64
    )
    nw_input_val = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)

    nw_input_val = utils.sense1(
        kspace_train * np.tile(cv_trn_mask[:, :, np.newaxis], (1, 1, args.ncoil_GLOB)), sens_maps
    )[np.newaxis]
    ref_kspace_val = (
        kspace_train * np.tile(cv_val_mask[:, :, np.newaxis], (1, 1, args.ncoil_GLOB))
    )[np.newaxis]

    print(
        "size of kspace: ",
        kspace_train[np.newaxis, ...].shape,
        ", maps: ",
        sens_maps.shape,
        ", mask: ",
        original_mask.shape,
    )

    # -------------------- Training data --------------------
    nw_input_trn = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)
    ref_kspace = np.empty(
        (args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64
    )

    trn_mask = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)
    loss_mask = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)

    for jj in range(args.num_reps):
        trn_mask[jj, ...], loss_mask[jj, ...] = utils.uniform_selection(
            kspace_train, remainder_mask, rho=args.rho_train
        )

        sub_kspace = kspace_train * np.tile(trn_mask[jj][..., np.newaxis], (1, 1, args.ncoil_GLOB))
        ref_kspace[jj, ...] = kspace_train * np.tile(
            loss_mask[jj][..., np.newaxis], (1, 1, args.ncoil_GLOB)
        )
        nw_input_trn[jj, ...] = utils.sense1(sub_kspace, sens_maps)

    # # zeropadded outer edges of k-space with no signal
    # if args.data_opt == "Coronal_PD":
    #     trn_mask[:, :, 0:17] = np.ones((args.num_reps, args.nrow_GLOB, 17))
    #     trn_mask[:, :, 352 : args.ncol_GLOB] = np.ones((args.num_reps, args.nrow_GLOB, 16))

    # -------------------- Prepare arrays for training --------------------
    sens_maps = np.tile(sens_maps[np.newaxis], (args.num_reps, 1, 1, 1))
    sens_maps = np.transpose(sens_maps, (0, 3, 1, 2))
    ref_kspace = utils.complex2real(np.transpose(ref_kspace, (0, 3, 1, 2)))
    nw_input_trn = utils.complex2real(nw_input_trn)

    ref_kspace_val = utils.complex2real(np.transpose(ref_kspace_val, (0, 3, 1, 2)))
    nw_input_val = utils.complex2real(nw_input_val)

    # -------------------- Dataloaders --------------------
    train_data = Dataset(nw_input_trn, trn_mask, loss_mask, sens_maps, ref_kspace)
    train_loader = DataLoader(train_data, batch_size=args.batchSize, shuffle=True, num_workers=0)

    val_data = Dataset(
        nw_input_val,
        cv_trn_mask[np.newaxis],
        cv_val_mask[np.newaxis],
        sens_maps[0][np.newaxis],
        ref_kspace_val,
    )
    val_loader = DataLoader(val_data, batch_size=args.batchSize, shuffle=False, num_workers=0)

    # -------------------- Model/optim --------------------
    directory = os.path.join(
        "saved_models",
        "ZS_SSL_Model_" + str(args.epochs) + "Epochs_Rate" + str(args.acc_rate) + "_" + str(args.nb_unroll_blocks) + "Unrolls",
    )
    os.makedirs(directory, exist_ok=True)

    model = UnrollNet.UnrolledNet(args, device=device).to(device)
    loss_fn = MixL1L2Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # -------------------- Resume (if available) --------------------
    last_ckpt_path = os.path.join(directory, "last.pth")
    log_path = os.path.join(directory, "TrainingLog.mat")

    total_train_loss, total_val_loss = [], []
    valid_loss_min = np.inf
    ep, val_loss_tracker = 0, 0

    if os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])

        ep = int(ckpt.get("epoch", -1)) + 1
        valid_loss_min = float(ckpt.get("best_valid_loss", ckpt.get("valid_loss_min", np.inf)))
        val_loss_tracker = int(ckpt.get("val_loss_tracker", 0))

        if os.path.exists(log_path):
            try:
                mat = sio.loadmat(log_path)
                if "trn_loss" in mat:
                    total_train_loss = [float(x) for x in np.squeeze(mat["trn_loss"]).tolist()]
                if "val_loss" in mat:
                    total_val_loss = [float(x) for x in np.squeeze(mat["val_loss"]).tolist()]
            except Exception:
                total_train_loss, total_val_loss = [], []

        print(f"Resumed from {last_ckpt_path}; next epoch to run: {ep+1}.")

    # -------------------- Train loop --------------------
    start_time = time.time()
    while ep < args.epochs and val_loss_tracker < args.stop_training:
        tic = time.time()
        trn_loss, lamdas = train(train_loader, model, loss_fn, optimizer, device=device)
        val_loss = validation(val_loader, model, loss_fn, device=device)
        total_train_loss.append(trn_loss)
        total_val_loss.append(val_loss)

        toc = time.time() - tic
        print(
            "Epoch:",
            ep + 1,
            ", elapsed_time = ""{:f}".format(toc),
            ", trn loss = ",
            "{:.3f}".format(trn_loss),
            ", val loss = ",
            "{:.3f}".format(val_loss),
        )

        checkpoint = {
            "epoch": ep,
            "val_loss": float(val_loss),
            "best_valid_loss": float(valid_loss_min),
            "val_loss_tracker": int(val_loss_tracker),
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
        }
        torch.save(checkpoint, last_ckpt_path)

        if val_loss <= valid_loss_min:
            valid_loss_min = val_loss
            torch.save(checkpoint, os.path.join(directory, "best.pth"))
            val_loss_tracker = 0
        else:
            val_loss_tracker += 1

        sio.savemat(log_path, {"trn_loss": total_train_loss, "val_loss": total_val_loss})
        ep += 1

    end_time = time.time()
    print("Training completed in ", str(ep), " epochs, ", ((end_time - start_time) / 60), " minutes")

    # -------------------- Optional inference --------------------
    if args.do_inference:
        test_mask = np.complex64(original_mask)
        nw_input_inference = utils.sense1(
            kspace_train * np.tile(test_mask[..., np.newaxis], (1, 1, args.ncoil_GLOB)),
            np.transpose(sens_maps[0], (1, 2, 0)),
        )
        ref_image = utils.sense1(kspace_train, np.transpose(sens_maps[0], (1, 2, 0)))
        if args.data_opt == "Coronal_PD":
            test_mask[:, 0:17] = np.ones((args.nrow_GLOB, 17))
            test_mask[:, 352 : args.ncol_GLOB] = np.ones((args.nrow_GLOB, 16))

        test_data = Dataset_Inference(
            utils.complex2real(nw_input_inference[np.newaxis]),
            test_mask[np.newaxis],
            test_mask[np.newaxis],
            sens_maps[0][np.newaxis],
        )
        test_loader = DataLoader(test_data, batch_size=args.batchSize, shuffle=False, num_workers=0)

        best_checkpoint = torch.load(os.path.join(directory, "best.pth"), map_location=device)
        model.load_state_dict(best_checkpoint["model_state"])
        zs_ssl_recon = test(test_loader, model, device)
        zs_ssl_recon = utils.real2complex(zs_ssl_recon.to("cpu").numpy())

        if args.data_opt == "Coronal_PD":
            factor = np.max(np.abs(ref_image[:]))
        else:
            factor = 1

        ref_image = np.abs(ref_image) / factor
        nw_input_inference = np.abs(nw_input_inference) / factor
        zs_ssl_recon = np.abs(zs_ssl_recon) / factor

        out_path = os.path.join(directory, "inference.mat")
        sio.savemat(
            out_path,
            {"ref_image": ref_image, "zero_filled": nw_input_inference, "zs_ssl_recon": zs_ssl_recon},
        )
        print(f"Saved inference outputs to {out_path}")

    # -------------------- Optional plots --------------------
    if args.plot_loss:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(np.asarray(total_train_loss).T)
        plt.plot(np.asarray(total_val_loss).T)
        plt.title("Loss Curves"), plt.xlabel("Epochs"), plt.ylabel("Loss")
        plt.legend(["trn loss", "val loss"])
        plt.grid()
        plt.show()


if __name__ == "__main__":
    main()

