"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from shutil import copy
import io
import contextlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from evaluate_package.ASVspoof2019.evaluation import calculate_tDCF_EER
from common.utils import create_optimizer, seed_worker, set_seed, str_to_bool
from common.models import get_model
from common.training import train_epoch
from data_loaders.loader_2021 import get_database_path, get_loader
from evaluate_package.evaluation_file_2019 import produce_evaluation_file as produce_evaluation_file_2019
from evaluate_package.evaluation_file_2021 import produce_evaluation_file
from evaluate_package.ASVspoof2021.main import evaluation_API

warnings.filterwarnings("ignore", category=FutureWarning)


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = args.track
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    database_path = Path('DB')

    # define model related paths
    model_tag = "{}_{}_{}".format(
        track,
        2021,
        os.path.splitext(os.path.basename(args.config))[0])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)

    # define dataset path
    paths = get_database_path(database_path, track=track)

    # define dataloaders
    trn_loader, dev_loader, eval_loader = get_loader(
        seed = args.seed,
        trn_meta_path=paths['trn_meta_path'],
        dev_meta_path=paths['dev_meta_path'],
        eval_meta_path=paths['eval_meta_path'],
        trn_flac_path=paths['trn_flac_path'],
        dev_flac_path=paths['dev_flac_path'],
        eval_flac_path=paths['eval_flac_path'],
        config=config)

    # evaluates pretrained model and exit script
    if args.eval:
        model.load_state_dict(
            torch.load(config["model_path"], map_location=device))
        print("Model loaded : {}".format(config["model_path"]))
        print("Start evaluation...")
        produce_evaluation_file(eval_loader, model, device,
                                eval_score_path, paths["eval_meta_path"])
        _buf = io.StringIO()
        with contextlib.redirect_stdout(_buf):
            eval_tdcf, eval_eer = evaluation_API(
                    cm_score_file=eval_score_path,
                    track=args.track,
                    label_dir=paths['asv_keys_path'],
                )
        table_text = _buf.getvalue()
        with open(model_tag / "t-DCF_EER.txt", "w") as fh:
            fh.write(table_text)
        print("DONE.")
        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_eer = 100.
    best_eval_eer = 100.
    best_dev_tdcf = 0.05
    best_eval_tdcf = 1.
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training
    for epoch in range(config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                   scheduler, config)
        produce_evaluation_file_2019(dev_loader, model, device,
                                metric_path/"dev_score.txt", paths["dev_meta_path"])
        dev_eer, dev_tdcf = calculate_tDCF_EER(
            cm_scores_file=metric_path/"dev_score.txt",
            asv_score_file=paths['asv_score_dev_path'],
            output_file=metric_path/"dev_t-DCF_EER_{}epo.txt".format(epoch),
            printout=False)
        print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_tdcf:{:.5f}".format(
            running_loss, dev_eer, dev_tdcf))
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        writer.add_scalar("dev_tdcf", dev_tdcf, epoch)

        best_dev_tdcf = min(dev_tdcf, best_dev_tdcf)
        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer
            torch.save(model.state_dict(),
                       model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))

            # do evaluation whenever best model is renewed
            if str_to_bool(config["eval_all_best"]):
                produce_evaluation_file(eval_loader, model, device,
                                        eval_score_path, paths["eval_meta_path"])
                tdcf_array, eer_array = evaluation_API(
                    cm_score_file=eval_score_path,
                    track=args.track,
                    label_dir=paths['asv_keys_path'])
                # extract pooled t-DCF and EER
                eval_eer = eer_array[-1][-1]
                eval_tdcf = tdcf_array[-1][-1]

                log_text = "epoch{:03d}, ".format(epoch)

                if track == "LA":
                    # primary metric for LA is t-DCF
                    # secondary metric is EER
                    if eval_eer < best_eval_eer:
                        log_text += "best eer, {:.4f}%".format(eval_eer)
                        best_eval_eer = eval_eer
                    if eval_tdcf < best_eval_tdcf:
                        log_text += "best tdcf, {:.4f}".format(eval_tdcf)
                        best_eval_tdcf = eval_tdcf
                        torch.save(model.state_dict(),
                                model_save_path / "best.pth")

                elif track == "DF":
                    # primary metric for DF is EER
                    if eval_eer < best_eval_eer:
                        log_text += "best eer, {:.4f}%".format(eval_eer)
                        best_eval_eer = eval_eer
                        torch.save(model.state_dict(),
                                model_save_path / "best.pth")

                if len(log_text) > 0:
                    print(log_text)
                    f_log.write(log_text + "\n")

            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
        writer.add_scalar("best_dev_tdcf", best_dev_tdcf, epoch)

    print("Start final evaluation")
    epoch += 1
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    produce_evaluation_file(eval_loader, model, device, eval_score_path,
                            paths["eval_meta_path"])
    tdcf_array, eer_array = evaluation_API(
        cm_score_file=eval_score_path,
        track=args.track,
        label_dir=paths['asv_keys_path'])
    # extract pooled t-DCF and EER
    eval_eer = eer_array[-1][-1] * 100
    eval_tdcf = tdcf_array[-1][-1]
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    if track == "LA":
        f_log.write("EER: {:.3f}, min t-DCF: {:.5f}".format(eval_eer, eval_tdcf))
    elif track == "DF":
        f_log.write("EER: {:.3f}".format(eval_eer))
    f_log.close()

    torch.save(model.state_dict(),
               model_save_path / "swa.pth")

    if eval_eer <= best_eval_eer:
        best_eval_eer = eval_eer
    if eval_tdcf <= best_eval_tdcf:
        best_eval_tdcf = eval_tdcf
        torch.save(model.state_dict(),
                   model_save_path / "best.pth")
    print("Exp FIN. EER: {:.3f}, min t-DCF: {:.5f}".format(
        best_eval_eer, best_eval_tdcf))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    parser.add_argument("--track",
                        type=str,
                        default="LA",
                        help="specify the track of ASVspoof (LA/DF)")
    main(parser.parse_args())
