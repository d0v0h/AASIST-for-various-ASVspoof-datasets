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

import torch
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_loaders.loader_asv5 import get_database_path, get_loader
from evaluate_package.ASVspoof5.eval.calculate_metrics import calculate_minDCF_EER_CLLR_actDCF
from evaluate_package.ASVspoof5.eval.util import load_cm_scores_keys
from common.models import get_model
from common.utils import create_optimizer, set_seed
from common.training import train_epoch
from evaluate_package.evaluation_file_asv5 import produce_evaluation_file

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
        'Track1',
        'asv5',
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

    # define database path
    paths = get_database_path(database_path)

    # define dataloaders
    trn_loader, dev_loader, eval_loader = get_loader(
        seed=args.seed,
        trn_meta_path=paths["trn_meta_path"],
        dev_meta_path=paths["dev_meta_path"],
        eval_meta_path=paths["eval_meta_path"],
        trn_flac_path=paths["trn_flac_path"],
        dev_flac_path=paths["dev_flac_path"],
        eval_flac_path=paths["eval_flac_path"],
        config=config)

    # evaluates pretrained model 
    # NOTE: Currently it is evaluated on the development set instead of the evaluation set
    if args.eval:
        model.load_state_dict(
            torch.load(config["model_path"], map_location=device))
        print("Model loaded : {}".format(config["model_path"]))
        print("Start evaluation...")
        produce_evaluation_file(eval_loader, model, device,
                                eval_score_path, paths["eval_meta_path"])

        cm_scores, cm_keys = load_cm_scores_keys(
            cm_scores_file=eval_score_path,
            cm_keys_file=paths["eval_meta_path"])
        
        minDCF, eer, cllr, actDCF = calculate_minDCF_EER_CLLR_actDCF(
            cm_scores = cm_scores,
            cm_keys = cm_keys,
            output_file=model_tag / "track1_result.txt")
        print("-eval_mindcf: {:.5f}\n-eval_eer (%): {:.3f}\n-eval_cllr (bits): {:.5f}\n-eval_actDCF: {:.5f}\n".format(
            minDCF, eer*100, cllr, actDCF))
        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_minDCF = 1.
    best_dev_actDCF = 1.
    best_dev_eer = 100.
    best_dev_cllr = 1.
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training
    for epoch in range(config["num_epochs"]):
        print("training epoch{:03d}".format(epoch))
        
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                   scheduler, config)
        
        produce_evaluation_file(dev_loader, model, device,
                                metric_path/"dev_score.txt", paths["dev_meta_path"])
        
        cm_scores, cm_keys = load_cm_scores_keys(
            cm_scores_file=metric_path/"dev_score.txt",
            cm_keys_file=paths["dev_meta_path"])

        dev_minDCF, dev_eer, dev_cllr, dev_actDCF = calculate_minDCF_EER_CLLR_actDCF(
            cm_scores = cm_scores,
            cm_keys = cm_keys,
            output_file=model_tag / "track1_result.txt")
        
        print("-eval_mindcf: {:.5f}\n-eval_eer (%): {:.3f}\n-eval_cllr (bits): {:.5f}\n-eval_actDCF: {:.5f}\n".format(
            dev_minDCF, dev_eer*100, dev_cllr, dev_actDCF))

        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_minDCF", dev_minDCF, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        writer.add_scalar("dev_cllr", dev_cllr, epoch)
        writer.add_scalar("dev_actDCF", dev_actDCF, epoch)

        torch.save(model.state_dict(),
                       model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))

        best_dev_minDCF = min(dev_minDCF, best_dev_minDCF)
        best_dev_actDCF = min(dev_actDCF, best_dev_actDCF)
        best_dev_cllr = min(dev_cllr, best_dev_cllr)
        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer
            
            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
        writer.add_scalar("best_dev_actDCF", best_dev_actDCF, epoch)
        writer.add_scalar("best_dev_minDCF", best_dev_minDCF, epoch)
        writer.add_scalar("best_dev_cllr", best_dev_cllr, epoch)


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
    main(parser.parse_args())
