

#!/usr/bin/env python3
"""
major actions here: fine-tune the features and evaluate different settings
"""
import os
import torch
import warnings

import numpy as np
import random

from time import sleep
from random import randint

import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model
from src.utils.file_io import PathManager

from launch import default_argument_parser, logging_train_setup
warnings.filterwarnings("ignore")


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # setup output dir
    output_dir = cfg.OUTPUT_DIR
    data_name = cfg.DATA.NAME
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    
    TRANSFER_TYPE = cfg.MODEL.TRANSFER_TYPE
    if TRANSFER_TYPE in ["prompt","prompt+bias"]:
        NUM_TOKENS = cfg.MODEL.PROMPT.NUM_TOKENS
        DEEP = "deep" if cfg.MODEL.PROMPT.DEEP else "shallow"
        output_folder = os.path.join(TRANSFER_TYPE,cfg.DATA.FEATURE,"seed{}_lr{}_tokens{}_wd{}_{}".format(cfg.SEED,lr,NUM_TOKENS,wd,DEEP))
    elif TRANSFER_TYPE in ["linear","side"]:
        MLPNUM = cfg.MODEL.MLP_NUM
        output_folder = os.path.join(TRANSFER_TYPE,cfg.DATA.FEATURE,"seed{}_lr{}_mlpnum{}_wd{}".format(cfg.SEED,lr,MLPNUM,wd))
    elif TRANSFER_TYPE in ["end2end","partial-1","partial-2","partial-4","tinytl-bias"]:
        output_folder = os.path.join(TRANSFER_TYPE,cfg.DATA.FEATURE,"seed{}_lr{}_wd{}".format(cfg.SEED,lr,wd))
    elif TRANSFER_TYPE == "adapter":
        FACTOR = cfg.MODEL.ADAPTER.REDUCATION_FACTOR
        output_folder = os.path.join(TRANSFER_TYPE,cfg.DATA.FEATURE,"seed{}_lr{}_factor{}_wd{}".format(cfg.SEED,lr,FACTOR,wd))
    elif TRANSFER_TYPE == "lora":
        RANK = cfg.MODEL.LORA.RANK
        TUNE = "".join([key[5] for key,value in dict(cfg.MODEL.LORA).items() if 'TUNE' in key and value])
        output_folder = os.path.join(TRANSFER_TYPE,cfg.DATA.FEATURE,"seed{}_lr{}_rank{}_wd{}_{}".format(cfg.SEED,lr,RANK,wd,TUNE))
    
    output_path = os.path.join(output_dir, data_name, output_folder)
    cfg.OUTPUT_DIR = output_path
    
    if not PathManager.exists(output_path):
        PathManager.mkdirs(output_path)
        
    cfg.freeze()
    return cfg


def get_loaders(cfg, logger):
    logger.info("Loading training data (final training data for vtab)...")
    if cfg.DATA.NAME.startswith("vtab-"):
        train_loader = data_loader.construct_trainval_loader(cfg)
    else:
        train_loader = data_loader.construct_train_loader(cfg)

    logger.info("Loading validation data...")
    # not really needed for vtab
    val_loader = data_loader.construct_val_loader(cfg)
    logger.info("Loading test data...")
    if cfg.DATA.NO_TEST:
        logger.info("...no test data is constructed")
        test_loader = None
    else:
        test_loader = data_loader.construct_test_loader(cfg)
    return train_loader,  val_loader, test_loader


def train(cfg, args):
    # clear up residual cache from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # main training / eval actions here

    # fix the seed for reproducibility
    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(cfg.SEED)

    # setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger((cfg['DATA']['NAME']))

    train_loader, val_loader, test_loader = get_loaders(cfg, logger)
    logger.info("Constructing models...")
    model, cur_device = build_model(cfg)

    logger.info("Setting up Evalutator...")
    evaluator = Evaluator()
    evaluator.compute_f1 = args.compute_f1
    logger.info("Setting up Trainer...")
    trainer = Trainer(cfg, model, evaluator, cur_device)

    if args.eval:
        trainer.cls_weights = train_loader.dataset.get_class_weights(trainer.cfg.DATA.CLASS_WEIGHTS_TYPE)
        trainer.model.eval()
        if args.eval_dataset in ["test","val","train"]:
            logger.info(f'Testing the {args.eval_dataset} dataset...')
            trainer.eval_classifier(eval(args.eval_dataset+"_loader"), args.eval_dataset, False)
        else:
            logger.info(f"No {args.eval_dataset} loader presented. Exit")
    else:
        if train_loader:
            trainer.train_classifier(train_loader, val_loader, test_loader)
        else:
            logger.info("No train loader presented. Exit")


def main(args):
    """main function to call from workflow"""

    # set up cfg and args
    cfg = setup(args)

    # Perform training.
    train(cfg, args)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
