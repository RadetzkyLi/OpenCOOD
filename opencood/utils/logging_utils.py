# -*- coding: utf-8 -*-
# Author: Rongsong Li <rongsong.li@qq.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Logging related utility functions
"""
import os
import logging
from matplotlib import pyplot as plt

def init_logging(save_dir=".", debug=False):
    """
    Initialize the logging.

    Parameters
    ----------
    save_dir: str
        The log file will be saved in "{save_dir}/training_log.log".

    debug: bool
        If set the logging level will be 'DEBUG', otherwise 'INFO'.

    Returns
    -------
    logger:
        The instance for logging.
    
    """
    level = logging.DEBUG if debug else logging.INFO
    # by name "training", we can get the same logger from anywhere
    logger = logging.getLogger("training")
    logger.setLevel(level)

    # formatter
    ffmt = logging.Formatter(
        fmt = "[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S"
    )
    sfmt = logging.Formatter(
        fmt = "[%(levelname)s][%(filename)s:%(lineno)d] %(message)s"
    )

    save_path = os.path.join(save_dir, "training_log.log")

    # file handler
    fh = logging.FileHandler(filename=save_path, encoding="utf8")
    fh.setLevel(level)
    fh.setFormatter(ffmt)
    logger.addHandler(fh)

    # stream handler
    sh = logging.StreamHandler()
    sh.setLevel(level = logging.INFO)
    sh.setFormatter(sfmt)
    logger.addHandler(sh)

    return logger


def draw_loss_figure(loss_train, loss_val, save_dir=".", init_epoch=0):
    """
    Draw loss changes over epochs.

    Parameters
    ----------
    loss_train: array-like
        Total training loss at each epoch.

    loss_val: array-like
        Total validation loss at each epoch.

    save_dir: str
        The resulting figure will be save in "{save_dir}/loss.png".

    Returns
    -------
    None
    """
    x = list(range(init_epoch, len(loss_train)+init_epoch))

    plt.plot(x, loss_train, color='r', label='train')
    plt.plot(x, loss_val, color='g', label='val')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    save_path = os.path.join(save_dir, "loss.png")
    plt.savefig(save_path)
    plt.close()