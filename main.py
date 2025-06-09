from sources.argparser import get_args

args = get_args()

""" Main file """
from sources.train  import *
from sources.inference   import *

args_to_function = {
    "train": train,
    "inference": inference
}

if __name__ == "__main__":
    args_to_function[args.action](args)