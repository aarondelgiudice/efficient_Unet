import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt


def dir2csv(path, output, file_extension=None):
    sub_dirs = glob.glob(f"{path}/*")

    for dir in sub_dirs:
        if file_extension is not None:
            filepaths = glob.glob(f"{dir}/*{file_extension}")
        else:
            filepaths = glob.glob(f"{dir}/*")

        for fp in filepaths:
            with open(output, "a") as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow([fp])
    return


def print_params(**params):
    print("parameters:")
    for k, v in params.items():
        if type(v) == str:
            if os.path.exists(v):
                print(f"{k}='{os.path.abspath(v)}'")
            else:
                print(f"{k}='{v}'")
        else:
            print(f"{k}={v}")
    return 


def read_csv(fp):
    with open(fp, newline='') as f:
        reader = csv.reader(f)
        data = [i[0] for i in reader]
    return data


def parse_args(defaults, args=None):
    p = argparse.ArgumentParser(description=sys.argv[0], add_help=False)

    # add any missing required arguments
    required_args = {
        "logging": 1,
        "message": "",
        "verbose": 0,
    }

    for k, v in required_args.items():
        if k not in defaults.keys():
            p.add_argument(
                f"--{k}", default=v,
                type=type(v) if v is not None else str)

    # add default arguments
    for k, v in defaults.items():
        p.add_argument(
            f"--{k}", default=v,
            type=type(v) if v is not None else str)

    # parse CLI arguments if provided, if not use defaults
    if args is not None:
        p = p.parse_args(args)
    else:
        p = p.parse_args()

    # return dictionary
    return vars(p)


def parse_filepath(fp, from_tensor=True):
    """
    Parse path and file from input array or tensor and return as strings.
    ARGS
        fp, array or tensor
        from_tensor, bool: set to True if input `fp` is from tensor

    RETURNS
        (path, file), tuple(string)
    """
    if from_tensor:
        path, file = os.path.split(fp.numpy()[0].decode("utf-8"))
    else:
        path, file = os.path.split(fp[0].decode("utf-8"))
    return path, file


def read_csv(fp):
    with open(fp, newline='') as f:
        reader = csv.reader(f)
        data = [i[0] for i in reader]
    return data


def visualize(savefig=None, title=None, **kwargs):
    plt.figure(figsize=(8 * len(kwargs), 10))

    for i, (k, v) in enumerate(kwargs.items()):
        plt.subplot(1, len(kwargs.keys()), i + 1)
        plt.title(k.replace("_", " ").title())
        plt.imshow(v)
        plt.axis('off')

    if title is not None:
        plt.suptitle(title.title())

    if savefig is not None:
        plt.savefig(savefig)

    plt.show()
    plt.close()
    return
