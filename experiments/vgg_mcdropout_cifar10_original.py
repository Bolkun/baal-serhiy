import pickle
import os
import sys #sys.exit()
import argparse
from pprint import pprint
import random
from copy import deepcopy
import csv
import datetime
import types

import pandas as pd
import numpy as np
import torch
import torch.backends
from torch import optim
from torch.hub import load_state_dict_from_url
from torch.nn import CrossEntropyLoss
from torchvision import datasets
from torchvision.models import vgg16
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from baal.active import get_heuristic, ActiveLearningDataset
from baal.active.active_loop import ActiveLearningLoop
from baal.bayesian.dropout import patch_module
from baal.modelwrapper import ModelWrapper
from baal.utils.metrics import Accuracy
from baal.active.heuristics import BALD
from baal.active.heuristics import Entropy

"""
Minimal example to use BaaL.
# pip install baal
# conda activate deepAugmentEnv
# cd experiments
# python vgg_mcdropout_cifar10_original.py 
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--initial_pool", default=1000, type=int) # we will start training with only 1000 labeled data samples out of the 50k and
    parser.add_argument("--query_size", default=100, type=int)    # request 100 new samples to be labeled at every cycle
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--heuristic", default="bald", type=str)
    parser.add_argument("--iterations", default=20, type=int)     # 20 sampling for MC-Dropout
    parser.add_argument("--shuffle_prop", default=0.05, type=float)
    parser.add_argument("--learning_epoch", default=20, type=int)
    parser.add_argument("--pretrained", default=0, type=int) # model pretraned 0 is false, 1 true
    return parser.parse_args()


def get_datasets(initial_pool):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )
    # Note: We use the test set here as an example. You should make your own validation set.
    train_ds = datasets.CIFAR10(
        ".", train=True, transform=transform, target_transform=None, download=True
    )
    test_set = datasets.CIFAR10(
        ".", train=False, transform=test_transform, target_transform=None, download=True
    )

    active_set = ActiveLearningDataset(train_ds, pool_specifics={"transform": test_transform})

    # We start labeling randomly.
    active_set.label_randomly(initial_pool)
    return active_set, test_set


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    random.seed(1337)
    torch.manual_seed(1337)
    if not use_cuda:
        print("warning, the experiments would take ages to run on cpu")

    now = datetime.datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%Hx%M")
    csv_filename = "uncertainties/org_metrics_cifarnet_" + dt_string + "_.csv"
    with open(csv_filename, "w+", newline="") as out_file:
      csvwriter = csv.writer(out_file)
      csvwriter.writerow(
        (
          "epoch",
          "test_acc",
          "train_acc",
          "test_loss",
          "train_loss",
          "Next training size",
        )
      )
    
    hyperparams = vars(args)

    active_set, test_set = get_datasets(hyperparams["initial_pool"])

    heuristic = get_heuristic(hyperparams["heuristic"], hyperparams["shuffle_prop"])
    criterion = CrossEntropyLoss()

    if (hyperparams["pretrained"] == 1):
        model = vgg16(num_classes=10) 
        weights = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth")
        weights = {k: v for k, v in weights.items() if "classifier.6" not in k}
        model.load_state_dict(weights, strict=False)
    else:
        model = vgg16(pretrained=False, num_classes=10) # nicht trainierte model

    # change dropout layer to MCDropout
    model = patch_module(model)

    if use_cuda:
        model.cuda()
    else: 
        print("WARNING! NO CUDA IN USE!")
    optimizer = optim.SGD(model.parameters(), lr=hyperparams["lr"], momentum=0.9)

    # Wraps the model into a usable API.
    model = ModelWrapper(model, criterion)
    model.add_metric(name='accuracy', initializer=lambda : Accuracy())

    logs = {}
    logs["epoch"] = 1

    # for prediction we use a smaller batchsize
    # since it is slower
    active_loop = ActiveLearningLoop(
        active_set,
        model.predict_on_dataset,
        heuristic,
        hyperparams.get("query_size", 1),
        batch_size=10,
        iterations=hyperparams["iterations"],
        use_cuda=use_cuda,
    )
    # We will reset the weights at each active learning step.
    init_weights = deepcopy(model.state_dict())

    layout = {
        "Loss/Accuracy": {
            "Loss": ["Multiline", ["loss/train", "loss/test"]],
            "Accuracy": ["Multiline", ["accuracy/train", "accuracy/test"]],
        },
    }

    writer = SummaryWriter("vgg_mcdropout_cifar10_original")    # baal-serhiy/experiments/vgg_mcdropout_cifar10_original
    writer.add_custom_scalars(layout)

    for epoch in tqdm(range(args.epoch)):
        # if we are in the last round we want to train for longer epochs to get a more comparable result
        if epoch == (args.epoch - 1):
            hyperparams["learning_epoch"] = 75
        # Load the initial weights.
        model.load_state_dict(init_weights)
        model.train_on_dataset(
            active_set,
            optimizer,
            hyperparams["batch_size"],
            hyperparams["learning_epoch"],
            use_cuda,
        )

        # Validation!
        model.test_on_dataset(test_set, hyperparams["batch_size"], use_cuda)
        metrics = model.metrics

        # get origin amount of labelled augmented/unaugmented images
        if(epoch == 0):
          with open(csv_filename, "a+", newline="") as out_file:
            csvwriter = csv.writer(out_file)
            csvwriter.writerow(
              (
                -1,
                0,
                0,
                0,
                0,
                active_set.n_labelled,
              )
            )

        should_continue = active_loop.step()
        oracle_indices = 0
        uncertainty = 0

        # last epoch calculate the uncertainty
        if ((epoch+1)-args.epoch) == 0:
            print(str(epoch) + " out of " + str(args.epoch))
            predictions = model.predict_on_dataset(
                active_set._dataset,
                batch_size=hyperparams["batch_size"],
                iterations=hyperparams["iterations"],
                use_cuda=use_cuda,
            )
            uncertainty = active_loop.heuristic.get_uncertainties(predictions)
            # save uncertainty and label map to pkl
            oracle_indices = np.argsort(uncertainty)
            active_set.labelled_map
            # convert to string
            date_time_str = now.strftime("org_%Y-%m-%d_%H-%M-%S")
            uncertainty_filename = date_time_str + (
                f"_uncertainty_epoch={epoch}" f"_labelled={len(active_set)}.pkl"
            )
            newPath = os.path.join(os.getcwd(), "uncertainties")
            isExist = os.path.exists("uncertainties")
            if not isExist:
                os.makedirs(newPath)
            uncertainty_path = os.path.join(newPath, uncertainty_filename)
            print("Saving file " + uncertainty_path)
            pickle.dump(
                {
                    "oracle_indices": oracle_indices,
                    "uncertainty": uncertainty,
                    "labelled_map": active_set.labelled_map,
                },
                open(uncertainty_path, "wb")
            )
            # generate excel file
            mypickle = pd.read_pickle(uncertainty_path)
            excel_filename = date_time_str + (
                f"_uncertainty_epoch={epoch}" f"_labelled={len(active_set)}.xlsx"
            )
            excel_path = os.path.join(newPath, excel_filename)

            uncertainty = mypickle['uncertainty']
            oracle_indices = mypickle['oracle_indices']
            labelled_map = mypickle['labelled_map']

            uncertainty_length = len(uncertainty)

            original = uncertainty[0:50000-1]

            matrix = np.vstack([original])

            df_lab_img = pd.DataFrame(matrix)

            uncertainties_std = df_lab_img.transpose()
            uncertainties_std.columns = ['original']
            uncertainties_std.to_excel(excel_path)

        if not should_continue:
            break

        #print(metrics.keys()) #prints keys
        #print(metrics.values()) #prints values

        train_accuracy = metrics["train_accuracy"].value
        test_accuracy = metrics["test_accuracy"].value
        train_loss = metrics["train_loss"].value
        test_loss = metrics["test_loss"].value

        logs = {
            "epoch": epoch,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_loss": train_loss,
            "test_loss": test_loss,
            #"labeled_data": active_set.labelled_map,
            "oracle_indices": oracle_indices,           # array([ 54130, 109117,   8053, ..., 146481,  32710,  16780])
            "uncertainty": uncertainty,                 # array([0.1608387 , 0.00041478, 0.00027927, ..., 0.01162097, 0.04465848, 0.00282653], dtype=float32)}
            "labelled_map": active_set.labelled_map,    # array([0, 0, 0, ..., 0, 0, 0]),
            "Next Training set size": len(active_set),
        }

        pprint(logs)

        with open(csv_filename, "a+", newline="") as out_file:
            csvwriter = csv.writer(out_file)
            csvwriter.writerow(
                (
                epoch,
                test_accuracy,
                train_accuracy,
                test_loss,
                train_loss,
                active_set.n_labelled,
                )
            )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/test", test_loss, epoch)
        writer.add_scalar("accuracy/train", train_accuracy, epoch)
        writer.add_scalar("accuracy/test", test_accuracy, epoch)
        #writer.add_scalar("uncertainty", uncertainty, epoch)
    writer.close()


if __name__ == "__main__":
    main()
