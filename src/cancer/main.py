import argparse

from ANN import *
from keras.models import model_from_json

ap = argparse.ArgumentParser()
ap.add_argument(
    "-m", "--model", required=False, help="Path of model file", default="model.h5"
)

ap.add_argument(
    "-j",
    "--json",
    required=False,
    help="Path to serialized json model file",
    default="model.json",
)

ap.add_argument(
    "-F",
    "--force_train",
    required=False,
    help="Ignore already trained models and retrain",
    action="store_true",
)

ap.add_argument(
    "-s",
    "--skip_examples",
    required=False,
    help="Skip post training examples",
    action="store_true",
)

ap.add_argument(
    "-td",
    "--train_data",
    required=False,
    help="Path to training images for classification.",
    default="Data/train/",
)

ap.add_argument(
    "-vd",
    "--val_data",
    required=False,
    help="Path to validation images for classification.",
    default="Data/test/",
)
args = vars(ap.parse_args())

if args["force_train"]:
    model = train(args["train_data"], args["val_data"])
    post_train_examples(args["train_data"], model)
else:
    # Try to open json file
    # Retrain if fail
    try:
        json_file = open(args["json"], "r")
    except FileNotFoundError:
        print("Json can't be opened!")
        print("Retraining...")
        model = train(args["train_data"], args["val_data"])
        if not args["skip_examples"]:
            post_train_examples(args["train_data"], model)
        exit(0)

    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # Try to load model weights
    # Retrain if fail
    try:
        loaded_model.load_weights(args["model"])
    except OSError:
        print("Model can't be opened!")
        print("Retraining...")
        model = train(args["train_data"], args["val_data"])
        if not args["skip_examples"]:
            post_train_examples(args["val_data"], model)
        exit(0)

    print("Succesfully loaded pretrained model!")

    if not args["skip_examples"]:
        post_train_examples(args["val_data"], loaded_model)
