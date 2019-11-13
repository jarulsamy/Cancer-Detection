import argparse
from keras.models import model_from_json
from cancer.classifiy import *

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=False,
                help="Path of model file", default="model.h5")

ap.add_argument("-j", "--json", required=False,
                help="Path to serialized json model file", default="model.json")

ap.add_argument("-F", "--force_train", required=False,
                help="Ignore already trained models and retrain", action="store_true")

ap.add_argument("-s", "--skip_examples", required=False,
                help="Skip post training examples", action="store_true")

ap.add_argument("-d", "--data", required=False,
                help="Path to images for classification.", default="Data/Original/")
args = vars(ap.parse_args())

if args["force_train"]:
    model = train(args["data"])
    post_train_examples(args["data"], model)
else:
    try:
        json_file = open(args["json"], "r")
    except FileNotFoundError:
        print("Json can't be opened!")
        print("Retraining...")
        model = train(args["data"])
        if not args["skip_examples"]:
            post_train_examples(args["data"], model)
        exit(0)

    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    try:
        loaded_model.load_weights(args["model"])
    except OSError:
        print("Model can't be opened!")
        print("Retraining...")
        model = train(args["data"])
        if not args["skip_examples"]:
            post_train_examples(args["data"], model)
        exit(0)

    print("Succesfully loaded pretrained model!")

    if not args["skip_examples"]:
        post_train_examples(args["data"], loaded_model)
