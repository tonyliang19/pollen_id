from inferencer import Inferencer
from argparse import ArgumentParser
import time
parser = ArgumentParser()
parser.add_argument("-b", "--bundle")
args = vars(parser.parse_args())
BUNDLE_NAME = args["bundle"]


def main(BUNDLE_NAME):
    inferencer = Inferencer(bundle_name="bundles/raw/"+BUNDLE_NAME)
    # Train data paths, Test data paths
    train, test = inferencer.generate_train_test_split()

    # Make predictions
    # train_predictions = inferencer.make_predictions(dataset=train)
    # test_predictions = inferencer.make_predictions(dataset=test)
    
    # Save them to jpgs
    
    # Print train predictions
    print("This is train set predictions")

    
    inferencer.save_predictions_jpg(dataset=train)
    time.sleep(2)
    print("This is the end of train set predictions")
    time.sleep(10)

    # Print test predictions
    print("This is test set predictions")
    time.sleep(2)
    inferencer.save_predictions_jpg(dataset=test)
    print("This is the end of test set predictions")
    # # Visualize train predictions 
    # inferencer.visualize_with_plt(train, opt="train")

    # # Visualize test set predictions
    # inferencer.visualize_with_plt(test, opt="val")


if __name__ == "__main__":
    main(BUNDLE_NAME)
