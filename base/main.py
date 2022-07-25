from base.detector.ml_bundle import MLBundle
from base.detector.trainer import Trainer
from argparse import ArgumentParser
#from base.utils import MLScriptParser

parser = ArgumentParser()
parser.add_argument("-b", "--bundle")
args = vars(parser.parse_args())
BUNDLE_NAME = args["bundle"]

def main(BUNDLE_NAME):
    mlb = MLBundle(BUNDLE_NAME)
    trainer = Trainer(mlb)
    print(trainer.config)

if __name__ == "__main__":
    main(BUNDLE_NAME)