from pollen_id.detector.ml_bundle import MLBundle
from detectron2.engine import DefaultTrainer
# from pollen_id.detector.trainer import Trainer
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-b", "--bundle")
args = vars(parser.parse_args())
BUNDLE_NAME = args["bundle"]

#VALIDATION_OUT_DIR = "validation_results"

def main(BUNDLE_NAME):

    mlb = MLBundle("bundles/"+BUNDLE_NAME)  
    cfg = mlb.config
    mlb.dataset.prepare()
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    


if __name__ == "__main__":
    main(BUNDLE_NAME)
