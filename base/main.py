#from base.detector.ml_bundle import MLBundle
#from base.detector.trainer import Trainer
from argparse import ArgumentParser
from base.utils import MLScriptParser

ourParse = MLScriptParser(ArgumentParser)

# parser = ArgumentParser()
# parser.add_argument("-b", "--bundle")
# args = vars(parser.parse_args())
BUNDLE_NAME = 0 #args["bundle"]

def main(BUNDLE_NAME):
    print(BUNDLE_NAME)
    print("FIX THIS LATER !!! passing arguments like name of bundle , and config")
    #print(ourParse)
    #print(BUNDLE_NAME)
    #mlb = MLBundle(BUNDLE_NAME)
    #trainer = Trainer(mlb)
    #print(mlb.config)

if __name__ == "__main__":
    main(BUNDLE_NAME)