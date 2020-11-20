# Slurm jobs (e.g. on Compute Canada)

## Install package on a virtual environment
More info on [CC's official doc](https://docs.computecanada.ca/wiki/Python#Creating_and_using_a_virtual_environment)
```sh
module load python/3.7
module load scipy-stack
module load cuda cudnn


# ADD YOUR OWN VALUES HERE
VIRT_ENV_LOCATION=~/projects/<account_name>/<user_name>/sticky_pi_virt_env/
virtualenv --no-download ${VIRT_ENV_LOCATION}
source ${VIRT_ENV_LOCATION}/bin/activate
pip install --no-index --upgrade pip


# we install dependencies by hand :(

# sticky pi client

pip install python-dotenv passlib numpy pandas requests  boto3 pillow psutil joblib sqlalchemy typeguard itsdangerous --no-index
# fail to compile 0.7.4
pip install imread==0.7.0
# this is from branch develop `@develop`
pip install "git+https://github.com/sticky-pi/sticky-pi-api@develop#egg=sticky_pi_api&subdirectory=src" --no-index  --no-deps


# Note that detectron2 is already built as a wheel in ComputeCanada
pip install torch==1.5 pillow-simd detectron2  torchvision==0.6 pycocotools 

pip install opencv  svgpathtools CairoSVG networkx shapely==1.6.4.post2 --no-index
pip install "git+https://github.com/sticky-pi/sticky-pi-ml@develop#egg=sticky_pi_ml&subdirectory=src" --no-index  --no-deps
pip install ipython
```
