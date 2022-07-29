from setuptools import setup, find_packages
__version__ = "3.0.0"

setup(
    name='pollen_id',
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    # install_requires=[
    #     'python-dotenv',
    #     'numpy',
    #     'pandas',
    #     'futures',
    #     'ffmpeg-python',
    #     'svgpathtools',
    #     'CairoSVG',
    #     'opencv_python',
    #     'networkx',
    #     'detectron2',
    #     'torch >= 1.4',
    #     'shapely',
    #     'torchvision',
    #     'sklearn']
)

exec(open('pollen_id/_version.py').read())
