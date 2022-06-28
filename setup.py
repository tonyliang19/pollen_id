from setuptools import setup, find_packages
__version__ = "2.0.0"

setup(
    name='base',
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

exec(open('base/_version.py').read())