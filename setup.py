#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Tony Liang",
    author_email='chunqingliang@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Machine Learning bundle for detectron2 that contains custom dataset and configs for pollen detection.",
    entry_points={
        'console_scripts': [
            'pollen_id=pollen_id.cli:main',
        ],
    },
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pollen_id',
    name='pollen_id',
    packages=find_packages(include=['pollen_id', 'pollen_id.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/tonyliang19/pollen_id',
    version='0.1.0',
    zip_safe=False,
    install_requires=requirements
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
