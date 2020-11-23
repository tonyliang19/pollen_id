from setuptools import setup, find_packages
exec(open('sticky_pi_ml/_version.py').read())

setup(
    name='sticky_pi_ml',
    version=__version__,
    long_description=__doc__,
    packages=find_packages(),
    scripts=['bin/universal_insect_detector.py'],
    include_package_data=True,
    zip_safe=False,
    install_requires=['numpy',
                      'pandas',
                      'futures',
                      'ffmpeg-python',
                      'svgpathtools',
                      'CairoSVG',
                      'opencv_python',
                      'networkx',
                      'detectron2',
                      'torch >= 1.4',
                      'shapely',
                      'torchvision',
                      'sticky_pi_api'],
    extras_require={
        'test': ['nose', 'pytest', 'pytest-cov', 'codecov', 'coverage'],
        'docs': ['mock', 'sphinx-autodoc-typehints', 'sphinx', 'sphinx_rtd_theme', 'recommonmark', 'mock']
    },
    test_suite='nose.collector'
)
