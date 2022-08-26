# pollen_id

Machine Learning bundle that contains custom Dataset and config files for pollen detection using Detectron2. It contains helpers that process raw SVG images to COCO-format JSONs that has annotations, segmentations that could represent the image.

## Installation

```bash
$ pip install pollen_id
```

## Usage

```python
from pollen_id.detector.ml_bundle import MLBundle

# creates the ML Bundle with config loaded on configs/ directory
bundle = MLBundle(SRC_PATH)

```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`pollen_id` was created by Tony Liang. It is licensed under the terms of the MIT license.

## Credits

`pollen_id` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
