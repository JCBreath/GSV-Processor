# GSV-Processor
The content in this repository only serves for researches in [BEDA Lab](https://bedalab.nd.edu/).

## Requirements
This processor runs on **Linux** with **Python 3.10** and **PyTorch 1.13 + CUDA 11.6**

The minimum GPU VRAM requirement is 4 GB.

Please run the following commands to install Conda environment and Python packages.
```
conda create -n gsv python=3.10
conda activate gsv
conda install -c nvidia/label/cuda-11.6.2 cuda-toolkit
python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install tqdm numpy==1.24 pillow opencv-python
```

## Usage
Run `python main.py --state {state_name} --building_detect --wallmat --condition --verbose`

For example, `python main --state Indiana --building_detect --wallmat --condition --verbose` will process all counties in Indiana.

Processing options:
* `--state` - Set state
* `--group_start` - Set starting group
* `--group_end` - Set ending group (inclusive)
* `--selected_county` - Set target county (will skip other counties)
* `--section_start` - Set starting section in the `selected_county`
* `--section_end` - Set ending section in the `selected_county`
* `--verbose` - Show progress
* `--skip` - Detect and skip processed sections

Task options:
* `--building_detect` - Detect and crop buildings
* `--wallmat` - Classify wall material
* `--condition` - Rate building condition

## Available Models
- Building Detector: [checkpoint](https://drive.google.com/file/d/1e1KLGYoHGR989Eeeo_oQ863bD98oSojA/view?usp=sharing)
  - Function: `BuildingProcessor.detect_and_crop()`
    - Input: `PIL.Image`
    - Output: `PIL.Image`
- Wall Material Classifier: [checkpoint](https://drive.google.com/file/d/1Bf9c4JELZrUsXgviNL-2cVB3pnPjwMnz/view?usp=sharing)
  - Function: `BuildingProcessor.get_wallmat()`
    - Input: `PIL.Image`
    - Output: `int` (index of item in classes: `'brick', 'other', 'wood', 'wood & brick'`)
- Building Condition Predictor: [checkpoint](https://drive.google.com/file/d/1EFM8vFHNTU1NMDyXl9HQQFZRMflmuhMq/view?usp=sharing)
  - Function: `BuildingProcessor.get_condition()`
    - Input: `PIL.Image`
    - Output: `int` (score from 1 to 5)

## Future Models
- Window-to-Wall Ratio Predictor
- External Shading Detector

