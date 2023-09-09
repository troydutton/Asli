## ASL Identifier (ASLI)

Real-time classifier for ASL Gestures. Uses MediaPipe to generate hand landmarks which are fed into a follow-on classifier. 

## Initial Setup

1. Clone the repository: `git clone https://github.com/Bushvacka/Asli.git`
2. Create a pip environment: `python -m venv \venv\`
3. Activate the environment: `.\venv\Scripts\activate`
4. Modify CUDA version in `requirements.txt` to match with the installed toolkit 
5. Install requirements: `pip install -r requirements.txt`

## Dataset Setup

These instructions detail how to install the ASL Alphabet dataset. 
Other datasets can be used by creating a class which inherits from `torch.utils.data.Dataset`.

1. Install the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) to `\data\`
2. Remove the `SPACE`, `DELETE`, and `NOTHING` folders from `\data\` as they are unused 
3. Convert the dataset to landmarks by calling `generateLandmarkDataset` in `train.py`

## Usage

`train.py` contains functions necessary to train new models.

`main.py` runs the selected model in real-time, taking video input from the primary webcam device.

## Authors
Created during the 2022 HackTX hackathon for team Asli.
- Troy Dutton
- Akhil Giridhar
- Rohain Jain
- Jibran Cutlerywala
