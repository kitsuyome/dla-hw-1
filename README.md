# Text-to-Speech

## Objective

In this project explored the task of speech recognition by implementing the DeepSpeech2 model. The model was developed based on the paper ["Deep Speech 2 : End-to-End Speech Recognition in English and Mandarin"](http://proceedings.mlr.press/v48/amodei16.pdf). For training the model, the [Librispeech dataset](https://www.kaggle.com/datasets/a24998667/librispeech) was used.

## Report

[Wandb link](https://api.wandb.ai/links/kitsuyomi/r1280ing)

## Installation

Clone the repository and install dependencies:

```
!git clone https://github.com/kitsuyome/dla-hw-1
%cd dla-hw-1
!pip install -r requirements.txt
```

## Test

Run the setup script to download the model checkpoint and test the model by data in the 'test_data' directory.

```
!python setup.py
!python test.py \
   -c default_test_model/config.json \
   -r default_test_model/checkpoint.pth \
   -t test_data/data \
   -o test_result.json
```

## Reproduce Training

Run the script to reproduce training:

```
!python train.py -c /kaggle/working/dla-hw-1/hw_asr/configs/config.json
```

## Test audio

To explore examples of text recognition, please refer to the [Wandb report](https://api.wandb.ai/links/kitsuyomi/r1280ing)

## License

[MIT License](LICENSE)
