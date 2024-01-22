# Text-to-Speech

## Objective

In this project explored the task of speech synthesis by implementing the FastSpeech2 model. The model was developed based on the paper ["FastSpeech 2: Fast and High-Quality End-to-End Text to Speech"](http://keithito.com/LJ-Speech-Dataset). For training the model, the [LJSpeech dataset](http://keithito.com/LJ-Speech-Dataset) was used.

An attempt was made to reproduce the results of the research and achieve high-quality synthesized speech, which was quite successful. Experiments were also conducted in speech synthesis with different pitches, speeds, and energies. The test audios are included in the Wandb report.

## Report

[Wandb link]()

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
   -c /kaggle/working/hw_asr/default_test_model/config.json \
   -r /kaggle/working/hw_asr/default_test_model/checkpoint.pth \
   -t test_data/data \
   -o test_result.json
```

## Reproduce Training

Run the script to reproduce training:

```
!python train.py -c /kaggle/working/dla-hw-1/hw_asr/configs/config.json
```

## Test audio

To explore examples of text recognition, please refer to the [Wandb report]()

## License

[MIT License](LICENSE)
