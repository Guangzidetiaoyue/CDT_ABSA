# Aspect-Level Sentiment Analysis Via Convolution over Dependency Tree
Requirements
---
* Python 3.6.8
* PyTorch 1.0.0
* CUDA 9.0

Preparation
---
First, download and unzip GloVe vectors(`glove.840B.300d.zip`) from https://nlp.stanford.edu/projects/glove/ .<br>
Then, put `glove.840B.300d.txt` into `./dataset/glove` directory. <br>
Next, prepare vocabulary with:<br>
```Bash
./build_vocab.sh
```
The `build_vocab.sh` contains commands to build vocab for different datasets.

Training
---
To train the CDT model, run: <br>
```Bash
./train.sh
```
The `train.sh` contains training commands for different datasets. <br>
Model checkpoints and logs will be saved to `./saved_models`. 

Evaluation
---
You need to specify `--data_dir`, `--vocab_dir` and `--model_dir` in `eval.py`. <br>
```Bash
python eval.py
```
