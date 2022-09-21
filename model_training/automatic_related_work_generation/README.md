# Automatic related work generation

**This code is used for the training and evaluation of text summarization models including BERTSumAbs and BART. The resulting models are used for automatic scientific review generation in the IdeaReader machine reading system.**


**Python version**: This code is in Python3.7

**Package Requirements**: 
```
nltk==3.7
numpy==1.21.5
pytorch_transformers==1.2.0
tensorboardX==2.5.1
torch==1.11.0
tqdm==4.64.0
transformers==4.19.2
```

Some codes are borrowed from PreSumm (https://github.com/nlpyang/PreSumm)

## Trained Models
[Download here](https://drive.google.com/file/d/1LKK3sK_BrhWZLMr8hiw7uMvMHvXcmH1h/view?usp=sharing)



## Data Preparation (Take the DELVE dataset as an example)
### Download the raw data

[Download here](https://drive.google.com/file/d/1LKK3sK_BrhWZLMr8hiw7uMvMHvXcmH1h/view?usp=sharing)

Unzip the zipfile and put all `.json` files into `raw_data/delve`

## Model Training (Take the DELVE dataset as an example)

### BERTSumAbs
You must first set `preprocess_dataset` in `config.py` to `'delve'` and `target_model` to `'bertsumabs'`.
#### Data Preprocessing
```
cd src
python preprocess.py
```
* You can change the `BertSumAbs_Preprocess_Args()` class in `config.py` as required.
#### Training
```
cd src
python train.py
```
* You can change the `BertSumAbs_Train_Args()` class in `config.py` as required.
* `self.mode` in class `BertSumAbs_Train_Args()` must be set to `'train'`.

**Attention:** For the first time, you should use single-GPU, so the code can download the BERT model. Set `self.visible_gpus` to `'-1'`, after downloading, you could kill the process and rerun the code with multi-GPUs.
#### Verification
```
cd src
python train.py
```
* You can change the `BertSumAbs_Train_Args()` class in `config.py` as required.
* `self.mode` in class `BertSumAbs_Train_Args()` must be set to `'validate'`.
* After the verification is completed, the top5 checkpoints will be selected by cross entropy to generate the text from the test set and store the corresponding files in the `results` folder along with the ground truth files.
#### Evaluation
```
cd src
python cal_rouge.py
```
* You need to set the path to the file to be evaluated in the `Rouge_Cal_Args()` class in `cal_rouge.py`.
* You can change the `Rouge_Cal_Args()` class as required.
### BART
You must first set `preprocess_dataset` in `config.py` to `'delve'` and `target_model` to `'bart'`, and other operations are the same as BERTSumAbs.

## Other
* This rep supports single-machine multi-GPUs training, gradient accumulation, and does not support multi-machine training.
* The hyperparameters we use during training and validation can be found in `config_copy.py`.
* The [S2ORC](https://drive.google.com/file/d/1LKK3sK_BrhWZLMr8hiw7uMvMHvXcmH1h/view?usp=sharing) dataset is also optional.