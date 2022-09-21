# Summary sentence extraction

## SciBERT binary classification model

**Python version**: This code is in Python3.7

**Package Requirements**:
```
pytorch==1.10.0
pandas==1.3.5
torchtext==0.11.0
transformers==4.18.0
matplotlib==3.5.1
scikit-learn==1.0.2
seaborn==0.11.2
```
## Dataset
[Scientific abstract sentence binary classification dataset](https://drive.google.com/file/d/1LKK3sK_BrhWZLMr8hiw7uMvMHvXcmH1h/view?usp=sharing)
## Trained Models
[SciBERT for summary sentence extraction](https://drive.google.com/file/d/1LKK3sK_BrhWZLMr8hiw7uMvMHvXcmH1h/view?usp=sharing)
## Run

The code runs in jupyter.

### Before running the code

* `source_folder` in **Parameters** is used to set the directory of the dataset.
* `destination_folder` in **Parameters** is used to set the directory of the optimal checkpoint and the training process log files.
* The cuda device number for training can be modified in **Libraries**.

```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```

### Training

Simply execute all code cells in sequence except for the **Evaluation** section.

### Evaluation

Simply execute all code cells in sequence except for the **Training** section.