# NoahQA

## Dataset Resouces

We uploaded the proposed dataset [here](https://drive.google.com/drive/folders/1-mQQ4j0qykGWAm46QCvrNy-aK077kQbp?usp=sharing).

In this folder, we provided resources in two languages, zh (chinese) and en (english).

## Model

We provided the code in two allennlp versions, 2.7.0 and 0.9.0, you can choose the version you want.

Firstly, we select the folder corresponding to the version and enter it.

### Installation

Development environment:
```
python == 3.6.0
# Requirements
pip install -r requirements.txt
```
Then, we downloaded the dataset and put it in the directory.

### Training the Model

We can edit the specific setting in configuration ''config_for_xxx.json''

```
allennlp train config_for_xxx.json -s xxx_dir --include-package xxx
```

the parameter and configuration of trained model will be saved in xxx_dir
