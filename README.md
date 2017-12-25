# ResNet-50 in Tensorflow
ResNet-50 model writtern entirely in tensorflow.

## Directory Structure
```
Project
|-- datasets
|   |-- dev_set
|   |-- test
|   |-- test_set
|   |-- train
|   `-- train_set
|-- models
|-- submissions
|-- datalab.py
|-- dataset_clusterer.py
|-- util.py
|-- make_file.py
|-- model.py
|-- predict.py
|-- test.py
`-- train.py
```

## Model
![](images/resnet-50.png)

## Usage
Run ```python dataset_clusterer.py``` to make batches of train data and test data and 
save them in ```./datasets/train_set``` and ```./datasets/train_set``` respectively.

Run ```python train.py``` to train the model and save it to ```./models/```

Run ```python predict.py``` to make probability predictions and save the output to ```./submissions/sub_1.csv```