# Edible Carrot or Poisonous Hemlock?
A crop-weed discrimination project for 16-824 Visual Learning &amp; Recognition

## Installation
Install the required python packages:

```
pip install -r requirements.txt
```

## Training
To train a new model, modify the hyperparameters in [config.yaml](config.yaml) and run the following command:

```
python train.py
```

This will create a new experiment in the `exp` directory containing:
- `config.yaml` - A copy of the `config.yaml` used for training
- `events.out.tfevents` - A log of training statistics viewable via tensorboard
- `weights.pth` - The weights from the best performing epoch
- `confusion_matrix.png` - A confusion matrix run on the test data

## Testing
To test a previously trained model, run the following command where `N` is the experiment you would like to test.

```
python test.py exp<N>
```

This will return the accuracy of the model and overwrite `confusion_matrix.png`, the confusion matrix run on the test data. Testing will use the `config.yaml` specified in the experiment