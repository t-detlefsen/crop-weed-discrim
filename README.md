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


## Pairwise Confusion Integration with ResNet

This implementation integrates Pairwise Confusion with a modified ResNet model that outputs both features and logits during the forward pass. To train the model with Pairwise Confusion, use the `config_pc.yaml` configuration, which specifies the ResNet variant supporting feature extraction.

### Loss Functions and Objectives

The Pairwise Confusion loss is computed in `trainer.py` through two additional loss functions:
- `PairwiseConfusion(features)`: Encourages feature compactness within a batch, helping to prevent overfitting.
- `EntropicConfusion(features)` : Counterbalances overconfidence by increasing the entropy of the model’s predictions, promoting a smoother and more robust output distribution.

These are combined with the standard classification loss (`CrossEntropyLoss`) to train the model. The classification loss drives the model to make accurate and confident predictions, while the Pairwise Confusion and Entropic Confusion losses regularize the features and predictions.

### Hyperparameters

- `lambda_pc`: Controls the weight of the Pairwise Confusion loss.
- `lambda_ec`: Controls the weight of the Entropic Confusion loss.

Both hyperparameters can be adjusted in `trainer.py` to tune the influence of these losses on the training process.
