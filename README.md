# Edible Carrot or Poisonous Hemlock?
A crop-weed discrimination project for 16-824 Visual Learning &amp; Recognition

Crop-weed discrimination is a critical challenge as weeds reduce yield while often being difficult to discriminate from crops by a non domain expert. This repository implements Pairwise Confusion for [Fine-Grained Visual Classification](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Abhimanyu_Dubey_Improving_Fine-Grained_Visual_ECCV_2018_paper.pdf) on the [Plant Seedlings Dataset](https://vision.eng.au.dk/plant-seedlings-dataset/#citation).

## Installation
Install the required python packages:

```
pip install -r requirements.txt
```

Download the [Plant Seedlings Dataset](https://vision.eng.au.dk/?download=/data/WeedData/NonsegmentedV2.zip) and place in the `data` folder

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

This implementation integrates Pairwise Confusion with a modified ResNet model that outputs both features and logits during the forward pass.

### Loss Functions and Objectives

The Pairwise Confusion loss is computed in [utils.py](crop_weed_discrim/utils/utils.py) through two additional loss functions:
- `PairwiseConfusion(features)`: Encourages feature compactness within a batch, helping to prevent overfitting.
- `EntropicConfusion(features)` : Counterbalances overconfidence by increasing the entropy of the model’s predictions, promoting a smoother and more robust output distribution.

These are combined with the standard classification loss (`CrossEntropyLoss`) to train the model. The classification loss drives the model to make accurate and confident predictions, while the Pairwise Confusion and Entropic Confusion losses regularize the features and predictions.

### Hyperparameters

- `lambda_pc`: Controls the weight of the Pairwise Confusion loss.
- `lambda_ec`: Controls the weight of the Entropic Confusion loss.

Both hyperparameters can be adjusted in [config.yaml](config.yaml) to tune the influence of these losses on the training process.
