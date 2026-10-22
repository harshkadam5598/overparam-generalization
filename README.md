# An Empirical Study of Generalization in Overparameterized Neural Network

## Motivation
Understanding generalization behavior in overparameterized models is a central topic
in modern machine learning theory. While theoretical results provide important insights,
empirical studies help illustrate how model size and regularization affect learning behavior
in practice.

## Research Question
How do network width and weight decay influence training–test generalization behavior
in neural networks?

## Research Positioning
This project is an empirical, simulation-based study inspired by theoretical work on
generalization in overparameterized models. The goal is to investigate these effects
numerically through controlled experiments rather than to provide new theoretical guarantees.

## Experimental Setup
- Dataset: MNIST (handwritten digits)
- Model: Fully connected neural network (MLP) with ReLU activations
- Variables of interest:
  - Network width
  - Weight decay (L2 regularization)
- Fixed settings:
  - Optimizer: Adam
  - Learning rate: 1e-3
  - Batch size: 128
  - Number of epochs: 50

## Methodology
We conduct controlled experiments by varying one parameter at a time while keeping
all other settings fixed. For each configuration, models are trained from scratch and
evaluated on both training and test data.

This repository contains an independent, empirical study exploring generalization
behavior in overparameterized neural networks through controlled experiments.


## Evaluation Metrics
- Training loss
- Test loss
- Training accuracy
- Test accuracy
- Generalization gap (training accuracy minus test accuracy)

## Results
Results will be presented through plots and tables showing trends as a function of
network width and regularization strength.

## Limitations
This study is limited to a simple neural network architecture and a single dataset.
The findings are empirical in nature and do not constitute theoretical proofs or
state-of-the-art performance claims.

## Observations
- Increasing network width improves test accuracy up to a saturation point.
- Larger models tend to exhibit a higher training–test gap, indicating increased overfitting.
- Weight decay consistently reduces the generalization gap across network widths.
- Stronger regularization improves stability but can slightly reduce peak test accuracy.
