# Fashion-MNIST Dialog Summary

During our dialog, we built eight Python scripts that form an incremental
Fashion-MNIST classification workflow. Each stage reuses the modules from the
earlier stages, progressing from data preparation and model training to result
inspection, visualization, and hyperparameter optimization.

## 1. `script1_load_fashion_mnist.py` — Load and split the data

This script downloads the Fashion-MNIST training and test sets into the local
`datasets` directory and converts the images to PyTorch tensors with values in
the range `[0, 1]`. It uses a generator seeded with `42` to reproducibly divide
the original training set into 55,000 training samples and 5,000 validation
samples. The reusable `load_fashion_mnist` function returns the training,
validation, and test datasets; running the script directly prints their sizes.

## 2. `script2_inspect_fashion_mnist_data.py` — Create data loaders

This script wraps the three datasets in `DataLoader` objects using batches of
32. Training data is shuffled with its own seeded generator so the initial
batch order is reproducible, while validation and test data remain unshuffled.
When run directly, the script inspects the first item in the training subset and
prints its tensor shape, data type, numeric target, and human-readable class
name.

## 3. `script3_image_classifier.py` — Define the neural network

This script defines `FashionMNISTClassifier`, a fully connected PyTorch model.
It flattens each 28-by-28 grayscale image, sends the 784 input values through
hidden layers of 300 and 100 neurons with ReLU activations, and returns logits
for the 10 Fashion-MNIST classes. A single optional hidden-layer size lets the
tuning scripts use the same width for both hidden layers. The module also seeds
PyTorch, selects CUDA when available (otherwise CPU), creates the baseline model,
and configures cross-entropy loss.

## 4. `script4_train_model.py` — Train and validate the classifier

This script implements the reusable `train_model` function and configures a
baseline run of 10 epochs with stochastic gradient descent and a learning rate
of `0.1`. Each epoch performs forward propagation, backpropagation, and optimizer
updates over the training loader before evaluating the model on the validation
loader without gradients. It prints the average training loss, training
accuracy, and validation accuracy and returns all three metrics in a history
dictionary for use by later scripts.

## 5. `script5_evaluate_model.py` — Examine model predictions

This script trains the baseline classifier and then evaluates the first batch
from the validation loader. It applies softmax to the model logits, reports the
predicted class for the fourth image, and prints each image's predicted class,
rounded class-probability tensor, and four most probable classes. It also counts
and displays the model's total number of parameters. Inference runs in evaluation
mode without gradient tracking, and MPS results are moved to the CPU before
display when necessary.

## 6. `script6_plot_accuracy.py` — Plot learning progress

This script trains the classifier and passes the returned metric history to
`plot_accuracy`. The function uses Matplotlib to display training and validation
accuracy for every epoch on the same chart. The plot includes labeled axes, a
title, legend, grid, tight layout, and a fixed accuracy range from zero to one.

## 7. `script7_optuna.py` — Tune hyperparameters

This script uses Optuna to search for a better learning rate and hidden-layer
width. A seeded TPE sampler runs five reproducible trials, each training for 10
epochs. The learning rate is sampled logarithmically from `1e-5` to `1e-1`, and
the shared hidden-layer size is sampled from 20 to 300 neurons. A trial's score
is its best validation accuracy, and the completed study prints the best
parameters and score.

## 8. `script8_optuna_pruning.py` — Add early pruning

This script extends the Optuna workflow to 20 trials and adds a median pruner.
It trains each sampled model one epoch at a time, reports validation accuracy to
Optuna after every epoch, and raises `TrialPruned` when Optuna determines that a
trial is unpromising. Completed trials return their best validation accuracy,
and the study reports the best learning rate, hidden-layer size, and score. This
approach avoids spending all 10 epochs on weak configurations.

## Complete workflow

Together, the scripts provide the following sequence:

1. Download and reproducibly split Fashion-MNIST.
2. Create consistent mini-batch data loaders and inspect a sample.
3. Define a baseline fully connected classifier.
4. Train the classifier and record loss and accuracy.
5. Inspect validation predictions and probability rankings.
6. Visualize training and validation accuracy.
7. Tune learning rate and hidden-layer width with Optuna.
8. Accelerate the search by pruning unpromising trials.

Because later scripts import functions, classes, and configuration from earlier
ones, the project avoids duplicating its core data-loading, modeling, and
training logic.
