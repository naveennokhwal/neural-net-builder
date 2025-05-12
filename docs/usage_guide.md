# 🧠 Usage Guide: Neural Network Framework from Scratch

This guide helps you understand and customize the deep learning framework. Whether you're experimenting with neural network architectures or just learning the core math behind training, this guide covers everything you need to get started.

---

## ⚙️ 1. Core Concept: The `Value` Class

`value.py` contains the heart of the framework: a custom `Value` class that mimics tensor-like behavior with support for automatic differentiation.

### 🔹 Key Features:
- Supports scalar operations: `+`, `-`, `*`, `/`, `**`, `tanh`, `exp`, `relu`, etc.
- Maintains a computation graph automatically
- Backpropagation with `.backward()` to compute gradients
- Each `Value` holds:
  - `data`: The scalar value
  - `grad`: The gradient after backpropagation
  - `_prev`: Set of parent nodes
  - `_op`: Operation used to generate this value
  - `label`: label for object

This class is **not tied to any model**, and all forward/backward logic flows from this class.

---

## 🧱 2. Neural Network Building Blocks: `layers.py`

This file provides modular components for constructing neural networks.

### 🔹 `Neuron` Class
- A single neuron with `nin` input weights and one bias.
- Computes output as `tanh(w·x + b)` using `Value` objects for all operations.

### 🔹 `Layer` Class
- A layer of multiple neurons.
- On a forward pass, each neuron in the layer processes the same input independently.

### 🔧 Why It Matters:
- This modular design allows building flexible, deep networks by stacking layers.
- Each neuron and layer is fully differentiable due to the use of `Value`.

---

## 🧠 3. Model Definition: `model.py`

This file defines the full model using layers from `layers.py`.

### 🔹 Example:
```python
model = MLP(nin=4, nouts=3)
```
### This means:
- Input layer with 4 features
- Output layer with 3 neurons (e.g., for 3-class classification)
- You can stack any number of layers with any number of neurons. The `MLP` class uses the `Layer` and `Neuron` classes internally.

### 🛠️ To Customize:

Edit model.py to define your own architecture using different nout values or new activation functions.

---

## 🔁 4. Training Loop: `train.py`

This script handles the training process.

### 🔹 What It Does:
- Loads and processes the dataset (Iris in the demo)
- Initializes the model
- Runs the forward pass, computes loss, and performs backpropagation using `.backward()`
- Updates weights using gradient descent

### 🛠️ Modify This File To:
- Use a different dataset
- Change the number of epochs, batch size, or learning rate
- Define a custom training strategy (momentum, etc.)
- You should keep the .backward() and manual weight updates as-is unless you're implementing advanced optimizers.

---
## 📊 5. Evaluation: `evaluate.py`
### A small script to:
- Load your trained model
- Run predictions on the test set
- Show accuracy or predictions

---
## 6. Utilities: `utils.py`

This module contains helper functions used throughout the project:

- **`csv_to_lists`**: Converts a CSV file into lists. This is necessary because the model expects data in list format.
- **`one_hot_encode_labels`**: Performs one-hot encoding on labels, essential for classification tasks.
- **`mse`** and **`cross_entropy_loss`**: Implements Mean Squared Error and Cross Entropy loss functions.
- **`softmax`**: Applies the softmax function to model predictions.
- **`train_test_split`**: Splits the dataset into training and testing sets.

## 🧠 7. How To Use This Framework
### ✅ Steps to Train Your Own Model:
- Prepare your dataset in CSV format.
- Edit model.py to define your architecture using the MLP class.
- Edit train.py to update:
    - Dataset path
    - Loss function
    - Training hyperparameters

- Train the model:
  ``` bash
  python train.py
  ```
- Evaluate the model:
  ``` bash
  python evaluate.py
  ```
