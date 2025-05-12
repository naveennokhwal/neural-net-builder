# 🔧 Neural Network Framework from Scratch (Python)

A fully customizable deep learning framework built from scratch in Python, featuring a core auto-differentiation engine (`Value`) for training neural networks. This project is designed for learning and rapid experimentation—users can define any feedforward architecture without relying on external ML libraries like PyTorch or TensorFlow.

---

## 💡 Why I Created This

The goal was to:
- Deeply understand how neural networks work internally.
- Gain hands-on experience with forward and backward propagation.
- Build a clean, reusable training and evaluation pipeline.
- Allow easy extension for custom models or datasets.

---

## ✨ Key Highlights

- ⚙️ **Custom `Value` class** for forward and backward passes  
- 🧠 Build **any neural network architecture**—you define the layers  
- 🔄 Full support for gradient computation using backpropagation  
- 📊 Includes an **example MLP model** trained on the Iris dataset  
- 🗂️ Modular and well-documented codebase  
- ✅ No deep learning libraries required

---

## 📁 Project Structure
```
neural-network-builder/
├── data/ # Raw and processed datasets
│ ├── raw/ # Original data files
│ └── processed/ # Cleaned and preprocessed data
├── src/ # Source code for the neural network
│ ├── init.py
│ ├── layers.py # Definitions of custom neural network layers
│ ├── model.py # Model architecture built using layers
│ ├── train.py # Script to train the model
│ ├── evaluate.py # Script to evaluate the trained model
│ ├── graph.py # Code to visualize the computational graph
│ ├── Value.py # Custom data type to enable flexible tensor operations
│ └── utils.py # Helper functions (e.g., data loading, preprocessing, metrics)
├── notebooks/ # Jupyter notebooks for experiments and exploration
│ └── exploration.ipynb
├── docs/ # Project documentation and usage guides
│ └── usage_guide.md
├── requirements.txt # List of Python dependencies
├── README.md # Project overview and setup instructions
└── .gitignore # Specifies files/folders to be ignored by Git
```
---

## 🧠 About the `Value` Class

At the heart of this framework is `value.py`, which implements a minimal autograd engine:

- Enables building and training neural networks **from scratch**
- Each mathematical operation tracks its **computation graph**
- Automatic gradient calculation via **backpropagation**
- Inspired by PyTorch’s dynamic computation graph (e.g., `torch.Tensor` with `.backward()`)

You can build your entire model on top of `Value`, making it highly extensible and educational.

---

## 🧪 Example: Iris Classification (MLP)

We’ve included a small demo model (`model.py`) that classifies the Iris dataset using a simple MLP.

> 📌 **Note**: This is **only a demo** to showcase how the code works.  
> You can plug in any dataset and define your own neural network architecture with any number of layers and neurons.

---

## 📦 Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/naveennokhwal/neural-network-builder.git
   cd neural-network-builder
   ```
Install dependencies:
  ```bash
    pip install -r requirements.txt
  ```
⚙️ How to Use
To train the example model:
  ```bash
    python train.py
  ```
To evaluate the saved model:
  ```bash
    python evaluate.py
  ```
### To build your own network:
- Edit model.py to define a new architecture.
- Tweak train.py for dataset/training config.
- Done!

### For full instructions, see `docs\usage_guide.md`
