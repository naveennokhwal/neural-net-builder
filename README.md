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

neural-network-framework/
│
├── data/
│ └── processed/
│ └── Iris_noindex.csv # Example dataset
│
├── value.py # Core: Auto-diff engine (Value class)
├── model.py # Example: Define your own architecture here
├── train.py # Training loop (edit to match your design)
├── evaluate.py # Evaluation script
├── utils.py # Data preprocessing and evaluation helpers
├── requirements.txt # Required dependencies
├── README.md # Project overview (this file)
└── usage_guide.md # Full usage + customization instructions

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
   git clone https://github.com/your-username/neural-network-framework.git
   cd neural-network-framework
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
To build your own network:

Edit model.py to define a new architecture.

Tweak train.py for dataset/training config.

Done!

For full instructions, see usage_guide.md

📚 Requirements
pandas>=1.5.0
(Note: No need for NumPy or PyTorch)
