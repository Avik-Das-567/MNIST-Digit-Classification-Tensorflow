# Basic Image Classification with TensorFlow

A deep learning project that builds, trains, and evaluates a fully connected neural network to classify handwritten digits from the **MNIST** dataset, achieving a **~96.2% accuracy** on the test set.

---

## Table of Contents

- [Project Objectives](#project-objectives)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Technologies Used](#technologies-used)

---

## Project Objectives

1. Learn to **create, train, and evaluate** neural network models with **TensorFlow** and **Keras**.
2. Understand the **fundamentals of neural networks**, including activation functions, weights, biases, and the forward pass.
3. Learn to solve **multi-class image classification** problems using neural networks.

By the end of this project, a neural network model is built that classifies images of handwritten digits (0–9) with a high degree of accuracy.

---

## Project Structure

The project is organized into **8 tasks**:

### Task 1: Introduction
- Overview of the handwritten digit classification problem.
- Introduction to **TensorFlow** (version `2.19.0`) and its role in building neural networks.

### Task 2: The Dataset
- Import the **MNIST** dataset via `tensorflow.keras.datasets.mnist`.
- Inspect the structure and shape of training and test arrays.
- Visualize sample images from the dataset using **Matplotlib**.

### Task 3: One Hot Encoding
- Explanation of **One Hot Encoding** and why it is needed for multi-class classification.
- Apply `tensorflow.keras.utils.to_categorical` to convert integer labels into one-hot vectors.
  - Example: label `5` → `[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]`
- Validate encoded shapes: `y_train_encoded`: `(60000, 10)`, `y_test_encoded`: `(10000, 10)`.

### Task 4: Neural Networks
- Visual explanation of **linear equations** as the basis of a single neuron: `y = W · X + b`.
- Explanation of **neural networks** as stacked layers of neurons capable of learning complex, non-linear mappings.
- Introduction to **activation functions** (ReLU and Softmax).

### Task 5: Pre-processing the Examples
- **Unrolling**: Flatten each 28×28 image into a 784-dimensional vector using `numpy.reshape`.
  - `x_train_reshaped`: `(60000, 784)`, `x_test_reshaped`: `(10000, 784)`.
- **Data Normalization**: Standardize pixel values using the global training mean and standard deviation (with a small `epsilon = 1e-10` for numerical stability):
  ```
  x_norm = (x - mean) / (std + epsilon)
  ```

### Task 6: Creating the Model
- Build a **Sequential** model with the **Keras API**.
- **Model Architecture**: Two hidden layers with ReLU activation and one output layer with Softmax.
- **Compile** the model with SGD optimizer and categorical cross-entropy loss.

### Task 7: Training the Model
- Train the model on normalized training data for **3 epochs**.
- Evaluate the trained model on the normalized test set.

### Task 8: Predictions
- Generate predictions on the full test set.
- **Visualize** a 5×5 grid of test image predictions with ground truth labels (correct predictions in green, incorrect in red).
- Plot the **probability distribution** (softmax output) for an individual test sample.

---

## Dataset

The **MNIST** (Modified National Institute of Standards and Technology) dataset is a benchmark dataset for image classification.

| Split      | Samples | Image Shape | Label Shape  |
|------------|---------|-------------|--------------|
| Training   | 60,000  | `(28, 28)`  | `(60000,)`   |
| Test       | 10,000  | `(28, 28)`  | `(10000,)`   |

- **Classes**: 10 (digits 0 through 9)
- **Pixel values**: Integers in the range `[0, 255]`

---

## Data Preprocessing

### 1. Flattening
Each 2D image of shape `(28, 28)` is reshaped into a 1D vector of 784 features:

```python
x_train_reshaped = np.reshape(x_train, (60000, 784))
x_test_reshaped  = np.reshape(x_test,  (10000, 784))
```

### 2. Normalization
Pixel values are standardized using the mean and standard deviation computed from the training set:

```python
x_mean = np.mean(x_train_reshaped)
x_std  = np.std(x_train_reshaped)
epsilon = 1e-10

x_train_norm = (x_train_reshaped - x_mean) / (x_std + epsilon)
x_test_norm  = (x_test_reshaped  - x_mean) / (x_std + epsilon)
```

### 3. Label Encoding
Integer labels are one-hot encoded to support categorical cross-entropy loss:

```python
from tensorflow.keras.utils import to_categorical

y_train_encoded = to_categorical(y_train)
y_test_encoded  = to_categorical(y_test)
```

---

## Model Architecture

A **fully connected (Dense) Sequential model** is built using the Keras API:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential([
    Input(shape=(784,)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10,  activation='softmax')
])
```

### Layer Summary

| Layer        | Type  | Output Shape | Parameters |
|--------------|-------|--------------|------------|
| Input        | —     | `(None, 784)` | 0          |
| Dense (ReLU) | Dense | `(None, 128)` | 100,480    |
| Dense (ReLU) | Dense | `(None, 128)` | 16,512     |
| Dense (Softmax) | Dense | `(None, 10)` | 1,290   |

- **Total parameters**: 118,282 (~462 KB)
- **Trainable parameters**: 118,282

### Compilation

```python
model.compile(
    optimizer='sgd',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## Training

The model is trained for **3 epochs** on the normalized training data:

```python
model.fit(x_train_norm, y_train_encoded, epochs=3)
```

### Training Log

| Epoch | Loss   | Accuracy |
|-------|--------|----------|
| 1     | 0.3648 | 89.31%   |
| 2     | 0.1814 | 94.68%   |
| 3     | 0.1380 | 95.90%   |

---

## Results

After 3 epochs, the model is evaluated on the **10,000 test samples**:

| Metric   | Value    |
|----------|----------|
| Test Loss | 0.1274  |
| **Test Accuracy** | **96.20%** |

### Prediction Visualization
- A **5×5 grid** of the first 25 test images is plotted with their predicted and ground truth labels. Correct predictions are shown in **green**, and incorrect ones in **red**.
- The **softmax probability distribution** for individual test samples is also visualized, showing the model's confidence across all 10 digit classes.

---

## Technologies Used

| Library      | Version | Purpose                                       |
|--------------|---------|-----------------------------------------------|
| Python       | 3.10.x  | Programming language                          |
| TensorFlow   | 2.19.0  | Deep learning framework                       |
| Keras        | (bundled with TF) | High-level neural network API      |
| NumPy        | —       | Numerical operations and array manipulation   |
| Matplotlib   | —       | Data visualization and prediction plots       |
