```markdown
# 🧠 MNIST Digit Recognizer – NumPy vs TensorFlow

This project is a comparison of implementing a neural network to classify handwritten digits (MNIST dataset) using:
- **NumPy** (from scratch)
- **TensorFlow** (high-level library)

It was done as part of my deep learning self-learning journey to strengthen my understanding of both **theory and implementation**.

---

## 🔍 Problem Statement

Build a neural network to classify **28x28 grayscale images** of handwritten digits (0–9) from the **MNIST dataset**, and compare:
- Manual implementation (NumPy-based)
- Library-based approach (TensorFlow)

---

## 📁 Project Structure

```

mnist-digit-recognizer/
├── mnist\_numpy.py            # NumPy implementation
├── mnist\_tensorflow\.py       # TensorFlow implementation
├── mnist\_train.csv           # Training data (from Kaggle)
├── mnist\_test.csv            # Testing data (from Kaggle)
├── model.pkl                 # Trained model (NumPy version)
├── README.md                 # Project documentation
└── assets/
└── accuracy\_plot.png     # Accuracy/loss curves 

````

---

## ✅ Features Implemented

### 🔧 NumPy Version (From Scratch)
- Data loading & preprocessing
- Forward Propagation (ReLU, Softmax)
- Backward Propagation (Cross Entropy Loss)
- One-hot encoding
- Accuracy & cost tracking
- Weight updates using gradient descent

### ⚙️ TensorFlow Version
- Dense layers with ReLU and Softmax
- Adam optimizer and categorical crossentropy loss
- Model training & evaluation

---

## 📊 Results

| Metric     | NumPy Model | TensorFlow Model |
|------------|-------------|------------------|
| Accuracy   | ~88%        | ~92%             |
| Epochs     | 400         | 20               |
| Optimizer  | Manual GD   | Adam             |

---

## 💡 What I Learned

- How neural networks learn from data step-by-step
- How to manually code forward and backward propagation
- The math behind softmax and cross entropy
- How libraries like TensorFlow automate these operations efficiently
- Debugging and optimizing deep learning models

---

## 🚀 How to Run

### NumPy Model
```bash
python mnist_numpy.py
````

### TensorFlow Model

```bash
python mnist_tensorflow.py
```

Ensure that `mnist_train.csv` and `mnist_test.csv` are in the same directory.

---

## 📚 Dataset

MNIST dataset downloaded from [Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

---

## 📌 Future Improvements

* Add a web-based UI to upload digit images and test predictions
* Save trained model in `.pkl` and load via API
* Experiment with CNN (Convolutional Neural Network)
* Extend to Fashion-MNIST

---

## 🔗 Connect With Me

Feel free to connect on [LinkedIn](https://www.linkedin.com/in/nithish-kannan-m/) or check out more projects on [GitHub](https://github.com/NithishKannanM).

---

## 🏷️ Tags

`#MachineLearning` `#DeepLearning` `#NumPy` `#TensorFlow` `#MNIST` `#FromScratch` `#NeuralNetworks`

```
