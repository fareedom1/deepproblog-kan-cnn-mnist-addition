# DeepProbLog: CNN vs. KAN on Symbolic MNIST Addition

This project compares a standard Convolutional Neural Network (CNN) and a Kolmogorov-Arnold Network (KAN) as neural backends in DeepProbLog — a framework that combines neural networks with logic programming. The task is symbolic MNIST digit addition.

The CNN serves as the baseline encoder, while the KAN model is a spline-based, interpretable neural architecture designed for high-performance function approximation. The KAN implementation in this project is adapted from the torch-conv-kan repository by Ivan Drokin.

DeepProbLog is a probabilistic logic programming framework that allows the integration of neural networks as probabilistic predicates in logic-based reasoning tasks. It supports both exact and approximate inference and is particularly useful for neurosymbolic learning applications.

---

## Quickstart

Clone the repository:  
git clone https://github.com/fareedom1/deepproblog-kan-cnn-mnist-addition.git  
cd deepproblog-kan-cnn-mnist-addition

Create a virtual environment:  
python -m venv venv

Activate the environment:  
On Mac/Linux: source venv/bin/activate  
On Windows: .\venv\Scripts\activate

Install all dependencies:  
pip install -r requirements.txt

---

## Running the Models

To run the CNN version:  
python addition.py

To run the KAN version:  
python addition_kan.py

Each script will train the model for 1 epoch, print average loss every 100 iterations, display the test set accuracy and confusion matrix, and ask if you'd like to continue training.

---

## Main Files

addition.py — trains the CNN model  
addition_kan.py — trains the KAN model  
deepproblog/examples/MNIST/models/addition.pl — logic program used for symbolic addition  
deepproblog/examples/MNIST/KANConv.py, KANLinear.py — spline-based KAN layers  
snapshot/ — saved model weights after each epoch  
log/ — training logs and accuracy summaries

---

## Dependencies

- Python 3.10 or 3.12  
- PyTorch 2.6.0  
- torchvision 0.21.0  
- scikit-learn  
- sympy  
- Pillow  
- pysdd  
- problog (installed from GitHub)

All dependencies are listed in requirements.txt and should install directly via pip with no special configuration.

---

## References

DeepProbLog GitHub: https://github.com/ML-KULeuven/deepproblog  
ProbLog engine: https://dtai.cs.kuleuven.be/problog  
KAN GitHub (torch-conv-kan): https://github.com/drokin/torch-conv-kan  
KAN Paper: https://arxiv.org/abs/2407.16674
