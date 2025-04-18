import torch

from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.MNIST.data import MNIST_train, MNIST_test, addition
from deepproblog.examples.MNIST.network import MNIST_KAN_Net
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from json import dumps

# Configuration
method = "exact"
N = 1
name = f"addition_kan_{method}_{N}"

# Load dataset
train_set = addition(N, "train")
test_set = addition(N, "test")

# Use your KAN-based model
network = MNIST_KAN_Net()
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

# DeepProbLog model
#model = Model("models/addition.pl", [net])
model = Model("deepproblog/examples/MNIST/models/addition.pl", [net])
model.set_engine(ExactEngine(model), cache=True)
model.add_tensor_source("train", MNIST_train)
model.add_tensor_source("test", MNIST_test)

# DataLoader
loader = DataLoader(train_set, batch_size=2, shuffle=False)
# Epoch loop
MAX_EPOCHS = 10  # Safety cap
for epoch in range(MAX_EPOCHS):
    print(f"\n==================== EPOCH {epoch + 1} ====================")
    
    train = train_model(model, loader, 1, log_iter=100, profile=0)

    # Save snapshot after every epoch
    snapshot_path = f"snapshot/{name}_epoch{epoch + 1}.pth"
    model.save_state(snapshot_path)
    print(f"Model snapshot saved to {snapshot_path}")

    # Log hyperparameters and accuracy
    train.logger.comment(dumps(model.get_hyperparameters()))
    acc = get_confusion_matrix(model, test_set, verbose=1).accuracy()
    train.logger.comment(f"Accuracy {acc}")
    train.logger.write_to_file(f"log/{name}_epoch{epoch + 1}")

    # Ask user to continue or break
    user_input = input("Continue to next epoch? (y/n): ").strip().lower()
    if user_input != 'y':
        print("🛑 Training stopped by user.")
        break