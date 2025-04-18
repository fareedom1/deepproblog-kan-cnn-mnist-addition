from json import dumps

import torch

from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.MNIST.data import MNIST_train, MNIST_test, addition
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model

method = "exact"
N = 1

name = "addition_{}_{}".format(method, N)

train_set = addition(N, "train")
test_set = addition(N, "test")

network = MNIST_Net()

pretrain = 0
if pretrain is not None and pretrain > 0:
    network.load_state_dict(torch.load("models/pretrained/all_{}.pth".format(pretrain)))
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

#model = Model("models/addition.pl", [net])
model = Model("deepproblog/examples/MNIST/models/addition.pl", [net])

if method == "exact":
    model.set_engine(ExactEngine(model), cache=True)
elif method == "geometric_mean":
    model.set_engine(
        ApproximateEngine(model, 1, ApproximateEngine.geometric_mean, exploration=False)
    )

model.add_tensor_source("train", MNIST_train)
model.add_tensor_source("test", MNIST_test)

loader = DataLoader(train_set, 2, False)

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
