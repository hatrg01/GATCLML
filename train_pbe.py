import os
import torch

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

import argparse

from model.model import GATCLML
from model.utils import plot_his

from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help="cuda or cpu")
parser.add_argument('--n_epoches', type=int, default=170, help="number of epoches")

args = parser.parse_args()



def train_phase(model, train_loader, cross_entropy_loss, optimizer, device):

    model.train()
    n_iterations = len(train_loader)
    total_loss = 0.0
    train_correct = 0
    train_total = 0

    for data in tqdm(train_loader):
        data = data.to(device)

        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.

        loss = cross_entropy_loss(out, data.y)  # Compute the loss.
        total_loss += loss.item()

        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

        

        _, train_predicted = torch.max(out, 1)
        train_total += data.y.size(0)
        train_correct += (train_predicted == data.y).sum().item()

    # print(total_loss)
    train_loss = total_loss / n_iterations
    train_accuracy = (train_correct / train_total) * 100

    return train_loss, train_accuracy


def test_phase(model, test_loader, cross_entropy_loss, device):

    model.eval()
    n_iterations = len(test_loader)
    total_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for data in tqdm(test_loader):
            data = data.to(device)

            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.

            loss = cross_entropy_loss(out, data.y)  # Compute the loss.
            total_loss += loss.item()

            _, test_predicted = torch.max(out, 1)
            test_total += data.y.size(0)
            test_correct += (test_predicted == data.y).sum().item()

    test_loss = total_loss / n_iterations
    test_accuracy = (test_correct / test_total) * 100

    return test_loss, test_accuracy


if __name__ == '__main__':

    num_epochs = args.n_epoches

    device = None
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')
    print(device)

    dataset = TUDataset(root='data/TUDataset', name='MUTAG')

    torch.manual_seed(12345)
    dataset = dataset.shuffle()

    train_dataset = dataset[:150]
    test_dataset = dataset[150:]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = GATCLML(in_channels=dataset.num_node_features, hidden_channels=64, out_features=dataset.num_classes)
    cross_entropy_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    history_loss_train = []
    history_loss_test = []
    history_acc_train = []
    history_acc_test = []

    loss_opt = 1e9
    acc_opt = -1e9

    for epoch in range(num_epochs):
        print()
        print(f'Epoch [{epoch+1}/{num_epochs}]:')

        train_loss, train_accuracy = train_phase(model, train_loader, cross_entropy_loss, optimizer, device)
        print('Train Loss: {:.4f}, Train Accuracy: {:.4f}%' .format(train_loss, train_accuracy))

        test_loss, test_accuracy = test_phase(model, test_loader, cross_entropy_loss, device)
        print('Test Loss: {:.4f}, Test Accuracy: {:.4f}%' .format(test_loss, test_accuracy))

        history_loss_train.append(train_loss)
        history_loss_test.append(test_loss)
        history_acc_train.append(train_accuracy)
        history_acc_test.append(test_accuracy)


        if test_loss < loss_opt and test_accuracy >= acc_opt:
            loss_opt = test_loss
            acc_opt = test_accuracy
            torch.save(model.state_dict(), 'best.pt')
        torch.save(model.state_dict(), 'last.pt')

    plot_his(history_loss_train, history_loss_test, history_acc_train, history_acc_test, num_epochs, "GATCLML.jpg")