import argparse

import torch
import torch.nn as nn

from model.model import GATCLML
from model.loss import CenterLoss
from model.utils import plot_his
from dataset.data_loader import CifarGraphLoader

from tqdm.auto import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--n_classes', type=int, default=10, help="number of classes")
parser.add_argument('--device', type=str, default='cuda', help="cuda or cpu")
parser.add_argument('--n_epoches', type=int, default=20, help="number of epoches")
parser.add_argument('--batch_size', type=int, default=64, help="size of each batch")
parser.add_argument('--learning_rate', type=float, default=0.001, help="learning rate")
parser.add_argument('--n_heads', type=int, default=8, help="number of heads each GAT layer")
parser.add_argument('--alpha', type=float, default=0.5, help="alpha in Center loss")
parser.add_argument('--in_features', type=int, default=97, help="input features dimension of model")
parser.add_argument('--hidden_dim', type=int, default=128, help="hidden features dimension of model")
parser.add_argument('--out_features', type=int, default=64, help="output features dimension of model")

args = parser.parse_args()


def train_phase(model, train_loader, cross_entropy_loss, center_loss, optimizer, device):

    model.train()
    n_iterations = len(train_loader)
    total_loss = 0.0
    train_correct = 0
    train_total = 0

    for data in tqdm(train_loader):
        data = data.to(device)
        labels = data.y
        optimizer.zero_grad()

        features, logits = model(data.x, data.edge_index, data.batch)
        total_cross_entropy_loss = cross_entropy_loss(logits, labels)
        total_center_loss = center_loss(features, labels)

        loss = total_cross_entropy_loss + total_center_loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        _, train_predicted = torch.max(logits, 1)
        train_total += labels.size(0)
        train_correct += (train_predicted == labels).sum().item()

    print(total_loss)
    train_loss = total_loss / n_iterations
    train_accuracy = (train_correct / train_total) * 100

    return train_loss, train_accuracy


def test_phase(model, test_loader, cross_entropy_loss, center_loss, device):

    model.eval()
    n_iterations = len(test_loader)
    total_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for data in tqdm(test_loader):
            data = data.to(device)
            labels = data.y

            features, logits = model(data.x, data.edge_index, data.batch)
            total_cross_entropy_loss = cross_entropy_loss(logits, labels)
            total_center_loss = center_loss(features, labels)

            loss = total_cross_entropy_loss + total_center_loss
            total_loss += loss.item()

            _, test_predicted = torch.max(logits, 1)
            test_total += labels.size(0)
            test_correct += (test_predicted == labels).sum().item()

    test_loss = total_loss / n_iterations
    test_accuracy = (test_correct / test_total) * 100

    return test_loss, test_accuracy


if __name__ == '__main__':

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    in_features = args.in_features
    hidden_dim = args.hidden_dim
    out_features = args.out_features
    num_classes = args.n_classes
    num_heads = args.n_heads
    learning_rate = args.learning_rate
    alpha = args.alpha
    num_epochs = args.n_epoches
    

    loader = CifarGraphLoader()
    train_loader, test_loader = loader.load_data()

    model = GATCLML(in_features=in_features, 
                                    hidden_dim=hidden_dim, 
                                    out_features=out_features, 
                                    num_classes=num_classes, 
                                    num_heads=num_heads)

    cross_entropy_loss = nn.CrossEntropyLoss()
    center_loss = CenterLoss(num_classes=num_classes, feat_dim=out_features, alpha=alpha)

    optimizer = torch.optim.Adam([
        {'params': model.gat.parameters()},
        {'params': model.metric_learning.parameters()},
        {'params': model.classifier.parameters()},
        {'params': center_loss.parameters(), 'lr': 0.005}],
        lr=learning_rate
    )

    history_loss_train = []
    history_loss_test = []
    history_acc_train = []
    history_acc_test = []

    loss_opt = 1e9
    acc_opt = -1e9

    for epoch in range(num_epochs):
        print()
        print(f'Epoch [{epoch+1}/{num_epochs}]:')

        train_loss, train_accuracy = train_phase(model, train_loader, cross_entropy_loss, center_loss, optimizer, device)
        print('Train Loss: {:.4f}, Train Accuracy: {:.4f}%' .format(train_loss, train_accuracy))

        test_loss, test_accuracy = test_phase(model, test_loader, cross_entropy_loss, center_loss, device)
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
