import matplotlib.pyplot as plt

def plot_his(his_loss_train, his_loss_test, his_acc_train, his_acc_test, n_epochs, name_fig):
    x = range(1, n_epochs+1)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    axes[0].plot(x, his_loss_train, c='b')
    axes[0].plot(x, his_loss_test, c='r')
    axes[0].set_title("Loss")
    axes[0].set_xlabel("n_epochs")
    axes[1].plot(x, his_acc_train, c='b')
    axes[1].plot(x, his_acc_test, c='r')
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("n_epochs")
    axes[1].legend(["Train", "Test"])
    plt.savefig(name_fig)
    plt.show()