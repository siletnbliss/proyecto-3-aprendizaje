import os
from dotenv import load_dotenv
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

load_dotenv()
HISTORY_FOLDER = os.path.join(os.environ.get('BASE_PATH'), 'models_save')
CHART_FOLDER = os.path.join(os.environ.get('BASE_PATH'), 'chart')
sns.set_theme(style="whitegrid", palette="tab10")


def load_dataset(file_path):
    return pd.read_csv(file_path)


def plot_results(df_fashion, df_digit, df_cifar10):
    fig, axis = plt.subplots(nrows=2, ncols=3, figsize=(
        14, 6), constrained_layout=True)

    axis[0, 0] = sns.lineplot(
        data=df_fashion[["accuracy", "val_accuracy"]], ax=axis[0, 0], linewidth=2.5)
    axis[0, 0].set(xlabel='Epochs')
    axis[0, 0].set_ylabel('Accuracy', weight='bold')
    axis[0, 0].set_title("Fashion-MNIST")

    axis[1, 0] = sns.lineplot(
        data=df_fashion[["loss", "val_loss"]], ax=axis[1, 0], linewidth=2.5)
    axis[1, 0].set_ylabel('Loss', weight='bold')
    axis[1, 0].set(xlabel='Epochs')

    axis[0, 1] = sns.lineplot(
        data=df_digit[["accuracy", "val_accuracy"]], ax=axis[0, 1], linewidth=2.5)
    axis[0, 1].set_title("MNIST-Digit")
    axis[0, 1].set_xlabel('Epochs')

    axis[1, 1] = sns.lineplot(
        data=df_digit[["loss", "val_loss"]], ax=axis[1, 1], linewidth=2.5)

    axis[1, 1].set(xlabel='Epochs')

    axis[0, 2] = sns.lineplot(
        data=df_cifar10[["accuracy", "val_accuracy"]], ax=axis[0, 2], linewidth=2.5)
    axis[0, 2].set_title("CIFAR-10")
    axis[0, 2].set(xlabel='Epochs')

    axis[1, 2] = sns.lineplot(
        data=df_cifar10[["loss", "val_loss"]], ax=axis[1, 2], linewidth=2.5)

    axis[1, 2].set(xlabel='Epochs')

    fig.text(0.19, -0.07, '(a)')
    fig.text(0.52, -0.07, '(b)')
    fig.text(0.84, -0.07, '(c)')

    plt.show()
    fig.savefig(os.path.join(CHART_FOLDER, 'result.pdf'), bbox_inches="tight")


def run():
    print('start generating...')
    df_fashion = load_dataset(os.path.join(
        HISTORY_FOLDER, 'fashion-block', 'history.csv'))
    df_digit = load_dataset(os.path.join(
        HISTORY_FOLDER, 'digit-block', 'history.csv'))
    df_cifar10 = load_dataset(os.path.join(
        HISTORY_FOLDER, 'cifar10-block', 'history.csv'))

    plot_results(df_fashion, df_digit, df_cifar10)

    df = pd.read_csv(os.path.join(HISTORY_FOLDER, 'comparitive.csv'))
    g = sns.catplot(
        data=df, kind="bar",
        x="Database", y="Accuracy", hue="Model",
        ci="sd")
    plt.show()
    g.savefig(os.path.join(CHART_FOLDER, 'compare.pdf'), bbox_inches="tight")
    print('file saved!')


if __name__ == "__main__":
    run()
