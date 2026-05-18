import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_data(df: pd.DataFrame) -> None:

    f,ax = plt.subplots(1,2)
    ax[0].plot(df.index, df['Train loss'], label="Training")
    ax[0].plot(df.index, df['Validation loss'],label="Validation")
    ax[0].plot(df.index, df['Test loss'],label="Test")
    ax[0].set_title("Loss")

    ax[1].plot(df.index, df['Train accuracy'], label="Training")
    ax[1].plot(df.index, df['Validation accuracy'], label="Validation")
    ax[1].plot(df.index, df['Test accuracy'], label="Test")
    ax[1].set_title("Accuracy")

    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot training and validation curves.")
    parser.add_argument("--csv_path", help="Path to the log file")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    # import ipdb; ipdb.set_trace()
    print(df.keys())
    plot_data(df)


if __name__ == "__main__":
    main()