import argparse
import pandas as pd
import os 
import glob
import matplotlib.pyplot as plt
import tqdm

def plot_data(df: pd.DataFrame, name, f, ax) -> None:

    ax[0,0].plot(df.index, df['Train loss'], label=name)
    ax[1,0].plot(df.index, df['Validation loss'],label=name)
    ax[2,0].plot(df.index, df['Test loss'],label=name)
    ax[0,0].set_title("Loss")
    ax[0,0].set_ylabel('Train')
    ax[1,0].set_ylabel('Validation')
    ax[2,0].set_ylabel('Test')

    ax[0,1].plot(df.index, df['Train accuracy'], label=name)
    ax[1,1].plot(df.index, df['Validation accuracy'], label=name)
    ax[2,1].plot(df.index, df['Test accuracy'], label=name)
    ax[0,1].set_title("Accuracy")


def main():
    parser = argparse.ArgumentParser(description="Plot training and validation curves.")
    parser.add_argument("--csv_path", help="Path to the log file")
    args = parser.parse_args()

    filelist = glob.glob(os.path.join(args.csv_path, '*_LOSO_*.csv'))

    f,ax = plt.subplots(3,2)
    for f in tqdm.tqdm(filelist):
        name = f.split('/')[-1]
        df = pd.read_csv(f)
        plot_data(df, name, f, ax)
    # plt.legend()
    plt.show()



if __name__ == "__main__":
    main()