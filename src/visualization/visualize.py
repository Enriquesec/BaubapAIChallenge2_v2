import seaborn as sns
import matplotlib.pyplot as plt


def plot_histograms(data_list, df_train, df_val):
    num_plots = len(data_list)
    num_rows = 5
    num_cols = 6

    if num_plots != num_rows * num_cols:
        raise ValueError("El número de datos debe ser igual a 30 para crear una cuadrícula de 5x5.")

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    for i, cols in enumerate(data_list):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        
        sns.histplot(df_train[cols], ax=ax, color="red", stat='probability', binwidth=.02, alpha=0.5)
        sns.histplot(df_val[cols], ax=ax,color="green", stat='probability', binwidth=.02, alpha=0.5)
        ax.set_title(f'{cols}')

    plt.tight_layout()
    plt.show()
