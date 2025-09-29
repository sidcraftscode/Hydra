import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_flops_per_token():
    """
    Reads the flops_summary.csv file, calculates FLOPs per token,
    and plots it against sequence length for transformer and hydra models.
    """
    flops_df = pd.read_csv('results/flops_summary.csv')

    # Calculate FLOPs per token
    # TFLOPs is 10^12 FLOPs, so we calculate (TFLOPs * 10^12) / seq_len
    flops_df['flops_per_token'] = (flops_df['tflops'] * 1e12) / flops_df['seq_len']

    # Filter for the models we want to plot
    transformer_df = flops_df[flops_df['model'] == 'transformer_base']
    hydra_df = flops_df[flops_df['model'] == 'hydra_base']

    # Create the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(transformer_df['seq_len'], transformer_df['flops_per_token'], marker='o', linestyle='-', label='Transformer (Base)')
    ax.plot(hydra_df['seq_len'], hydra_df['flops_per_token'], marker='s', linestyle='-', label='Hydra (Base)')

    ax.set_title('FLOPs per Token vs. Sequence Length')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('FLOPs per Token')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, which="both", ls="--")

    # Save the figure
    plt.savefig('results/fig_flops_per_token.png')
    print("Successfully created results/fig_flops_per_token.png")

if __name__ == "__main__":
    plot_flops_per_token()
