import pandas as pd
import numpy as np

# H100 Specs
# https://www.nvidia.com/en-us/data-center/h100/
# Using FP16 Tensor Core FLOPS
H100_FP16_TFLOPS = 1979  # 989 * 2 with sparsity

# Model dimensions (assuming 'base' size from papers like GPT-3)
# These are approximations for calculation.
D_MODEL = 768
N_LAYERS = 12
N_HEADS = 12
D_FF = 4 * D_MODEL
VOCAB_SIZE = 50257

def calculate_transformer_flops(seq_len):
    """
    Calculates the FLOPs for a single forward pass of a Transformer model.
    Formula is based on the PaLM paper's appendix.
    FLOPs ≈ 6 * N_layers * d_model^2 * seq_len
    """
    flops = 6 * N_LAYERS * (D_MODEL**2) * seq_len
    return flops

def calculate_ssm_flops(seq_len):
    """
    Calculates the FLOPs for a single forward pass of an SSM-based model (like Mamba/Hydra).
    Formula is based on the Mamba paper.
    FLOPs ≈ 6 * N_layers * d_model^2 * seq_len (dominated by linear projections)
    The SSM part itself is linear in seq_len, but the main compute is in the linear layers.
    The key difference from Transformers is the lack of the seq_len^2 term from attention.
    """
    flops = 6 * N_LAYERS * (D_MODEL**2) * seq_len
    return flops

def main():
    # Read the throughput data
    throughput_df = pd.read_csv('results/throughput_summary.csv')

    # Calculate FLOPs for each row
    flops_data = []
    for _, row in throughput_df.iterrows():
        model = row['model']
        seq_len = row['seq_len']
        
        if 'transformer' in model:
            # Transformer FLOPs have a quadratic term for attention
            # FLOPs ≈ 2 * N_layers * (2 * d_model * seq_len^2) for attention
            # And 6 * N_layers * d_model^2 * seq_len for projections
            attention_flops = 2 * N_LAYERS * (2 * D_MODEL * seq_len**2)
            projection_flops = 6 * N_LAYERS * (D_MODEL**2) * seq_len
            total_flops = attention_flops + projection_flops
        else:
            # SSM-based models have FLOPs linear in sequence length
            total_flops = calculate_ssm_flops(seq_len)
            
        flops_data.append({
            'model': model,
            'seq_len': seq_len,
            'tflops': total_flops / 1e12  # Convert to TFLOPs
        })

    # Create a new DataFrame for the FLOPs summary
    flops_df = pd.DataFrame(flops_data)

    # Save the results to a new CSV file
    flops_df.to_csv('results/flops_summary.csv', index=False)
    print("Successfully created results/flops_summary.csv")

if __name__ == "__main__":
    main()
