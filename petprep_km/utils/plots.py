import matplotlib.pyplot as plt
import os

def plot_fit(tac_times, tac_values, fit_times, fit_values, region_name, output_path):
    plt.figure(figsize=(8, 4))
    plt.plot(tac_times, tac_values, 'o-', label='TAC (Data)')
    plt.plot(fit_times, fit_values, '--', label='Model Fit')
    plt.title(f"Model Fit - {region_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Radioactivity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()