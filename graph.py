import matplotlib.pyplot as plt
import re
import os

def plot_training_logs(log_file="training_log.txt"):
    if not os.path.exists(log_file):
        print(f"[!] Error: Could not find '{log_file}'.")
        print("Run your training like this: python train.py > training_log.txt")
        return

    with open(log_file, 'r') as f:
        log_text = f.read()

    epochs = []
    d_losses = []
    g_losses = []
    data_losses = []

    pattern = r"Epoch \[(\d+)/\d+\] \| D Loss: ([\d.]+) \| G Total Loss: ([\d.]+) \| Data Loss: ([\d.]+)"

    for line in log_text.strip().split('\n'):
        match = re.search(pattern, line)
        if match:
            epochs.append(int(match.group(1)))
            d_losses.append(float(match.group(2)))
            g_losses.append(float(match.group(3)))
            data_losses.append(float(match.group(4)))

    if not epochs:
        print("[!] No epoch data found in the log file. Check the formatting.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Image Quality
    ax1.plot(epochs, d_losses, label='Discriminator Loss', color='red', alpha=0.8, linewidth=2)
    ax1.plot(epochs, g_losses, label='Generator Total Loss', color='blue', alpha=0.8, linewidth=2)
    ax1.set_title('Adversarial Tug-of-War (Image Quality)')
    ax1.set_ylabel('Loss Value')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Data Extraction
    ax2.plot(epochs, data_losses, label='Data Loss (MSE)', color='green', linewidth=2)
    ax2.set_title('Steganography Payload Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Squared Error')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_training_logs()