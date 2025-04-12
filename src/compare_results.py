# compare_results.py
import pickle
import matplotlib.pyplot as plt

# Load baseline accuracy
with open("baseline_accs.pkl", "rb") as f:
    baseline_accs = pickle.load(f)

# Load final improved v3 accuracy
with open("v3_accs.pkl", "rb") as f:
    v3_accs = pickle.load(f)

# Set dynamic epoch ranges
epochs_baseline = list(range(1, len(baseline_accs) + 1))
epochs_v3 = list(range(1, len(v3_accs) + 1))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(epochs_baseline, baseline_accs, label="Baseline", marker='o')
plt.plot(epochs_v3, v3_accs, label="Improved v3 (Final)", marker='o')

plt.title("CIFAR-10: Baseline vs Improved v3 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy (%)")
plt.ylim(90, 100)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
