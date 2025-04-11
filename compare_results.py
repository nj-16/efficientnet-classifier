# compare_results.py
import pickle
import matplotlib.pyplot as plt

# Load accuracy lists
with open("baseline_accs.pkl", "rb") as f:
    baseline_accs = pickle.load(f)

with open("tuned_accs.pkl", "rb") as f:
    tuned_accs = pickle.load(f)

epochs = [1, 2, 3]
plt.plot(epochs, baseline_accs, label="Baseline", marker='o')
plt.plot(epochs, tuned_accs, label="Improved", marker='o')
plt.title("CIFAR-10: Baseline vs Improved Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.ylim(90, 100)
plt.legend()
plt.grid(True)
plt.show()
