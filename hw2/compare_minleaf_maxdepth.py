import pandas as pd
import matplotlib.pyplot as plt

# Data for maximum depth with default minimum leaf
data_max_depth = [[1, 0.8865058087578195, 0.8875],
                  [5, 0.9436997319034852, 0.8791666666666667],
                  [10, 0.9794459338695264, 0.85625],
                  [15, 0.9883824843610366, 0.8479166666666667],
                  [20, 0.9883824843610366, 0.8479166666666667],
                 ]

# Data for minimum leaf with default maximum depth
data_min_leaf = [[1, 0.9445933869526363, 0.8791666666666667],
                 [5, 0.9436997319034852, 0.8791666666666667],
                 [10, 0.938337801608579, 0.8770833333333333],
                 [15, 0.9329758713136729, 0.8708333333333333],
                 [20, 0.9186773905272565, 0.8666666666666667],
                ]


# Create DataFrames for both datasets
df_max_depth = pd.DataFrame(data_max_depth, columns=["Max Depth", "Training Accuracy", "Test Accuracy"])
df_min_leaf = pd.DataFrame(data_min_leaf, columns=["Min Leaf", "Training Accuracy", "Test Accuracy"])

# Plot the first graph for max depth with default min leaf
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(df_max_depth["Max Depth"], df_max_depth["Training Accuracy"], label="Training Accuracy")
plt.plot(df_max_depth["Max Depth"], df_max_depth["Test Accuracy"], label="Test Accuracy")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Max Depth")
plt.legend()
plt.grid(True)

# Plot the second graph for min leaf with default max depth
plt.subplot(1, 2, 2)
plt.plot(df_min_leaf["Min Leaf"], df_min_leaf["Training Accuracy"], label="Training Accuracy")
plt.plot(df_min_leaf["Min Leaf"], df_min_leaf["Test Accuracy"], label="Test Accuracy")
plt.xlabel("Min Leaf")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Min Leaf")
plt.legend()
plt.grid(True)

plt.tight_layout()  # Ensures the subplots don't overlap
plt.show()
