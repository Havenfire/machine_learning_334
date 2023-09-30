import pandas as pd
import matplotlib.pyplot as plt

data_training_acc = [[1, 1.0],
                    [5, 0.942],
                    [10, 0.928],
                    [15, 0.92],
                    [20, 0.928],
                    [25, 0.926],
                    [30, 0.931]
                    ]

data_test_acc =     [[1, 0.903],
                    [5, 0.925],
                    [10, 0.914],
                    [15, 0.923],
                    [20, 0.926],
                    [25, 0.924],
                    [30, 0.922]
                    ]

# Create DataFrames
df_training_acc = pd.DataFrame(data_training_acc, columns=["k", "Accuracy"])
df_test_acc = pd.DataFrame(data_test_acc, columns=["k", "Accuracy"])

# Plot the data
plt.plot(df_training_acc["k"], df_training_acc["Accuracy"], label="Training Accuracy")
plt.plot(df_test_acc["k"], df_test_acc["Accuracy"], label="Test Accuracy")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Training and Test Accuracy vs. k")
plt.legend()
plt.grid(True)
plt.show()
