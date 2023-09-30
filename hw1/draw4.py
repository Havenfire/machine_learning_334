import pandas as pd
import matplotlib.pyplot as plt

no_preprocessing = [
    [1, 0.8583333333333333],
    [2, 0.8770833333333333],
    [3, 0.875],
    [5, 0.8666666666666667],
    [7, 0.8708333333333333],
    [10, 0.8708333333333333],
    [15, 0.8625],
    # [20, 0.8604166666666667],
    # [25, 0.8604166666666667],
    # [30, 0.8645833333333334],
]

standard_scale =  [
    [1, 0.875],
    [2, 0.8875],    
    [3, 0.8854166666666666],
    [5, 0.8645833333333334],
    [7, 0.8666666666666667],
    [10, 0.8791666666666667],
    [15, 0.8833333333333333],
    # [20, 0.8791666666666667],
    # [25, 0.8916666666666667],
    # [30, 0.88125],
]   

min_max_scale =  [
    [1, 0.88125],
    [2, 0.8791666666666667],
    [3, 0.875],
    [5, 0.88125],
    [7, 0.8666666666666667],
    [10, 0.86875],
    [15, 0.8895833333333333],
    # [20, 0.8854166666666666],
    # [25, 0.8833333333333333],
    # [30, 0.875],
]   
with_irr_feat =  [
    [1, 0.8375],
    [2, 0.86875],
    [3, 0.8666666666666667],
    [5, 0.8541666666666666],
    [7, 0.8729166666666667],
    [10, 0.8625],
    [15, 0.86875],
    # [20, 0.8645833333333334],
    # [25, 0.8625],
    # [30, 0.8645833333333334],
]   

# Create DataFrames
df_no_preprocessing = pd.DataFrame(no_preprocessing, columns=["k", "Accuracy"])
df_standard_scale = pd.DataFrame(standard_scale, columns=["k", "Accuracy"])
df_min_max_scale = pd.DataFrame(min_max_scale, columns=["k", "Accuracy"])
df_with_irr_feat = pd.DataFrame(with_irr_feat, columns=["k", "Accuracy"])

# Plot the data
plt.plot(df_no_preprocessing["k"], df_no_preprocessing["Accuracy"], label="no-preprocessing")
plt.plot(df_standard_scale["k"], df_standard_scale["Accuracy"], label="standard scale")
plt.plot(df_min_max_scale["k"], df_min_max_scale["Accuracy"], label="min max scale")
plt.plot(df_with_irr_feat["k"], df_with_irr_feat["Accuracy"], label="with irrelevant feature")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. k")
plt.legend()
plt.grid(True)
plt.show()
