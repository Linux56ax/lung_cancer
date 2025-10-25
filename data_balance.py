import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("dataset_med.csv")
# Assume df is your dataset and 'target' is the label column
df_majority = df[df.survived == 0]
df_minority = df[df.survived == 1]

df_majority_downsampled = df_majority.sample(len(df_minority), random_state=42)
df_balanced = pd.concat([df_majority_downsampled, df_minority])

# Shuffle the dataset
df_balanced = df_balanced.sample(frac=1, random_state=42)
counts = df_balanced['survived'].value_counts()
# Basic bar plot using matplotlib
plt.figure(figsize=(6,4))
plt.bar(['Survived', 'Not Survived'], counts, color=['green', 'red'])
plt.title("Balanced Dataset: Survived vs Not Survived")
plt.ylabel("Number of Samples")
plt.show()

# Optional: Using seaborn for a nicer plot
plt.figure(figsize=(6,4))
sns.countplot(x='survived', data=df_balanced, palette=['green','red'])
plt.xticks([0,1], ['Survived', 'Not Survived'])
plt.title("Balanced Dataset: Survived vs Not Survived")
plt.show()

df_balanced.to_csv("balanced_dataset.csv", index=False)

print("Balanced dataset saved as 'balanced_dataset.csv'")