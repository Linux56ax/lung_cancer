# Lung Cancer Survival Prediction using PyTorch (Improved)
# --------------------------------------------------------
# Author: SM
# Description: Enhanced version with class weighting, deeper architecture, LR scheduling, and feature regularization.

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score
import matplotlib.pyplot as plt


# --------------------------
# 1. Load and Explore Dataset
# --------------------------
df = pd.read_csv("balanced_dataset.csv")

# Drop unnecessary columns
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)

# Handle date columns: convert to treatment duration
if 'diagnosis_date' in df.columns and 'end_treatment_date' in df.columns:
    df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'], errors='coerce')
    df['end_treatment_date'] = pd.to_datetime(df['end_treatment_date'], errors='coerce')
    df['treatment_duration'] = (df['end_treatment_date'] - df['diagnosis_date']).dt.days
    df.drop(columns=['diagnosis_date', 'end_treatment_date'], inplace=True)

# Fill missing values
df.fillna(df.mode().iloc[0], inplace=True)

# --------------------------
# 2. Encode Categorical Data
# --------------------------
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# --------------------------
# 3. Feature Engineering
# --------------------------
if "bmi" in df.columns and "cholesterol_level" in df.columns:
    df["bmi_cholesterol_ratio"] = df["bmi"] / (df["cholesterol_level"] + 1)

if "age" in df.columns and "cancer_stage" in df.columns:
    df["age_stage_interact"] = df["age"] * df["cancer_stage"]

# --------------------------
# 4. Split Features and Target
# --------------------------
X = df.drop('survived', axis=1)
y = df['survived']

# Normalize numerical features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

# --------------------------
# 5. Define Enhanced Neural Network
# --------------------------
class LungCancerPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # no sigmoid here because BCEWithLogitsLoss handles it
        )

    def forward(self, x):
        return self.layers(x)

input_dim = X_train.shape[1]
model = LungCancerPredictor(input_dim)

# --------------------------
# 6. Training Setup (with class weighting)
# --------------------------
num_pos = y_train.sum()
num_neg = len(y_train) - num_pos
pos_weight = torch.tensor(num_neg / num_pos, dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
epochs = 100

train_losses, test_losses = [], []
print(df['survived'].value_counts())

# --------------------------
# 7. Training Loop
# --------------------------
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_losses.append(loss.item())

    # Validation
    model.eval()
    with torch.no_grad():
        val_out = model(X_test_tensor)
        val_loss = criterion(val_out, y_test_tensor)
        test_losses.append(val_loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {loss.item():.4f}, Test Loss: {val_loss.item():.4f}")

# --------------------------
# 8. Evaluation
# --------------------------
model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(X_test_tensor))
    preds = (probs > 0.4).int()  # threshold tuned for imbalance

print("\nClassification Report:")
print(classification_report(y_test_tensor, preds))

acc = accuracy_score(y_test_tensor, preds)
f1 = f1_score(y_test_tensor, preds)
auc = roc_auc_score(y_test_tensor, probs)
cm = confusion_matrix(y_test_tensor, preds)

print(f"\nAccuracy: {acc*100:.2f}%")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print("Confusion Matrix:\n", cm)

# --------------------------
# 9. Plot Loss Graph
# --------------------------
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss", linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Testing Loss")
plt.legend()
plt.show()

# --------------------------
# 10. Save the trained model
# --------------------------
torch.save(model.state_dict(), "lung_cancer_model_v2.pth")
print("✅ Improved model saved as lung_cancer_model_v2.pth")
