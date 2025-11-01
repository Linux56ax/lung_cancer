

import os, math, sys, time
import pandas as pd, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns


# Paths
DATA_PATH = "dataset_med.csv"
BEST_MODEL_PATH = "best_lung_cancer_model.pth"
CM_IMAGE_PATH = "lung_confusion_matrix.png"


# Load dataset
df = pd.read_csv(DATA_PATH)
print("Loaded dataset:", DATA_PATH, "shape:", df.shape)

# Drop id if present
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Handle date columns if present
if 'diagnosis_date' in df.columns and 'end_treatment_date' in df.columns:
    df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'], errors='coerce')
    df['end_treatment_date'] = pd.to_datetime(df['end_treatment_date'], errors='coerce')
    df['treatment_duration'] = (df['end_treatment_date'] - df['diagnosis_date']).dt.days.fillna(0)
    df['diagnosis_month'] = df['diagnosis_date'].dt.month.fillna(0).astype(int)
    df = df.drop(columns=['diagnosis_date', 'end_treatment_date'])

# Identify numerical and categorical
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
# Ensure target column 'survived' exists
if 'survived' not in df.columns:
    raise ValueError("Target column 'survived' not found in dataset. Please ensure the CSV contains 'survived'.")

# Exclude target from numerical list
if 'survived' in numerical_cols:
    numerical_cols.remove('survived')

# Fill missing values
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "")

# Label encode categorical columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Feature engineering as in script
if "bmi" in df.columns and "age" in df.columns:
    df["bmi_age_ratio"] = df["bmi"] / (df["age"].replace(0, np.nan))
    df["bmi_age_ratio"] = df["bmi_age_ratio"].fillna(df["bmi_age_ratio"].median())

if "cancer_stage" in df.columns and "treatment_type" in df.columns:
    df["stage_treatment_interaction"] = df["cancer_stage"] * df["treatment_type"]

medical_conditions = ['hypertension', 'asthma', 'cirrhosis', 'other_cancer']
if all(col in df.columns for col in medical_conditions):
    df['risk_score'] = df[medical_conditions].sum(axis=1)

# Prepare X and y
X = df.drop(columns=['survived'])
y = df['survived'].astype(int)

# Feature selection via mutual info (handle cases where features < 12)
try:
    mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': mi_scores})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    top_k = min(12, X.shape[1])
    top_features = feature_importance['feature'].head(top_k).tolist()
    X = X[top_features]
except Exception as e:
    # fallback: use all features
    print("Mutual info failed:", e)
    top_features = X.columns.tolist()

# Normalize features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print("Using features:", X_scaled.columns.tolist())

# Convert to numpy for indexing
X_np = X_scaled.reset_index(drop=True)
y_np = y.reset_index(drop=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define model
class EnhancedLungCancerPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.batch_norm1 = nn.BatchNorm1d(input_dim)
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.output = nn.Linear(64, 1)
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x)

# Training params (reduced epochs to 30 for practical runtime)
n_splits = 7
epochs = 25
batch_size = 128
best_overall_auc = 0.0
best_model_state = None
fold_metrics = []

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y_np)):
    print(f"\n--- Fold {fold} ---")
    X_train = torch.FloatTensor(X_np.iloc[train_idx].values).to(device)
    y_train = torch.FloatTensor(y_np.iloc[train_idx].values).reshape(-1,1).to(device)
    X_val = torch.FloatTensor(X_np.iloc[val_idx].values).to(device)
    y_val = torch.FloatTensor(y_np.iloc[val_idx].values).reshape(-1,1).to(device)
    
    model = EnhancedLungCancerPredictor(X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_auc = 0.0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_train.size(0))
        epoch_loss = 0.0
        for i in range(0, X_train.size(0), batch_size):
            idx = perm[i:i+batch_size]
            batch_X = X_train[idx]
            batch_y = y_train[idx]
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        epoch_loss = epoch_loss / X_train.size(0)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_probs = torch.sigmoid(val_outputs).cpu().numpy()
            val_preds = (val_probs > 0.5).astype(int)
            val_auc = roc_auc_score(y_val.cpu().numpy(), val_probs)
            val_acc = accuracy_score(y_val.cpu().numpy(), val_preds)
        scheduler.step(val_loss)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f"best_model_fold_{fold}.pth")
        if (epoch+1) % 10 == 0 or epoch==0:
            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - val_auc: {val_auc:.4f} - val_acc: {val_acc:.4f}")
    
    # Load best for this fold and evaluate
    model.load_state_dict(torch.load(f"best_model_fold_{fold}.pth", map_location=device))
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_probs = torch.sigmoid(val_outputs).cpu().numpy().reshape(-1)
        val_preds = (val_probs > 0.5).astype(int)
        acc = accuracy_score(y_val.cpu().numpy(), val_preds)
        auc = roc_auc_score(y_val.cpu().numpy(), val_probs)
        f1 = f1_score(y_val.cpu().numpy(), val_preds)
        cm = confusion_matrix(y_val.cpu().numpy(), val_preds)
    fold_metrics.append({'fold': fold, 'accuracy': acc, 'auc': auc, 'f1': f1, 'confusion_matrix': cm})
    print(f"Fold {fold} -> Accuracy: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
    
    # track global best
    if auc > best_overall_auc:
        best_overall_auc = auc
        best_model_state = torch.load(f"best_model_fold_{fold}.pth", map_location=device)
        best_confusion = cm.copy()
        best_fold = fold

# Save best overall model
if best_model_state is not None:
    torch.save(best_model_state, BEST_MODEL_PATH)
    print("Saved best overall model to", BEST_MODEL_PATH)

# Aggregate metrics
metrics_df = pd.DataFrame([{'fold': m['fold'], 'accuracy': m['accuracy'], 'auc': m['auc'], 'f1': m['f1']} for m in fold_metrics])
metrics_df.loc['mean'] = metrics_df.mean(numeric_only=True)
print("\nPer-fold metrics:\n", metrics_df)

# Plot confusion matrix for best fold
plt.figure(figsize=(5,4))
sns.heatmap(best_confusion, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (Best Fold {best_fold})')
plt.tight_layout()
plt.savefig(CM_IMAGE_PATH)
plt.show()
print("Saved confusion matrix image to", CM_IMAGE_PATH)

