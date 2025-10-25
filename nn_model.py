# Lung Cancer Survival Prediction using PyTorch 

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns


#Load and Explore Dataset

df = pd.read_csv("balanced_dataset.csv")

# Drop unnecessary columns
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)

# Handle date columns: convert to treatment duration and extract month
if 'diagnosis_date' in df.columns and 'end_treatment_date' in df.columns:
    df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'], errors='coerce')
    df['end_treatment_date'] = pd.to_datetime(df['end_treatment_date'], errors='coerce')
    df['treatment_duration'] = (df['end_treatment_date'] - df['diagnosis_date']).dt.days
    df['diagnosis_month'] = df['diagnosis_date'].dt.month
    df.drop(columns=['diagnosis_date', 'end_treatment_date'], inplace=True)

#Fill missing values with more sophisticated approach
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)


#Enhanced Feature Engineering
#Encode categorical variables with memory-efficient categorical dtype
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

#Create interaction features
if "bmi" in df.columns and "age" in df.columns:
    df["bmi_age_ratio"] = df["bmi"] / df["age"]

if "cancer_stage" in df.columns:
    df["stage_treatment_interaction"] = df["cancer_stage"] * df["treatment_type"]

#Create risk score based on medical conditions
medical_conditions = ['hypertension', 'asthma', 'cirrhosis', 'other_cancer']
if all(col in df.columns for col in medical_conditions):
    df['risk_score'] = df[medical_conditions].sum(axis=1)


#Feature Selection

X = df.drop('survived', axis=1)
y = df['survived']

#Calculate feature importance using mutual information
mi_scores = mutual_info_classif(X, y)
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': mi_scores})
feature_importance = feature_importance.sort_values('importance', ascending=False)

#Select top features
top_features = feature_importance['feature'].head(12).tolist()
X = X[top_features]

#Normalize features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


#Enhanced Neural Network
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


#Training with Cross-validation
def train_model(X, y, fold_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        if fold != fold_num:  # Skip other folds
            continue
            
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train.values).to(device)
        y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1).to(device)
        X_val_tensor = torch.FloatTensor(X_val.values).to(device)
        y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1).to(device)
        
        model = EnhancedLungCancerPredictor(X_train.shape[1]).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        best_val_auc = 0
        epochs = 100
        batch_size = 256
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            # Mini-batch training
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_probs = torch.sigmoid(val_outputs)
                val_preds = (val_probs > 0.5).float()
                
                val_auc = roc_auc_score(y_val_tensor.cpu(), val_probs.cpu())
                val_acc = accuracy_score(y_val_tensor.cpu(), val_preds.cpu())
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    torch.save(model.state_dict(), f'best_model_fold_{fold}.pth')
                
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(X_train_tensor):.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}")
        
        #Load best model and evaluate
        model.load_state_dict(torch.load(f'best_model_fold_{fold}.pth'))
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_probs = torch.sigmoid(val_outputs)
            val_preds = (val_probs > 0.5).float()
            
            accuracy = accuracy_score(y_val_tensor.cpu(), val_preds.cpu())
            auc = roc_auc_score(y_val_tensor.cpu(), val_probs.cpu())
            f1 = f1_score(y_val_tensor.cpu(), val_preds.cpu())
            
            fold_results.append({
                'accuracy': accuracy,
                'auc': auc,
                'f1': f1
            })
            
            print(f"\nFold {fold} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"AUC-ROC: {auc:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_val_tensor.cpu(), val_preds.cpu()))
            
    return model, fold_results

#Train the model with cross-validation
for fold in range(5):
    print(f"\nTraining Fold {fold}")
    model, results = train_model(X, y, fold)
    
    # Plot feature importance for this fold
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'Feature Importance (Fold {fold})')
    plt.tight_layout()
    plt.savefig(f'feature_importance_fold_{fold}.png')
    plt.close()

print("\nTraining completed! Check the feature importance plots and model performance for each fold.")