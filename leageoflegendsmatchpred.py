## Write your code here
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
import torch
import sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
import torch.optim as optim
datas=pd.read_csv( "league_of_legends_data_large.csv")
X = datas.drop('win', axis=1)  # Features
y = datas['win']               # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)
## Write your code here
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

input_dim = X_train.shape[1]
model = LogisticRegressionModel(input_dim)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 1000
y_train = y_train.view(-1, 1).float()
y_test = y_test.view(-1, 1).float()

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


model.eval()
with torch.no_grad():
    train_outputs = model(X_train)
    test_outputs = model(X_test)

    train_preds = (train_outputs >= 0.5).float()
    test_preds = (test_outputs >= 0.5).float()

    train_accuracy = (train_preds == y_train).sum().item() / y_train.size(0)
    test_accuracy = (test_preds == y_test).sum().item() / y_test.size(0)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

input_dim = X_train.shape[1]
model = LogisticRegressionModel(input_dim)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)

epochs = 1000
y_train = y_train.view(-1, 1).float()
y_test = y_test.view(-1, 1).float()

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


model.eval()
with torch.no_grad():
    train_outputs = model(X_train)
    test_outputs = model(X_test)

    train_preds = (train_outputs >= 0.5).float()
    test_preds = (test_outputs >= 0.5).float()

    train_accuracy = (train_preds == y_train).sum().item() / y_train.size(0)
    test_accuracy = (test_preds == y_test).sum().item() / y_test.size(0)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
## Write your code here
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import itertools


y_test_np = y_test.numpy()
y_pred_test_np = test_outputs.numpy()

# Binary predictions using 0.5 threshold
y_pred_test_labels = (y_pred_test_np >= 0.5).astype(int)


cm = confusion_matrix(y_test_np, y_pred_test_labels)

plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = [0, 1]
plt.xticks(tick_marks, ['Loss', 'Win'])
plt.yticks(tick_marks, ['Loss', 'Win'])

thresh = cm.max() / 2
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

print("Classification Report:\n")
print(classification_report(y_test_np, y_pred_test_labels, target_names=['Loss', 'Win']))

fpr, tpr, thresholds = roc_curve(y_test_np, y_pred_test_np)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Save the model
model_path = "logistic_regression_lol.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
# Load the model
loaded_model = LogisticRegressionModel(input_dim)
loaded_model.load_state_dict(torch.load(model_path))
# Ensure the loaded model is in evaluation mode
loaded_model.eval()
print("Model loaded and set to evaluation mode.")

# Evaluate the loaded model
loaded_model.eval()
print("Model loaded and set to evaluation mode.")
with torch.no_grad():
    # Get predictions
    loaded_test_outputs = loaded_model(X_test)
    
    # Convert probabilities to binary predictions
    loaded_test_preds = (loaded_test_outputs >= 0.5).float()
    
    # Calculate accuracy
    loaded_test_accuracy = (loaded_test_preds == y_test).sum().item() / y_test.size(0)

print(f"Test Accuracy of Loaded Model: {loaded_test_accuracy * 100:.2f}%")
# Define the learning rates to test
learning_rates = [0.01, 0.05, 0.1]
num_epochs = 100
input_dim = X_train.shape[1]

# Dictionary to store test accuracies for each learning rate
accuracy_results = {}

for lr in learning_rates:
    print(f"\nTraining with learning rate: {lr}")

    # Reinitialize the model
    model = LogisticRegressionModel(input_dim)
    
    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)  # include L2 regularization

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_preds = (test_outputs >= 0.5).float()
        test_accuracy = (test_preds == y_test).sum().item() / y_test.size(0)
        accuracy_results[lr] = test_accuracy
        print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Identify the best learning rate
best_lr = max(accuracy_results, key=accuracy_results.get)
print("\nHyperparameter Tuning Results:")
for lr, acc in accuracy_results.items():
    print(f"Learning Rate: {lr}, Test Accuracy: {acc*100:.2f}%")
print(f"\nBest Learning Rate: {best_lr} with Test Accuracy: {accuracy_results[best_lr]*100:.2f}%")
## Write your code here
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt

# Extract the weights of the linear layer
## Write your code here
# Ensure the model is trained and in evaluation mode
model.eval()


weights = model.fc1.weight.data.numpy().flatten()  # Flatten to 1D array

# Create a DataFrame with feature names and their importance
feature_names = X.columns  # Assuming X is the original DataFrame of features
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': weights
})

# Sort features by absolute importance
feature_importance['abs_importance'] = feature_importance['Importance'].abs()
feature_importance = feature_importance.sort_values(by='abs_importance', ascending=False)

# Display the feature importance
display(feature_importance[['Feature', 'Importance']])


plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel("Weight Value (Feature Importance)")
plt.title("Feature Importance in Logistic Regression Model")
plt.gca().invert_yaxis()  # Highest importance at the top
plt.show()
# ........................................