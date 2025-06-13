import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
import torch.nn.init as init

# ----------------------
# 1. data loading
# ----------------------
# simulation data
def read_excel_data(file_path):
    df = pd.read_excel(file_path, header=None)
    samples = []
    for col in range(0, df.shape[1], 3):  # samples
        # labels
        labels = df.iloc[0:4, col].values.astype(np.float32)
        # i-v data
        vgs = df.iloc[5:56, col].values.astype(np.float32)
        ids = df.iloc[5:56, col + 1].values.astype(np.float32)
        samples.append({
            "vgs": vgs,
            "ids": ids,
            "labels": labels
        })
    return samples

# measured data
def read_excel_data_2(file_path):
    df = pd.read_excel(file_path, header=None)
    samples = []
    for col in range(0, df.shape[1], 3):
        labels = df.iloc[0:4, col].values.astype(np.float32)
        vgs = df.iloc[5:56, col].values.astype(np.float32)
        ids = df.iloc[5:56, col + 1].values.astype(np.float32)
        samples.append({
            "vgs": vgs,
            "ids": ids,
            "labels": labels
        })
    return samples

# ----------------------
# 2. data preprocessing
# ----------------------
def preprocess_sample(sample, target_length=150):
    vgs = sample["vgs"]
    ids = sample["ids"]
    valid_mask = (~np.isnan(vgs)) & (~np.isnan(ids))
    vgs = vgs[valid_mask]
    ids = ids[valid_mask]
    ids = np.where(ids < 1e-11, 1e-11, ids)  # clamping data value for small fluctuating current
    log_ids = np.log10(ids)
    f_vgs = interp1d(np.linspace(0, 1, len(vgs)), vgs, kind='linear')  # Unify IV curve lengths
    f_ids = interp1d(np.linspace(0, 1, len(ids)), ids, kind='cubic')
    f_log_ids = interp1d(np.linspace(0, 1, len(log_ids)), log_ids, kind='cubic')
    _vgs = f_vgs(np.linspace(0, 1, target_length))
    _ids = f_ids(np.linspace(0, 1, target_length))
    _log_ids = f_log_ids(np.linspace(0, 1, target_length))
    dlogids = np.gradient(_log_ids, _vgs)  # first-order derivative calculation
    return np.stack([_ids, _log_ids, dlogids], axis=1).astype(np.float32)

# ----------------------
# 3. dataset
# ----------------------
class IVDataset(Dataset):
    def __init__(self, samples, input_scaler=None, label_scaler=None):
        self.inputs = []
        self.labels = []
        for sample in samples:
            input_seq = preprocess_sample(sample)
            self.inputs.append(input_seq)
            self.labels.append(sample["labels"])
        self.inputs = np.array(self.inputs)
        self.labels = np.array(self.labels)
        # Normalization
        if input_scaler is None:
            self.input_scaler = MinMaxScaler(feature_range=(0, 1))
            all_inputs = self.inputs.reshape(-1, 3)
            self.input_scaler.fit(all_inputs)
        else:
            self.input_scaler = input_scaler
        self.inputs = np.stack([
            self.input_scaler.transform(seq)
            for seq in self.inputs
        ])
        if label_scaler is None:
            self.label_scaler = MinMaxScaler(feature_range=(0, 1))
            self.label_scaler.fit(self.labels)
        else:
            self.label_scaler = label_scaler

        self.labels = self.label_scaler.transform(self.labels)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )


# ----------------------
# 4. LSTM model
# ----------------------
class TrapLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self._init_weights()
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def _init_weights(self):
        # LSTM initial weight
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                init.xavier_normal_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight,
                                     mode='fan_in',
                                     nonlinearity='relu')
                if layer.bias is not None:
                    init.constant_(layer.bias, 0.01)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.permute(0, 2, 1)
        lstm_out = self.pool(lstm_out).squeeze()
        output = self.fc(lstm_out)
        output = self.softplus(output)
        return output

# loss function
class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights  # weighted loss for labels
        self.base_loss = nn.MSELoss(reduction='none')

    def forward(self, inputs, targets):
        per_param_loss = self.base_loss(inputs, targets) + 0.01*torch.mean((torch.abs(inputs - targets))/((torch.abs(inputs)+torch.abs(targets))/2))
        weighted_loss = per_param_loss * self.weights

        return weighted_loss.mean()


# ----------------------
# 5. training and testing
# ----------------------
if __name__ == "__main__":
    BATCH_SIZE = 16
    EPOCHS = 1000
    TEST_RATIO = 0.02
    # simulation data loading
    samples = read_excel_data("simulation_data_address")
    train_samples, test_samples = train_test_split(
        samples, test_size=TEST_RATIO, random_state=42)
    train_dataset = IVDataset(train_samples)
    test_dataset = IVDataset(
        test_samples,
        input_scaler=train_dataset.input_scaler,
        label_scaler=train_dataset.label_scaler
    )
    # measured data loading
    real_test_samples = read_excel_data_2(
        "measured_data_address")
    real_test_dataset =IVDataset(
        real_test_samples,
        input_scaler=train_dataset.input_scaler,
        label_scaler=train_dataset.label_scaler
    )
    # creating dataset for neural network
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    real_test_loader = DataLoader(real_test_dataset, batch_size=6)
    # initailizing network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrapLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, min_lr=1e-7)
    # weights for weighted loss of labels (disabled by setting 1.0)
    param_weights = {
        'Qox': 1.0,
        'Dexp': 1.0,
        'Lambda': 1.0,
        'Dconst': 1.0
    }
    weight_values = torch.tensor([
        param_weights['Qox'],
        param_weights['Dexp'],
        param_weights['Lambda'],
        param_weights['Dconst']
    ], dtype=torch.float32).to(device)
    # loss function
    criterion = WeightedMSELoss(weights=weight_values)
    torch.autograd.set_detect_anomaly(True)
    # training for lstm
    train_loss_history = []
    test_loss_history = []
    lowest_loss = 1
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        avg_epoch_loss = train_loss / len(train_dataset)
        train_loss_history.append(avg_epoch_loss)
        if avg_epoch_loss <= lowest_loss:
            lowest_loss = avg_epoch_loss
            torch.save(model, 'model.pkl')
        scheduler.step(avg_epoch_loss)
        model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_test_loss += loss.item() * inputs.size(0)
        avg_test_loss = epoch_test_loss / len(test_dataset)
        test_loss_history.append(avg_test_loss)
        print(f"Epoch {epoch + 1}/{EPOCHS} | "f"Train Loss: {avg_epoch_loss:.4f} | "f"Test Loss: {avg_test_loss:.4f}")
    # loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), train_loss_history,
             'b-o', linewidth=2, markersize=8,
             label='Training Loss')
    plt.plot(range(1, EPOCHS + 1), test_loss_history,
             'r-o', linewidth=2, markersize=8,
             label='Training Loss')
    plt.title("Learning Curve", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss (MSE)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # testing with simulation iv
    model1 = model
    model = torch.load("model.pkl")
    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            test_preds.append(outputs.cpu().numpy())
            test_labels.append(labels.numpy())
    test_preds = np.concatenate(test_preds)
    test_labels = np.concatenate(test_labels)
    # inverse normalization
    test_preds_denorm = train_dataset.label_scaler.inverse_transform(test_preds)
    test_labels_denorm = train_dataset.label_scaler.inverse_transform(test_labels)
    # evaluation index calculation
    mask = test_labels_denorm != 0
    y_true_filtered = test_labels_denorm[mask]
    y_pred_filtered = test_preds_denorm[mask]
    mape = np.mean(np.abs(y_pred_filtered - y_true_filtered) / y_true_filtered)*100
    mae = np.mean(np.abs(test_preds_denorm - test_labels_denorm))
    print(f"Test MAPE: {mape:.4e}%, MAE: {mae:.4e}")
    # visualization
    param_names = ["Qox", "Dexp", "Lambda", "Dconst"]
    plt.figure(figsize=(12, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.scatter(test_labels_denorm[:, i], test_preds_denorm[:, i], alpha=0.6)
        plt.plot([min(test_labels_denorm[:, i]), max(test_labels_denorm[:, i])],
                 [min(test_labels_denorm[:, i]), max(test_labels_denorm[:, i])], 'r--')
        plt.xlabel("True " + param_names[i])
        plt.ylabel("Pred " + param_names[i])
    plt.tight_layout()
    plt.show()
    # testing with measured iv
    real_test_preds = []
    real_test_labels = []
    with torch.no_grad():
        for inputs, labels in real_test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            real_test_preds.append(outputs.cpu().numpy())
            real_test_labels.append(labels.numpy())
    real_test_preds = np.concatenate(real_test_preds)
    real_test_labels = np.concatenate(real_test_labels)
    # inverse normalization
    real_test_preds_denorm = train_dataset.label_scaler.inverse_transform(real_test_preds)
    real_test_labels_denorm = train_dataset.label_scaler.inverse_transform(real_test_labels)
    for i in range(len(real_test_preds_denorm)):
        print("Device"+str(i+1))
        for j in range(0,4):
            error=np.abs(real_test_preds_denorm[i][j] - real_test_labels_denorm[i][j]) / real_test_labels_denorm[i][j]*100
            print(param_names[j]+":"+str(real_test_preds_denorm[i][j])+"(predicted) "+str(real_test_labels_denorm[i][j])+"(label) "+f"{error:.4e}%")
    # evaluation index calculation
    mape = np.mean(np.abs(real_test_preds_denorm - real_test_labels_denorm) / real_test_labels_denorm)*100
    mae = np.mean(np.abs(real_test_preds_denorm - real_test_labels_denorm))
    print(f"Test MAPE: {mape:.4e}%, MAE: {mae:.4e}")