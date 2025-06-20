from scipy.io import arff
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import wandb
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import itertools
import pickle

# Dataset Class


class AdultDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature, label = self.features[idx], self.labels[idx]
        if self.transform:
            feature = self.transform(feature)
        return feature, label

# Data Preprocessing Functions


def load_arff_data(file_path):
    """Load data from ARFF file using scipy.io"""
    try:
        print(f"Reading file: {file_path}")
        # Load ARFF file using scipy.io
        data, meta = arff.loadarff(file_path)

        # Convert to pandas DataFrame
        df = pd.DataFrame(data)
        print("Dataset loaded with shape:", df.shape)

        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"File '{file_path}' not found.")
    except Exception as e:
        raise RuntimeError(f"Error loading ARFF file: {str(e)}")


def preprocess_data(data, target_column):
    """Preprocess the Adult dataset from ARFF format"""

    #  Convert byte strings to regular strings if necessary
    for col in data.select_dtypes(include=['object']):
        data[col] = data[col].str.decode('utf-8') if data[col].dtype == 'object' else data[col]

    # Separate features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Identify categorical and numerical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

    print("\nCategorical columns:", categorical_columns.tolist())
    print("Numerical columns:", numerical_columns.tolist())

    # Initialize encoders and scalers
    encoders = {"categorical": {},
                "scaler": StandardScaler(),
                "label_encoder": LabelEncoder()
                }

    # Encode categorical features
    for col in categorical_columns:
        mode_value = X.loc[X[col] != '?', col].mode()[0]
        X[col] = X[col].replace('?', mode_value)

        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])
        encoders["categorical"][col] = encoder
        print(f"Encoded {col}: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")

    # Scale numerical features
    X[numerical_columns] = encoders["scaler"].fit_transform(X[numerical_columns])
    print("\nFeature scaling completed.")

    # Encode target
    y = encoders["label_encoder"].fit_transform(y)
    print("\nTarget encoding completed. Classes:", encoders["label_encoder"].classes_)

    return X.values, y, encoders


# GAN Models
class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, output_dim, n_classes=None):
        super(Generator, self).__init__()
        self.is_conditional = n_classes is not None
        input_dim = noise_dim + (n_classes if self.is_conditional else 0)
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden_dim // 4, output_dim)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):  
                nn.init.xavier_normal_(m.weight) 
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z, labels=None):
        if self.is_conditional:
            if labels.dim() == 1:
                labels = torch.nn.functional.one_hot(labels.long(), num_classes=2).float().to(device)
            z = torch.cat([z, labels], dim=1)
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes=None):
        super(Discriminator, self).__init__()
        self.is_conditional = n_classes is not None
        input_dim = input_dim + (n_classes if self.is_conditional else 0)

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels=None):
        if self.is_conditional:
            if labels.dim() == 1:
                labels = torch.nn.functional.one_hot(labels.long(), num_classes=2).float().to(device)
            x = torch.cat([x, labels], dim=1)
        return self.model(x)


def train_discriminator(discriminator, gen_model, d_optimizer, real_data, labels, noise_dim, device, criterion, epoch, num_epochs):
    batch_size = real_data.size(0)
    real_data = real_data.to(device)
    labels = labels.to(device)

    real_labels = torch.ones(batch_size).to(device)  # Labels for real data: 1
    fake_labels = torch.zeros(batch_size).to(device)  # Labels for fake data: 0

    # add noise to real data
    real_data_noisy = real_data + 0.1 * torch.randn_like(real_data)

    # real data pass
    d_output_real = discriminator(real_data_noisy, labels)
    d_loss_real = criterion(d_output_real, real_labels.view(-1, 1)) 

    # fake data pass
    noise = torch.randn(batch_size, noise_dim).to(device)
    fake_data = gen_model(noise, labels)
    d_output_fake = discriminator(fake_data.detach(), labels)
    d_loss_fake = criterion(d_output_fake, fake_labels.view(-1, 1))

    current_d_loss = d_loss_real + d_loss_fake

    d_optimizer.zero_grad()
    current_d_loss.backward()
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
    d_optimizer.step()

    return current_d_loss


def train_generator(generator, discriminator, g_optimizer, labels, noise_dim, device, criterion):
    batch_size = labels.size(0)

    # Generate fake data
    noise = torch.randn(batch_size, noise_dim).to(device)
    fake_data = generator(noise, labels) 
    d_output_fake = discriminator(fake_data, labels)

    g_loss = criterion(d_output_fake, torch.ones(batch_size, 1).to(device))

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    return g_loss


def train_gan(gen_model, discriminator, train_loader, num_epochs, device, noise_dim, optimizer_name='Adam', learning_rate=0.0002, patience=20):
    g_losses = []
    d_losses = []
    best_g_loss = float('inf')
    no_improvement_count = 0 

    for epoch in range(num_epochs):
        gen_model.train()
        discriminator.train()

        g_loss_epoch = 0
        d_loss_epoch = 0

        for batch_idx, (real_data, labels) in enumerate(train_loader):
            # Train Discriminator

            if epoch % 10 == 0:
                d_loss = train_discriminator(discriminator, gen_model, d_optimizer, real_data, labels, noise_dim, device, criterion, epoch, num_epochs)
                d_loss_epoch += d_loss.item()
            # Train Generator
            g_loss = train_generator(gen_model, discriminator, g_optimizer, labels, noise_dim, device, criterion)
            g_loss_epoch += g_loss.item()

        g_losses.append(g_loss_epoch / len(train_loader))
        d_losses.append(d_loss_epoch / len(train_loader))
        best_loss = min(g_losses)
        if epoch % 10 == 0:
            wandb.log({"discriminator_loss": d_losses[-1]}, step=epoch+1)

        wandb.log({"generator_loss": g_losses[-1], }, step=epoch+1)

        if epoch > 0:
            if np.abs(g_losses[-2]-g_losses[-1]) < 0.00005:
                no_improvement_count += 1

        # Print loss for each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] | Gen Loss: {g_losses[-1]:.4f} | Disc Loss: {d_losses[-1]:.4f}")

        if no_improvement_count >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs with no improvement.")
            break

    return g_losses, d_losses, best_loss, epoch+1


# Generate Synthetic Data
def generate_synthetic_data(generator, n_samples, noise_dim, device, class_ratios=None):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(n_samples, noise_dim).to(device)

        if generator.is_conditional and class_ratios is not None:
            labels = []
            for class_idx, ratio in enumerate(class_ratios):
                n_class_samples = int(n_samples * ratio)
                labels.extend([class_idx] * n_class_samples)
            labels = torch.tensor(labels).to(device)
            synthetic_data = generator(noise, labels)
        else:
            synthetic_data = generator(noise)

    return synthetic_data.cpu().numpy()


def rf_detection(synthetic_data, X_train):
    
    real_data_labels = np.ones(X_train.shape[0], dtype=int)  
    fake_data_labels = np.zeros(X_train.shape[0], dtype=int)  
    X = np.vstack((X_train, synthetic_data))
    y = np.hstack((real_data_labels, fake_data_labels))

    n_folds = 4
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    auc_scores = []

    for train_index, test_index in skf.split(X, y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # print("Train class distribution:", np.bincount(y_train))
        # print("Test class distribution:", np.bincount(y_test))
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)

        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        y_pred_prob = rf.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_pred_prob)
        auc_scores.append(auc)

    average_auc = np.mean(auc_scores)
    print(f"Average AUC across {n_folds} folds: {average_auc:.4f}")
    return average_auc


def evaluate_gan_efficacy(X_train_real, X_test_real, y_train_real, y_test_real, X_synthetic, y_synthetic):

    # 1. train real data
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train_real, y_train_real)
    real_preds = rf_classifier.predict_proba(X_test_real)[:, 1]
    real_auc = roc_auc_score(y_test_real, real_preds)
    # print(f"AUC score with real data: {real_auc:.4f}")

    # 2. train synthetic data
    rf_synthetic = RandomForestClassifier(random_state=42)
    rf_synthetic.fit(X_synthetic, y_synthetic)
    synthetic_preds = rf_synthetic.predict_proba(X_test_real)[:, 1]
    synthetic_auc = roc_auc_score(y_test_real, synthetic_preds)
    print(f"AUC score with synthetic data: {synthetic_auc:.4f}")

    efficacy_ratio = synthetic_auc / real_auc
    print(f"Efficacy ratio: {efficacy_ratio:.4f}")

    return real_auc, synthetic_auc, efficacy_ratio


if __name__ == "__main__":
    # Hyperparameter combinations
    hidden_dims = [128, 256]  
    random_states = [42, 1234, 57]
    batch_sizes = [32, 64]    
    learning_rates = [0.0002, 0.0005] 
    optimizers = ['Adam', 'RMSprop']  
    criterion = torch.nn.BCEWithLogitsLoss() 
    conditional = [True, False]

    NOISE_DIM = 500
    NUM_EPOCHS = 1000 

    # Dictionary to store results
    results = {}

    # Load and preprocess data once
    data = load_arff_data('adult.arff')
    target_column = 'income'
    X, y, encoders = preprocess_data(data, target_column)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Grid search over hyperparameters
    for hidden_dim, batch_size, lr, opt_name, random_state, condition in itertools.product(
            hidden_dims, batch_sizes, learning_rates, optimizers, random_states, conditional):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

        architecture = "cGAN" if condition else "GAN"
        print(f"\nTraining with parameters:")
        print(f"Hidden Dim: {hidden_dim}, Batch Size: {batch_size}, Learning Rate: {lr}")
        print(f"Optimizer: {opt_name}, Criterion: {criterion}")
        print(f"Random state: {random_state}, Architecture: {architecture} ")

        wandb.init(
                project="GAN_new",
                name=f"{architecture}_hidden_size{hidden_dim}_b{batch_size}_lr{lr}_{opt_name}_seed{random_state}",
                config={
                    "hidden_dim": hidden_dim,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "optimizer": opt_name,
                    "random_state": random_state,
                    "architecture": architecture,
                    "noise_dim": NOISE_DIM,
                    "num_epochs": NUM_EPOCHS
                }
            )
        
        # Create dataset and dataloader
        train_dataset = AdultDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize models
        input_dim = X_train.shape[1]
        n_samples = X_train.shape[0]
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        class_ratios = class_counts / total_samples

        if condition:
            gen_model = Generator(NOISE_DIM, hidden_dim, input_dim, n_classes=2).to(device)
            discriminator = Discriminator(input_dim, hidden_dim, n_classes=2).to(device)
        else:
            gen_model = Generator(NOISE_DIM, hidden_dim, input_dim).to(device)
            discriminator = Discriminator(input_dim, hidden_dim).to(device)

        # Initialize optimizer based on selected optimizer type
        if opt_name == 'Adam':
            g_optimizer = torch.optim.Adam(gen_model.parameters(), lr=lr, betas=(0.5, 0.999))
            d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr * 0.1, betas=(0.5, 0.999))
        elif opt_name == 'RMSprop':
            g_optimizer = torch.optim.RMSprop(gen_model.parameters(), lr=lr)
            d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=lr * 0.1)

        # Train models with early stopping
        g_losses, d_losses, best_loss, epoch = train_gan(
            gen_model, discriminator, train_loader, NUM_EPOCHS, device, NOISE_DIM,
            optimizer_name=opt_name, learning_rate=lr, patience=10
        )

        if condition:
            synthetic_data = generate_synthetic_data(gen_model, n_samples, NOISE_DIM, device, class_ratios=class_ratios)
        
        else:
            synthetic_data = generate_synthetic_data(gen_model, n_samples, NOISE_DIM, device, class_ratios=None)

        average_auc = rf_detection(synthetic_data, X_train) 
        
        print("\nEvaluating Efficacy:")
        real_auc, synthetic_auc, efficacy_ratio = evaluate_gan_efficacy(
            X_train, X_test, y_train, y_test,
            synthetic_data, y_train)  

        # Store results
        config = f"Architecture{architecture}_h{hidden_dim}_b{batch_size}_lr{lr}_{opt_name}_seed{random_state}"
        results[config] = {
            'g_losses': g_losses,
            'd_losses': d_losses,
            'best_loss': best_loss,
            'params': {
                'architecture': architecture,
                'hidden_dim': hidden_dim,
                'batch_size': batch_size,
                'learning_rate': lr,
                'optimizer': opt_name,
                'criterion': criterion,
                'random state': random_state,
                'num epochs': epoch,
                'real auc': real_auc,
                'synthetic auc': synthetic_auc,
                'efficacy ratio': efficacy_ratio,
                'detection score': average_auc
            }
        }
        # Log evaluation metrics to wandb
        wandb.log({
                "num_epochs": epoch,
                "random_forest_detection_auc": average_auc,
                "real_auc": real_auc,
                "synthetic_auc": synthetic_auc,
                "efficacy_ratio": efficacy_ratio
            })

        # Finish wandb run
        wandb.finish()

    # Find best configuration
    best_config = max(results.items(), key=lambda x: x[1]['params']['efficacy ratio'])
    print("\nBest configuration:")
    print(best_config[0])
    print("Parameters:", best_config[1]['params'])
    print("Best loss:", best_config[1]['best_loss'])
