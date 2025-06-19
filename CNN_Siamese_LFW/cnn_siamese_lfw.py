import torch
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import time
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
from torchvision.utils import make_grid
import seaborn as sns
from torch.nn import functional as F
from itertools import product
import wandb
import pickle


class ImageDataset(Dataset):
    """
    A custom Dataset class for loading and preprocessing facial image pairs for Siamese network training.

    This dataset handles both positive pairs (same person) and negative pairs (different people),
    with optional image resizing based on architecture requirements.

    Args:
        text_dir (str): Path to the text file containing image pair information
        img_dir (str): Directory containing the image files
        arch (str, optional): Architecture type ('paper' or None). Determines image resizing
        img_size (int, optional): Target size for image resizing if arch=='paper'. Defaults to 105

    Returns:
        tuple: Contains:
            - tuple: (name (str), image1 tensor (torch.Tensor))
            - tuple: (name (str), image2 tensor (torch.Tensor))
            - label (torch.Tensor): 1.0 for same person, 0.0 for different people
    """

    def __init__(self, text_dir, img_dir, arch=None, img_size=105):
        self.img_dir = img_dir
        self.img_size = img_size if arch == "paper" else None  # resize only for "paper" architecture

        self.lst_pairs = self.read_names(text_dir)

        self.transform = transforms.Compose([
                            transforms.Resize((self.img_size, self.img_size)) if self.img_size else transforms.Lambda(lambda x: x), 
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])
        self.load_and_transform(self.lst_pairs[0][0], self.lst_pairs[0][1])

    def __len__(self):
        return len(self.lst_pairs)
    
    def __getitem__(self, idx):
        return self.read_images(self.lst_pairs[idx])

    def path(self, name, pic_num):
        return f'lfw2/{name}/{name}_{pic_num.zfill(4)}.jpg'
    
    def read_names(self, text_path):
        lst_pairs = []
        try:
            with open(text_path, 'r') as f:
                next(f)  # Skip the first line
                for line in f:
                    parts = line.rstrip('\n').split('\t')
                    lst_pairs.append(parts)
        except FileNotFoundError:
            print(f"Error: File {text_path} not found.")
        except Exception as e:
            print(f"An error occurred while reading {text_path}: {e}")
        return lst_pairs

    def read_images(self, lst_pairs):
        try:
            if len(lst_pairs) == 3:  # positive sample
                img1 = self.load_and_transform(lst_pairs[0], lst_pairs[1])
                img2 = self.load_and_transform(lst_pairs[0], lst_pairs[2])
                label = torch.tensor([1.])
            elif len(lst_pairs) == 4:  # negative sample
                img1 = self.load_and_transform(lst_pairs[0], lst_pairs[1])
                img2 = self.load_and_transform(lst_pairs[2], lst_pairs[3])
                label = torch.tensor([0.])
            else:
                raise ValueError("The sample should contain 3 or 4 elements")
            
            return (img1, img2, label)
        except Exception as e:
            print(f"Error processing image pair: {e}")
            return None

    def load_and_transform(self, name, pic_num):
        img_path = os.path.join(self.img_dir, self.path(name, pic_num))
        try:
            img = Image.open(img_path)
            return name, self.transform(img)
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}")
            return torch.zeros((1, self.img_size, self.img_size))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros((1, self.img_size, self.img_size))


def analyze_lfw_dataset(root_dir):

    images_per_person = []
    
    for person_name in os.listdir(root_dir):
        person_dir = os.path.join(root_dir, person_name)

        if os.path.isdir(person_dir):
            num_images = len([f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            images_per_person.append(num_images)

    data_stats = {
        'total_images': sum(images_per_person),
        'total_people': len(images_per_person),
        'avg_images': np.mean(images_per_person),
        'median_images': np.median(images_per_person),
        'min_images': min(images_per_person),
        'max_images': max(images_per_person),
        'single_image_count': sum(1 for x in images_per_person if x == 1)
    }

    return data_stats


class Siamese_model(nn.Module):
    """
    Implementation of Siamese Neural Network for facial recognition.

    Supports three architectures:
    - 'paper': Original architecture from the paper
    - 'new_arch_1': Modified architecture with smaller kernels and adaptive pooling
    - 'new_arch_2': Similar to new_arch_1 with added BatchNorm layers

    Args:
        arch (str): Architecture type ('paper', 'new_arch_1', or 'new_arch_2')

    Returns:
        torch.Tensor: Similarity score between two input images (0 to 1)
    """
    def __init__(self, arch):
        super(Siamese_model, self).__init__()

        if arch == "paper":
            self.conv = nn.Sequential(
            nn.Conv2d(1, 64, (10, 10)), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, (7, 7)), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, (4, 4)), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, (4, 4)), nn.ReLU(),
            )
            self.vec = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
            self.output = nn.Sequential(nn.Linear(4096, 1), nn.Sigmoid())

            self.conv.apply(self.initialize_weights)
            self.vec.apply(self.initialize_weights)
            self.output.apply(self.initialize_weights)
            self.method = 'paper'

        if arch == "new_arch_1":
            self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 5)), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=(5, 5)), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=(3, 3)), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=(3, 3)), nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6))

        )
            self.vec = nn.Sequential(nn.Linear(6 * 6 * 256, 2048), nn.ReLU())
            self.output = nn.Sequential(nn.Linear(2048, 1), nn.Sigmoid())
            
            self.conv.apply(self.initialize_weights_xavier)
            self.vec.apply(self.initialize_weights_xavier)
            self.output.apply(self.initialize_weights_xavier)
            self.method = 'xavier'

        if arch == "new_arch_2":
            self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 5)), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=(5, 5)), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=(3, 3)), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=(3, 3)), nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6))

        )
            self.vec = nn.Sequential(nn.Linear(6 * 6 * 256, 2048), nn.ReLU(), nn.BatchNorm1d(2048))
            self.output = nn.Sequential(nn.Linear(2048, 1), nn.Sigmoid())
            
            self.conv.apply(self.initialize_weights_xavier)
            self.vec.apply(self.initialize_weights_xavier)
            self.output.apply(self.initialize_weights_xavier)
            self.method = 'xavier'

    def forward(self, img1, img2):
        input1, input2 = self.conv(img1), self.conv(img2)
        input1, input2 = input1.view(input1.shape[0], -1), input2.view(input2.shape[0], -1)
        input1, input2 = self.vec(input1), self.vec(input2)

        result = self.output(torch.abs(input1-input2))

        return result
         
    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
          torch.nn.init.normal_(m.weight, 0, 0.01)
          m.bias.data.normal_(0, 0.01)
        if isinstance(m, nn.Linear):
          torch.nn.init.normal_(m.weight, 0, 0.5)

    def initialize_weights_xavier(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data) 
            nn.init.normal_(m.bias.data)
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data) 
            nn.init.normal_(m.bias.data)


def weight_reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def save_example_images(y, preds, x1, x2, arch_name, counters, step):

    conditions = {
            (1, 1): ("TP", "True Positive"),
            (0, 0): ("TN", "True Negative"),
            (0, 1): ("FP", "False Positive"),
            (1, 0): ("FN", "False Negative")
        }
    
    for i in range(len(y)):
        condition = (y[i].item(), preds[i].item())
        if condition in conditions:
            label, _ = conditions[condition]
            counters[label] += 1
            example_counter_key = f"{label}_example_counter"
            if (step == 0 or step == 1) and counters[example_counter_key] <= 1:
                path = f"{arch_name}/{label}_examples_{counters[label]}"
                os.makedirs(path, exist_ok=True)
                save_image(x1[1][i], os.path.join(path, f"{label}_example_img_1_{i+1}_{x1[0][i]}.jpg"))
                save_image(x2[1][i], os.path.join(path, f"{label}_example_img_2_{i+1}_{x2[0][i]}.jpg"))
                counters[example_counter_key] += 1

    return counters


def save_confusion_matrix(confusion_matrix, arch_name):
    
    os.makedirs(arch_name, exist_ok=True)
    labels = ['Positive', 'Negative']
    plt.ioff()
    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {arch_name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    save_path = os.path.join(arch_name, "confusion_matrix.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def train(dataloader, model, criterion, optimizer, epoch, device, bool=True):
    """
    Training loop for one epoch of model training with L2 regularization.

    This function performs one complete training epoch, updating model weights using
    backpropagation and tracking various metrics like loss and accuracy.

    Args:
        dataloader (DataLoader): PyTorch DataLoader containing the training data
        model (nn.Module): The neural network model to train
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimization algorithm
        epoch (int): Current epoch number (for logging)
        device (str): Device to run training on ('cuda' or 'cpu')
        bool (bool, optional): Whether to print intermediate results. Defaults to True

    Returns:
        tuple: Contains:
            - avg_loss (float): Average loss over the entire epoch
            - avg_acc (float): Average accuracy over the entire epoch

    Note:
        - Uses L2 regularization with lambda=0.00001
        - Prints training progress every 5 steps if bool=True
        - Accuracy is calculated using 0.5 as threshold for binary classification
    """
    
    model.train()
    
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for step, (x1, x2, y) in enumerate(dataloader):
        x1[1], x2[1], y = x1[1].to(device), x2[1].to(device), y.to(device)
        batch_size = y.size(0)

        pred = model(x1[1], x2[1])
        loss = criterion(pred, y)

        l2_lambda = 0.00001
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss_with_l2 = loss + l2_lambda * l2_norm
        
        optimizer.zero_grad()

        loss_with_l2.backward()
        optimizer.step()
        
        preds = (pred > 0.5).float()
        acc = (preds == y).float().mean()
        
        total_loss += loss.item() * batch_size
        total_acc += acc.item() * batch_size
        total_samples += batch_size

        if bool and (step + 1) % 5 == 0:
            avg_loss = total_loss / total_samples
            avg_acc = total_acc / total_samples
            print(f'Step {step+1}, Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_acc:.4f}')

    avg_loss = total_loss / total_samples 
    avg_acc = total_acc / total_samples  
    print(f'\nEpoch {epoch+1} Completed, Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_acc:.4f}\n')
    return avg_loss, avg_acc


def val(dataloader, model, criterion, device):
    model.eval()
    
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    with torch.no_grad():
        for step, (x1, x2, y) in enumerate (dataloader):
            x1[1], x2[1], y = x1[1].to(device), x2[1].to(device), y.to(device)

            pred = model(x1[1], x2[1])
            loss = criterion(pred, y)        
            preds = (pred > 0.5).float()
            acc = (preds == y).float().mean()
            
            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_acc += acc.item() * batch_size
            total_samples += batch_size
                    
    avg_loss = total_loss / total_samples 
    avg_acc = total_acc / total_samples  
    return avg_loss, avg_acc


def test(dataloader, model, criterion, device, bool=True, arch_name=None, report=True):
    """
    Evaluates model performance on a dataset and generates detailed metrics.

    Args:
        dataloader (DataLoader): DataLoader for evaluation
        model (nn.Module): The neural network model
        criterion (nn.Module): Loss function
        device (str): Device to run evaluation on ('cuda' or 'cpu')
        bool (bool, optional): Whether to print results. Defaults to True
        arch_name (str, optional): Architecture name for saving results
        report (bool, optional): Whether to generate detailed report. Defaults to True

    Returns:
        tuple: Contains:
            - avg_acc (float): Average accuracy
            - avg_loss (float): Average loss
            - total_confusion_matrix (np.ndarray): Confusion matrix
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    total_confusion_matrix = np.zeros((2, 2), dtype=int)
    with torch.no_grad():
        for step, (x1, x2, y) in enumerate(dataloader):
            x1[1], x2[1], y = x1[1].to(device), x2[1].to(device), y.to(device)

            batch_size = y.size(0)
            pred = model(x1[1], x2[1])
            loss = criterion(pred, y)
        
            total_loss += loss.item() * batch_size
            preds = (pred > 0.5).float()  
            total_acc += (preds == y).float().sum().item()
            total_samples += batch_size

            if report:
                x1[1] = x1[1]*0.5 + 0.5 
                x2[1] = x2[1]*0.5 + 0.5 
                counters = {
                                        "TP": 0,
                                        "TN": 0,
                                        "FP": 0,
                                        "FN": 0,
                                        "TP_example_counter": 0,
                                        "TN_example_counter": 0,
                                        "FP_example_counter": 0,
                                        "FN_example_counter": 0,
                                    }
                counters = save_example_images(y, preds, x1, x2, arch_name, counters, step)
                confusion_matrix = np.array([[counters["TP"], counters["FP"]], [counters["FN"], counters["TN"]]], dtype=int)
                total_confusion_matrix += confusion_matrix

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / total_samples
    if bool:
        print(f"---------------\nTest Summary\n Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.4f}\n")

    return avg_acc, avg_loss, total_confusion_matrix


def experiment(train_loader, val_loader, test_loader, model, criterion, optimizer, epochs, device, exp_name):
    """
    Runs a complete training experiment including training, validation, and testing phases.

    Args:
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        test_loader (DataLoader): DataLoader for test data
        model (nn.Module): The neural network model
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimization algorithm
        epochs (int): Maximum number of training epochs
        device (str): Device to run the experiment on ('cuda' or 'cpu')
        exp_name (str): Name of the experiment for logging

    Returns:
        tuple: Contains:
            - test_loss (float): Final loss on test set
            - test_acc (float): Final accuracy on test set
            - conf_matrix (np.ndarray): Confusion matrix on test set
    """
    model.to(device)
    
    avg_loss_train_prev = 0
    patience = 5
    no_improvement_epochs = 0
    tolerance = 0.001 

    start_time = time.time()
    for epoch in range(epochs):
        
        print(f"Epoch {epoch+1}/{epochs} for {exp_name}")
        avg_loss_train, avg_acc_train = train(train_loader, model, criterion, optimizer, epoch, device)

        wandb.log({"train_loss": avg_loss_train, "train_accuracy": avg_acc_train}, step=epoch+1)

        if avg_loss_train - avg_loss_train_prev < tolerance:
            no_improvement_epochs += 1
            if no_improvement_epochs >= patience:
                print("Early stopping: No significant improvement")
                break
        else:
            no_improvement_epochs = 0 

        avg_loss_train_prev = avg_loss_train

    end_time = time.time()
    run_time = end_time - start_time    
    print(f'Runtime: {run_time}')
    wandb.log({"training_runtime": run_time})

    avg_loss_val, avg_acc_val = val(val_loader, model, criterion, device)
    wandb.log({
        "val_loss": avg_loss_val,
        "val_accuracy": avg_acc_val
    })
    print(f'Validation results: Val Loss: {avg_loss_val:.4f}, Val accuracy: {avg_acc_val:.4f}\n')

    print(f"Testing {exp_name}")
    test_acc, test_loss, conf_matrix = test(test_loader, model, criterion, device, report=True, arch_name=exp_name)
    
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_acc
    })
    print(f'Test results: Test Loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}\n')

    wandb.finish()
    return test_loss, test_acc, conf_matrix


def grid_search(train_dataset, val_dataset, test_dataset, architectures, learning_rates, batch_sizes, optimizers, epochs, device):
    """
    Performs grid search over specified hyperparameters to find the best model configuration.

    Args:
        train_dataset (Dataset): Training dataset
        val_dataset (Dataset): Validation dataset
        test_dataset (Dataset): Test dataset
        architectures (list): List of architecture names to try
        learning_rates (list): List of learning rates to try
        batch_sizes (list): List of batch sizes to try
        optimizers (list): List of optimizer classes to try
        epochs (int): Maximum number of epochs for each trial
        device (str): Device to run the search on ('cuda' or 'cpu')

    Returns:
        tuple: Contains:
            - best_exp (str): Name of the best experiment configuration
            - list: [best_loss, best_acc, best_confusion_matrix]
    """
    results = {}
    best_loss = float("inf")
    best_acc = 0.0
    best_confusion_matrix = None
    num_experiment = 0

    for arch, lr, batch_size, optimizer_class in product(architectures, learning_rates, batch_sizes, optimizers):
        num_experiment += 1
        model = Siamese_model(arch=arch).to(device)

        criterion = torch.nn.BCELoss()

        optimizer = optimizer_class(model.parameters(), lr=lr)

        exp_name = f"Arch-{arch}_LR-{lr}_Batch-{batch_size}_Opt-{optimizer_class.__name__}"

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        print(f'Experiment No.{num_experiment} -- {exp_name}\n')

        os.environ["WANDB_PROGRAM"] = "Exercise_2_new.py"
        wandb.init(project="siamese_network", name=exp_name, tags=["grid_search", arch])
        wandb.config.update({
            "learning_rate": lr,
            "batch_size": batch_size,
            "optimizer": optimizer_class.__name__,
            "architecture": arch
        })
        test_loss, test_acc, conf_matrix = experiment(train_loader, val_loader, test_loader, model, criterion, optimizer, epochs, device, exp_name)

        results[exp_name] = (test_loss, test_acc, conf_matrix)

        if test_loss < best_loss or test_acc > best_acc:
            print(f"New best model found for {exp_name} with Loss: {test_loss:.4f}, Acc: {test_acc:.4f}\n")
            best_loss = test_loss
            best_acc = test_acc
            best_confusion_matrix = conf_matrix
            best_exp = exp_name
        save_confusion_matrix(best_confusion_matrix, best_exp)
    return best_exp, [best_loss, best_acc, best_confusion_matrix]


def get_best_architecture(results):
    best_arch = None
    best_metrics = None
    best_acc = 0.0
    best_loss = float('inf')

    for arch, metrics in results.items():
        test_loss, test_acc, _ = metrics

        if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
            best_arch = arch
            best_metrics = metrics
            best_acc = test_acc
            best_loss = test_loss
    with open('my_dict.pkl', 'wb') as f:
        pickle.dump(results, f)
    return best_arch, best_metrics


text_dir_train = r'd:/Niv/Deep Learning/Exercise 2/lfw2/pairsDevTrain.txt'
text_dir_test = r'd:/Niv/Deep Learning/Exercise 2/lfw2/pairsDevTest.txt'
image_folder = r'd:/Niv/Deep Learning/Exercise 2/lfw2/'
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device\n")

architectures = ["paper", "new_arch_1", "new_arch_2"]
learning_rates = [1e-3, 1e-4, 1e-5]
batch_sizes = [32, 64, 128]
optimizers = [torch.optim.Adam]
epochs = 50

best_arch = {}

for arch in architectures:
    print(f"Testing architecture: {arch}")

    train_dataset = ImageDataset(text_dir_train, image_folder, arch=arch)
    train_pairs, val_pairs = train_test_split(train_dataset.lst_pairs, test_size=0.2, random_state=42)
    train_dataset.lst_pairs = train_pairs

    val_dataset = ImageDataset(text_dir_train, image_folder, arch=arch)
    val_dataset.lst_pairs = val_pairs

    test_dataset = ImageDataset(text_dir_test, image_folder, arch=arch)

    best_exp, metrics = grid_search(train_dataset, val_dataset, test_dataset, [arch], learning_rates, batch_sizes, optimizers, epochs, device)
    print(f"Results for {arch}: {metrics}")
    best_arch[best_exp] = metrics


best_architecture, best_metrics = get_best_architecture(best_arch)

print(f"Best Architecture: {best_architecture}")
print(f"Best Metrics: {best_metrics}")


stats_path = r'd:/Niv/Deep Learning/Exercise 2/lfw2/lfw2'
stats = analyze_lfw_dataset(stats_path)

print("\nLFW Dataset Statistics:\n")
print(f"Total number of images: {stats['total_images']}")
print(f"Total number of people: {stats['total_people']}")
print(f"Average images per person: {stats['avg_images']:.2f}")
print(f"Median images per person: {stats['median_images']:.2f}")
print(f"Minimum images for a person: {stats['min_images']}")
print(f"Maximum images for a person: {stats['max_images']}")
print(f"Number of people with only one image: {stats['single_image_count']}")
