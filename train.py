"""
A Fault Diagnosis Method for Critical Rotating Components in Trains Based on Multi-Pooling Attention Convolution and an
Improved Vision Transformer
********************************************************************
*                                                                  *
* Copyright © 2025 All rights reserved                             *
* Written by Mr.XiePenghui                                         *
* [January 18,2025]                                                *
*                                                                  *
********************************************************************
"""
import os
import torch
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.optim import lr_scheduler
from MPAIT_Net import MPAIT_Net
from process import convert, save
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from visualize import (
    draw_Tr_losses, draw_Te_losses, draw_Te_acc, draw_Tr_acc,
    MLP_T_SNE, draw_LR, plot_confusion_matrix
)

# Configuration
CONSOLE_COLOR = {
    'cyan': '\033[36m',
    'end': '\033[0m'
}


class TrainingConfig:
    """Class to store and manage training configuration parameters"""

    def __init__(self, args):
        self.class_num = args.class_num
        self.num_patches = args.num_patches
        self.num_dim = args.num_dim
        self.num_depth = args.num_depth
        self.num_head = args.num_head
        self.num_mlp_dim = args.num_mlp_dim
        self.dropout = args.dropout
        self.linear_dim = args.linear_dim
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.data_path = args.train_txt
        self.pre_training = args.pre_training_weight
        self.load_weight = args.load_weight
        self.load_loss_acc = args.load_loss_acc
        self.weights = args.weights
        self.sr_ratio = args.sr_ratio
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_accuracy(model, data, label, device):
    """Calculate accuracy of model predictions"""
    model.eval()
    with torch.no_grad():
        y_pre = model(data)
        label_pre = y_pre.argmax(dim=1)
        label_true = label.argmax(axis=1)
        correct = (label_pre == label_true).sum().item()
        total = label_true.size(0)
        accuracy = correct / total
    return accuracy


def generate_classification_report(output, label):
    """Generate and print classification report"""
    label_pred = output.argmax(dim=1)
    label_true = label.argmax(axis=1)
    report = classification_report(label_true, label_pred)
    print(report)
    return report


def load_training_history(load_flag):
    """Load previous training history or initialize new lists"""
    if load_flag:
        try:
            tr_losses = np.load('Loss_Acc/Tr_loss.npy').tolist()
            te_losses = np.load('Loss_Acc/Te_loss.npy').tolist()
            tr_accuracy = np.load('Loss_Acc/Tr_acc.npy').tolist()
            te_accuracy = np.load('Loss_Acc/Te_acc.npy').tolist()
            return tr_losses, te_losses, tr_accuracy, te_accuracy
        except Exception as e:
            print(f"Error loading previous training history: {e}")
            return [], [], [], []
    else:
        return [], [], [], []


def create_data_loaders(tr_data, tr_label, batch_size):
    """Create data loaders for training"""
    dataset = TensorDataset(tr_data, tr_label)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def random_sample_data(data, label, num_samples, device):
    """Randomly sample a subset of data and labels"""
    indices = torch.randperm(len(data))[:num_samples]
    sampled_data = data[indices].to(device)
    sampled_label = label[indices].to(device)
    return sampled_data, sampled_label


def train(model, tr_data, tr_label, te_data, te_label, config):
    """Main training function with improved structure and error handling"""
    device = config.device

    # Load previous training history or initialize new lists
    tr_losses, te_losses, tr_accuracy, te_accuracy = load_training_history(config.load_loss_acc)

    # Load pre-trained weights if specified
    if config.load_weight and os.path.exists(config.pre_training):
        try:
            model.load_state_dict(torch.load(config.pre_training))
            print(f"Successfully loaded pre-trained weights from: {config.pre_training}")
        except Exception as e:
            print(f"Error loading pre-trained weights: {e}")

    # Training parameters
    best_accuracy = 0.0
    best_loss = float('inf')
    best_model_weights = None
    best_epoch = 0
    current_time = None
    learning_rates = []

    # Create data loader
    train_loader = create_data_loaders(tr_data, tr_label, config.batch_size)

    # Define optimizer, scheduler and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epoch)
    loss_func = nn.CrossEntropyLoss().to(device)

    # Training loop
    print(f"{CONSOLE_COLOR['cyan']}Starting training on {device}{CONSOLE_COLOR['end']}")
    for epoch in range(1, config.epoch + 1):
        model.train()
        tqdm_loader = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{config.epoch}",
            bar_format=f"{CONSOLE_COLOR['cyan']}{{l_bar}}{{bar}}{{r_bar}}{CONSOLE_COLOR['end']}",
            unit="batch"
        )

        # Batch training
        for step, (batch_x, batch_y) in enumerate(tqdm_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass
            tr_output = model(batch_x)
            tr_loss = loss_func(tr_output, batch_y)

            # Backward pass
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()

            # Evaluation at the end of each epoch
            if step == len(tqdm_loader) - 1:
                model.eval()
                try:
                    with torch.no_grad():
                        # Sample data for evaluation
                        num_samples = min(783, len(tr_data), len(te_data))

                        # Sample and evaluate training data
                        sampled_tr_data, sampled_tr_label = random_sample_data(
                            tr_data, tr_label, num_samples, device
                        )

                        # Sample and evaluate test data
                        sampled_te_data, sampled_te_label = random_sample_data(
                            te_data, te_label, num_samples, device
                        )

                        # Forward pass for evaluation
                        tr_output = model(sampled_tr_data)
                        tr_loss = loss_func(tr_output, sampled_tr_label)
                        te_output = model(sampled_te_data)
                        te_loss = loss_func(te_output, sampled_te_label)

                        # Record losses
                        tr_losses.append(tr_loss.item())
                        te_losses.append(te_loss.item())

                        # Calculate and record accuracies
                        tr_acc = calculate_accuracy(model, sampled_tr_data, sampled_tr_label, device)
                        te_acc = calculate_accuracy(model, sampled_te_data, sampled_te_label, device)
                        tr_accuracy.append(tr_acc)
                        te_accuracy.append(te_acc)

                        # Update progress bar
                        tqdm_loader.set_postfix(
                            Tr_loss=tr_loss.item(),
                            Te_loss=te_loss.item(),
                            Tr_acc=tr_acc,
                            Te_acc=te_acc
                        )

                        # Check if current model is the best
                        if te_acc >= best_accuracy and te_loss.item() <= best_loss:
                            best_accuracy = te_acc
                            best_loss = te_loss.item()
                            best_model_weights = model.state_dict()
                            best_epoch = epoch
                            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                except Exception as e:
                    print(f"Error during evaluation: {e}")

        # Record learning rate and update scheduler
        learning_rates.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    # Final evaluation and visualization
    try:
        # T-SNE visualization
        num_samples = min(783, len(te_data))
        sampled_te_data, sampled_te_label = random_sample_data(te_data, te_label, num_samples, device)

        model.eval()
        with torch.no_grad():
            te_output = model(sampled_te_data)

        # Calculate final accuracy
        label_pred = te_output.argmax(dim=1)
        label_true = sampled_te_label.argmax(axis=1)
        correct = (label_pred == label_true).sum().item()
        total = label_true.size(0)
        final_accuracy = correct / total
        print(f"Final Accuracy: {final_accuracy:.4f}")

        # T-SNE visualization
        MLP_T_SNE(te_output, sampled_te_label)

        # Confusion matrix
        te_output_cpu = te_output.cpu()
        sampled_te_label_cpu = sampled_te_label.cpu()
        label_pred = te_output_cpu.argmax(dim=1)
        label_true = sampled_te_label_cpu.argmax(axis=1)
        confusion_mat = confusion_matrix(label_true, label_pred)
        plot_confusion_matrix(confusion_mat, classes=range(config.class_num))

        # Classification report
        generate_classification_report(te_output_cpu, sampled_te_label_cpu)
    except Exception as e:
        print(f"Error during final evaluation: {e}")

    # Save best model weights
    if best_model_weights is not None:
        try:
            os.makedirs(config.weights, exist_ok=True)
            model_path = os.path.join(
                config.weights,
                f'Model_{best_epoch}_{current_time}.pth'
            )
            torch.save(best_model_weights, model_path)
            print(f"Best model weights saved to: {model_path}")
        except Exception as e:
            print(f"Error saving model weights: {e}")

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save training history
    try:
        save(tr_losses, te_losses, tr_accuracy, te_accuracy)

        # Plot training curves
        draw_Tr_losses(tr_losses)
        draw_Te_losses(te_losses)
        draw_Tr_acc(tr_accuracy)
        draw_Te_acc(te_accuracy)
        draw_LR(learning_rates)
    except Exception as e:
        print(f"Error during visualization: {e}")


def parse_arguments():
    """Parse command line arguments with improved descriptions"""
    parser = argparse.ArgumentParser(description="Train a MPAIT-Net model for bearing fault diagnosis")

    # Model parameters
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument("--class_num", type=int, default=10, help="Number of output classes")
    model_group.add_argument("--num_patches", type=int, default=49, help="Number of image patches")
    model_group.add_argument("--num_dim", type=int, default=192, help="Embedding dimension for patches")
    model_group.add_argument("--num_depth", type=int, default=1, help="Number of transformer layers")
    model_group.add_argument("--num_head", type=int, default=8, help="Number of attention heads")
    model_group.add_argument("--num_mlp_dim", type=int, default=768, help="Dimension of MLP hidden layer")
    model_group.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    model_group.add_argument("--linear_dim", type=int, default=64, help="Dimension of FC hidden layer")
    model_group.add_argument("--sr_ratio", type=int, default=1, help="Spatial reduction ratio")

    # Training parameters
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument("--epoch", type=int, default=30, help="Number of training epochs")
    train_group.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    train_group.add_argument("--learning_rate", type=float, default=0.0004, help="Initial learning rate")

    # Data and file paths
    path_group = parser.add_argument_group('Path Parameters')
    path_group.add_argument("--train_txt", type=str, default=r'predata', help="Path to training data directory")
    path_group.add_argument("--pre_training_weight", type=str, default="", help="Path to pre-trained weights")
    path_group.add_argument("--weights", type=str, default="./weights/", help="Directory to save model weights")

    # Control flags
    flag_group = parser.add_argument_group('Control Flags')
    flag_group.add_argument("--load_weight", type=bool, default=False, help="Whether to load pre-trained weights")
    flag_group.add_argument("--load_loss_acc", type=bool, default=False,
                            help="Whether to continue from previous training")

    return parser.parse_args()


def load_and_preprocess_data(config):
    """Load and preprocess training data"""
    try:
        # Load data
        data_path = config.data_path
        data = np.load(os.path.join(data_path, 'data_-2.npy'))
        label = np.load(os.path.join(data_path, 'label_-2.npy'))

        # Split into train and test sets
        data_train, data_test, label_train, label_test = train_test_split(
            data, label, test_size=0.3, random_state=42
        )

        # Convert data to appropriate format
        data_train, data_test, label_train, label_test = convert(
            data_train, data_test, label_train, label_test
        )

        return data_train, data_test, label_train, label_test
    except Exception as e:
        print(f"Error loading or preprocessing data: {e}")
        raise


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    config = TrainingConfig(args)

    # Print configuration
    print(f"{CONSOLE_COLOR['cyan']}Configuration:{CONSOLE_COLOR['end']}")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")

    # Load and preprocess data
    data_train, data_test, label_train, label_test = load_and_preprocess_data(config)

    # Initialize model
    model = MPAIT_Net(
        num_classes=config.class_num,
        num_patches=config.num_patches,
        dim=config.num_dim,
        depth=config.num_depth,
        heads=config.num_head,
        mlp_dim=config.num_mlp_dim,
        linear_dim=config.linear_dim,
        dropout=config.dropout,
        sr_ratio=config.sr_ratio
    ).double().to(config.device)

    # Train model
    train(model, data_train, label_train, data_test, label_test, config)


if __name__ == "__main__":
    main()
