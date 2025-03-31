import argparse
import logging
import os
import optuna
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import random_split, Subset
from torch import GradScaler, autocast
from torch_geometric import utils
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, roc_auc_score  # For OGB evaluation metrics


# Import OGB dataset utilities
try:
    from ogb.graphproppred import GraphPropPredDataset, Evaluator
    OGB_AVAILABLE = True
except ImportError:
    OGB_AVAILABLE = False
    print("Warning: OGB package not found. Install with 'pip install ogb' to use OGB datasets.")


from networks import Net
from utils import EarlyStopping, ModelSaveCallback
from utils import plotter, sample_dataset


def arg_parse(args = None):
    parser = argparse.ArgumentParser(description='JacobiPool')
    parser.add_argument('--dataset', type=str, default='NCI1', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ogbg-molhiv/ogbg-molpcba')
    parser.add_argument('--epochs', type=int, default=300, help='maximum number of epochs')
    parser.add_argument('--seed', type=int, default=777, help='seed')
    parser.add_argument('--device', type=str, default='cpu', help='device selection: cuda or cpu')
    parser.add_argument('--batch_size', type=int, default=16 , help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--approx_func', type=str, default='jacobi', help='desired approximation function (e.g. jacobi, chebyshev)')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--num_hidden', type=int, default=64, help='hidden size')
    parser.add_argument('--pooling_ratio', type=float, default=0.8, help='pooling ratio')
    parser.add_argument('--dropout_ratio', type=float, default=0.4, help='dropout ratio')
    parser.add_argument('--num_heads', type=int, default=2, help="number of hidden attention heads")
    parser.add_argument("--hop_num", type=int, default=3, help="hop number")
    parser.add_argument("--p_norm", type=int, default=0.0, help="p_norm")
    parser.add_argument('--early_stop', default=False, help="indicates whether to use early stop or not")
    parser.add_argument('--patience', type=int, default=30, help='patience for earlystopping')
    parser.add_argument('--verbose', type=int, default=1, help='level of verbosity: 0: Just the important outputs, 1: Partial verbosity including model training per epoch, 2: Complete verbosity and printing all INFOs')
    parser.add_argument('--hyptune', type=int, default=0, help='whether you want Optuna find the best hyperparameters (1 for hyperparameter tunning)')
    parser.add_argument('--sample_size', type=float, default=-1, help='if want to train on a subset of dataset, specify the number of samples')
    parser.add_argument('--colab', type=bool, default=True, help='Indicate whether you are using Google Colab')
    parser.add_argument('--a', type=float, default=1.0, help='Jacobi hyperparameter a')
    parser.add_argument('--b', type=float, default=1.0, help='Jacobi hyperparameter b')
    parser.add_argument('--test_only', action='store_true', help='Skip training and only test using saved model')
    args = parser.parse_args(args)
    return args

def test(model, loader, args):
    '''
    Evaluation and test function
    Inputs:
        - model: PyTorch model
        - loader: dataloader corresponding to evaluation
        - args: specified arguments
    Outputs:
        - accuracy/score, loss
    '''
    model.eval()
    correct = 0
    loss = 0
    y_true = []
    y_pred = []
    y_scores = []
    
    # Debug: Print dataset size
    # print(f"Evaluating on dataset with {len(loader.dataset)} samples")
    
    # Select loss function based on dataset
    if args.dataset.startswith('ogbg-molpcba'):
        loss_fcn = torch.nn.BCEWithLogitsLoss()
    elif args.dataset == 'ogbg-molhiv':
        # Calculate positive weight for binary classification to handle class imbalance
        pos_weight = torch.tensor([2.0]).to(args.device)  # Initial estimate, will be refined during training
        loss_fcn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fcn = torch.nn.CrossEntropyLoss()
    
    # Debug: Count positive samples
    total_positive = 0
    total_samples = 0
    
    with torch.no_grad():  # Ensure we don't store gradients during testing
        for data in loader:
            data = data.to(args.device)
            out = model(data)
            
            # Handle different dataset formats
            if args.dataset.startswith('ogbg-mol'):
                if args.dataset == 'ogbg-molhiv':
                    # Binary classification
                    y = data.y.float()
                    
                    if out.ndim == 1 and y.ndim == 2:
                        y = y.squeeze(1)
                    elif out.ndim == 2 and y.ndim == 1:
                        y = y.unsqueeze(1)
                    
                    # Debug: Count positive samples
                    total_samples += y.shape[0]
                    total_positive += y.sum().item()
                    
                    loss += loss_fcn(out, y).item()
                    pred = (out > 0).float()
                    y_true.append(y.detach().cpu())
                    y_scores.append(out.detach().cpu())
                    y_pred.append(pred.detach().cpu())
                elif args.dataset == 'ogbg-molpcba':
                    y = data.y.float()
                    # Skip NaN targets
                    is_valid = ~torch.isnan(y)
                    loss += loss_fcn(out[is_valid], y[is_valid]).item()
                    y_true.append(y.detach().cpu())
                    y_scores.append(out.detach().cpu())
            else:
                pred = out.max(dim=1)[1]
                correct += pred.eq(data.y).sum().item()
                loss += loss_fcn(out, data.y).item()
    
    if args.dataset == 'ogbg-molhiv':
        # ROC-AUC for binary classification
        y_true = torch.cat(y_true, dim=0).numpy()
        y_scores = torch.cat(y_scores, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        
        # Debug: Print class distribution
        # positive_count = np.sum(y_true == 1)
        # percent_positive = (positive_count / len(y_true)) * 100
        # print(f"Class distribution: {positive_count}/{len(y_true)} positive samples ({percent_positive:.2f}%)")
        # print(f"Positive rate in processed dataset: {total_positive}/{total_samples} ({(total_positive/total_samples)*100:.2f}%)")
        
        # # Debug: Print prediction distribution
        # pred_positive = np.sum(y_pred == 1)
        # pred_percent = (pred_positive / len(y_pred)) * 100
        # print(f"Prediction distribution: {pred_positive}/{len(y_pred)} positive predictions ({pred_percent:.2f}%)")
        
        # # Debug: Print model output stats
        # print(f"Model output stats: min={y_scores.min():.4f}, max={y_scores.max():.4f}, mean={y_scores.mean():.4f}, std={y_scores.std():.4f}")
        
        if OGB_AVAILABLE:
            # Ensure inputs are 2D arrays as required by OGB evaluator
            if y_true.ndim == 1:
                y_true = y_true.reshape(-1, 1)
            if y_scores.ndim == 1:
                y_scores = y_scores.reshape(-1, 1)
                
            evaluator = Evaluator(name='ogbg-molhiv')
            input_dict = {"y_true": y_true, "y_pred": y_scores}
            result_dict = evaluator.eval(input_dict)
            score = result_dict["rocauc"]
            
            # Debug: Manual calculation to verify
            try:
                manual_roc = roc_auc_score(y_true, y_scores)
                print(f"Manual ROC-AUC calculation: {manual_roc:.4f} (OGB evaluator: {score:.4f})")
            except Exception as e:
                print(f"Error in manual ROC-AUC calculation: {e}")
        else:
            score = roc_auc_score(y_true, y_scores)
            
        return score, loss / len(loader)
    
    elif args.dataset == 'ogbg-molpcba':
        # Average precision for multi-label classification
        y_true = torch.cat(y_true, dim=0).numpy()
        y_scores = torch.cat(y_scores, dim=0).numpy()
        
        if OGB_AVAILABLE:
            # Ensure inputs are 2D arrays (already the case for molpcba, but check to be safe)
            if y_true.ndim == 1:
                y_true = y_true.reshape(-1, 1)
            if y_scores.ndim == 1:
                y_scores = y_scores.reshape(-1, 1)
                
            evaluator = Evaluator(name='ogbg-molpcba')
            input_dict = {"y_true": y_true, "y_pred": y_scores}
            result_dict = evaluator.eval(input_dict)
            score = result_dict["ap"]
        else:
            # Fallback to sklearn for AP calculation
            valid_mask = ~np.isnan(y_true)
            score = average_precision_score(y_true[valid_mask], y_scores[valid_mask], average='micro')
            
        return score, loss / len(loader)
    
    else:
        # For TU datasets, we're already scaling by the dataset size
        # We could add debugging to verify the calculation if needed
        # print(f"Test Loss: {loss}, Dataset size: {len(loader.dataset)}, Final Loss: {loss / len(loader.dataset)}")
        return correct / len(loader.dataset), loss / len(loader.dataset)

def model_train(args, train_loader, val_loader):
    model = Net(args).to(args.device)
    
    # Set the task type in the model to handle different output requirements
    if hasattr(args, 'task_type'):
        model.task_type = args.task_type

    print(f'[INFO]: Model architecture:\n{model}')

    # Print parameter initialization stats
    # print("[INFO]: Checking model initialization...")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"  {name}: min={param.data.min().item():.4f}, max={param.data.max().item():.4f}, mean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.dataset.startswith('ogbg-mol'):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',           # Maximize metrics for OGB datasets
            factor=0.5,           # Multiply LR by this factor when reducing
            patience=10,          # Number of epochs with no improvement after which LR will be reduced
            verbose=True,         # Print message when LR is reduced
            min_lr=1e-6           # Lower bound on the learning rate
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',           # Minimize validation loss for TU datasets
            factor=0.5,           # Multiply LR by this factor when reducing
            patience=10,          # Number of epochs with no improvement after which LR will be reduced
            verbose=True,         # Print message when LR is reduced
            min_lr=1e-6           # Lower bound on the learning rate
        )
    
    # Select loss function based on dataset
    if args.dataset.startswith('ogbg-molpcba'):
        loss_fcn = torch.nn.BCEWithLogitsLoss()
    elif args.dataset == 'ogbg-molhiv':
        # Calculate positive weight for binary classification to handle class imbalance
        pos_weight = torch.tensor([2.0]).to(args.device)  # Initial estimate, will be refined during training
        loss_fcn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fcn = torch.nn.CrossEntropyLoss()

    min_loss = 1e10
    patience = 0
    best_val_perf = 0 if args.dataset.startswith('ogbg-mol') else 0  # Higher is better for both metrics

    stats = {}
    train_losses = list()
    val_losses = list()
    train_perfs = list()
    val_perfs = list()
    epoch_times = list()
    forward_times = list()
    learning_rates = list()

    # Training the model
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        loss_all = 0
        forward_time = 0
        
        # For tracking performance
        y_true = []
        y_pred = []
        y_scores = []
        correct = 0

        # Print scaling factors once at the start of training to help understand the loss calculation
        if epoch == 0:
            if args.dataset.startswith('ogbg-mol'):
                print(f"[INFO]: OGB dataset - train loss will be scaled by batch count: 1/{len(train_loader)}")
            else:
                print(f"[INFO]: TU dataset - train loss will be scaled by sample count: 1/{len(train_loader.dataset)}")
                print(f"[INFO]: For reference - batch scaling would be: 1/{len(train_loader)}")
                print(f"[INFO]: Scaling factor difference: {len(train_loader.dataset)/len(train_loader):.2f}x")
                
        for i, data in enumerate(train_loader):
            data = data.to(args.device)
            
            forward_start = time.time()
            out = model(data)
            forward_time += time.time() - forward_start

            # Handle different dataset formats for loss calculation
            if args.dataset.startswith('ogbg-mol'):
                y = data.y.float()
                if args.dataset == 'ogbg-molpcba':
                    # Skip NaN targets for molpcba
                    is_valid = ~torch.isnan(y)
                    loss = loss_fcn(out[is_valid], y[is_valid])
                else:
                    # For ogbg-molhiv, ensure dimensions match
                    # Debug shapes
                    if i == 0 and epoch == 0:
                        print(f"Output shape: {out.shape}, Target shape: {y.shape}")
                    
                    if out.ndim == 1 and y.ndim == 2:
                        y = y.squeeze(1)
                    elif out.ndim == 2 and y.ndim == 1:
                        y = y.unsqueeze(1)
                    
                    # Dynamically update positive weight based on batch statistics
                    if i == 0 and epoch % 5 == 0:
                        batch_pos_weight = ((1 - y).sum() / y.sum()).item()
                        # Clip to reasonable range to prevent instability
                        batch_pos_weight = max(1.0, min(5.0, batch_pos_weight))
                        loss_fcn.pos_weight = torch.tensor([batch_pos_weight]).to(args.device)
                        if args.verbose >= 1:
                            print(f"Updated positive weight to {batch_pos_weight:.2f}")
                    
                    loss = loss_fcn(out, y)
                    
                # Store predictions for metric calculation
                y_true.append(y.detach().cpu())
                y_scores.append(out.detach().cpu())
                if args.dataset == 'ogbg-molhiv':
                    pred = (out > 0).float()
                    y_pred.append(pred.detach().cpu())
            else:
                # Standard classification
                loss = loss_fcn(out, data.y)
                pred = out.max(dim=1)[1]
                correct += pred.eq(data.y).sum().item()

            loss_all += loss.item()
            
            loss.backward()
            
            # Apply gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()

        # Calculate training performance
        if args.dataset == 'ogbg-molhiv':
            y_true = torch.cat(y_true, dim=0).numpy()
            y_scores = torch.cat(y_scores, dim=0).numpy()
            
            if OGB_AVAILABLE:
                # Ensure inputs are 2D arrays as required by OGB evaluator
                if y_true.ndim == 1:
                    y_true = y_true.reshape(-1, 1)
                if y_scores.ndim == 1:
                    y_scores = y_scores.reshape(-1, 1)
                    
                evaluator = Evaluator(name='ogbg-molhiv')
                input_dict = {"y_true": y_true, "y_pred": y_scores}
                result_dict = evaluator.eval(input_dict)
                train_perf = result_dict["rocauc"]
            else:
                train_perf = roc_auc_score(y_true, y_scores)
                
        elif args.dataset == 'ogbg-molpcba':
            y_true = torch.cat(y_true, dim=0).numpy()
            y_scores = torch.cat(y_scores, dim=0).numpy()
            
            if OGB_AVAILABLE:
                # Ensure inputs are 2D arrays (already the case for molpcba, but check to be safe)
                if y_true.ndim == 1:
                    y_true = y_true.reshape(-1, 1)
                if y_scores.ndim == 1:
                    y_scores = y_scores.reshape(-1, 1)
                    
                evaluator = Evaluator(name='ogbg-molpcba')
                input_dict = {"y_true": y_true, "y_pred": y_scores}
                result_dict = evaluator.eval(input_dict)
                train_perf = result_dict["ap"]
            else:
                valid_mask = ~np.isnan(y_true)
                train_perf = average_precision_score(y_true[valid_mask], y_scores[valid_mask], average='micro')
        else:
            # Standard accuracy for TU datasets
            train_perf = correct / len(train_loader.dataset)

        # Evaluate on validation set
        val_perf, val_loss = test(model, val_loader, args)
        
        # Calculate training loss with appropriate scaling
        if args.dataset.startswith('ogbg-mol'):
            # OGB datasets - keep as is 
            train_loss = loss_all / len(train_loader)
        else:
            # TU datasets - scale by dataset size to match validation loss calculation
            train_loss = loss_all / len(train_loader.dataset)
        
        # Step the scheduler based on the appropriate metric
        if args.dataset.startswith('ogbg-mol'):
            scheduler.step(val_perf)
        else:
            scheduler.step(val_loss)
        
        # Store current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        forward_times.append(forward_time)
        
        perf_name = "AP" if args.dataset == 'ogbg-molpcba' else "ROC-AUC" if args.dataset == 'ogbg-molhiv' else "Acc"
        
        print('Epoch: {0} | Train Loss: {1:.4f} | Val Loss: {2:.4f} | Train {3}: {4:.4f} | Val {3}: {5:.4f} | LR: {6:.6f}'.format(
            epoch, train_loss, val_loss, perf_name, train_perf, val_perf, current_lr))
            
        # Print note about scaling on first epoch to explain the change
        if epoch == 0 and not args.dataset.startswith('ogbg-mol'):
            print("[NOTE]: Train and validation losses are now scaled by the same factor (sample count) for direct comparison.")
        
        if args.verbose == 2:
            print('Epoch Time: {:.2f}s | Forward Pass Time: {:.2f}s'.format(epoch_time, forward_time))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_perfs.append(train_perf)
        val_perfs.append(val_perf)

        # For OGB datasets, we want to maximize ROC-AUC or AP
        # For TU datasets, we want to minimize validation loss
        save_model = False
        if args.dataset.startswith('ogbg-mol'):
            if val_perf > best_val_perf:
                best_val_perf = val_perf
                save_model = True
                patience = 0
                # Save the best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_perf': val_perf,
                }, 'best_model.pth')
            else:
                patience += 1
        else:
            if val_loss < min_loss:
                min_loss = val_loss
                save_model = True
                patience = 0
                # Save the best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                }, 'best_model.pth')
            else:
                patience += 1
        
        if save_model:
            print(f'Model saved at epoch {epoch} with validation {perf_name}: {val_perf:.4f}')
        
        if patience > args.patience:
            print(f'Maximum patience reached at epoch {epoch}')
            break

    stats['train_losses'] = train_losses
    stats['val_losses'] = val_losses
    stats['train_perfs'] = train_perfs
    stats['val_perfs'] = val_perfs
    stats['epoch_times'] = epoch_times
    stats['forward_times'] = forward_times
    stats['learning_rates'] = learning_rates
    
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_forward_time = sum(forward_times) / len(forward_times)
    print('\nTiming Statistics:')
    print(f'Average Epoch Time: {avg_epoch_time:.2f}s')
    print(f'Average Forward Pass Time: {avg_forward_time:.2f}s')
    
    # Load the best model for return
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    perf_key = 'val_perf' if args.dataset.startswith('ogbg-mol') else 'val_loss'
    perf_value = checkpoint.get(perf_key, 0.0)
    print(f"\nLoaded best model from epoch {checkpoint['epoch']} with validation {perf_name}: {perf_value:.4f}")
    
    return model.state_dict(), stats

# Hyperparameter tunning based on Optuna
def objective(trial: optuna.Trial, args, train_loader, val_loader):
    # Hyperparametrs to tune
    args.num_hidden = trial.suggest_categorical('num_hidden', [32])
    args.lr = trial.suggest_categorical('lr', [0.0005])
    args.weight_decay = trial.suggest_categorical('weight_decay', [0.0005])
    args.pooling_ratio = trial.suggest_categorical('pooling_ratio', [0.35])
    args.dropout_ratio = trial.suggest_categorical('dropout_ratio', [0.2])
    args.hop_num = trial.suggest_categorical('hop_num', [4])
    args.a = trial.suggest_float('a', -1.0, 2.0, step=0.5)
    args.b = trial.suggest_float('b', -0.5, 2.0, step=0.5)

    # Train the model
    model_state_dict, stats = model_train(args, train_loader, val_loader)

    # Objective is the mean on all val_accs in that study
    score = torch.tensor(stats['val_perfs']).mean().item()
    trial.set_user_attr('model_state_dict', model_state_dict)
    trial.set_user_attr('stats', stats)

    return score

# create and run the optuna study
def run_optimization(args, train_loader, val_loader):
    if args.colab:
        # from google.colab import drive
        # drive.mount('/content/drive')
        #  TODO: Tensor files is not JSON Serializable
        storage_path = 'sqlite:////content/drive/My Drive/optuna_study.db'
    else:
        storage_path = 'optuna_study.db'
    
    model_save_callback = ModelSaveCallback()

    study = optuna.create_study(direction='maximize', study_name='Jacobi 2nd', load_if_exists=True)
    study.optimize(lambda trial: objective(trial, args, train_loader, val_loader), n_trials=100, n_jobs=-1, callbacks=[model_save_callback])

    # Print best hyperparameters and model
    print("Best hyperparameters: ", study.best_params)
    print("Best validation accuracy: ", study.best_value)

    model_state_dict = model_save_callback.best_model_state_dict
    stats = model_save_callback.best_stats

    # Load the best model state_dict into a new model instance
    model = Net(args).to(args.device)
    model.load_state_dict(model_state_dict)

    return model, stats

def main(args):
    logging.basicConfig(level=logging.INFO)
    # device selection
    if args.device == 'cpu':
        torch.manual_seed(args.seed)
    elif args.device == 'cuda':
        if torch.cuda.is_available():
            # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            # os.environ['TORCH_USE_CUDA_DSA'] = '1'
            torch.cuda.manual_seed(args.seed)
            args.device = 'cuda'
        else:
            print('[WARN]: No cuda device available, cpu will be used')
            args.device = 'cpu'
            torch.manual_seed(args.seed)

    print(f'[INFO]: Used device: {args.device}')
    
    # Loading the dataset - first check if it's an OGB dataset
    if args.dataset.startswith('ogbg-mol'):
        if not OGB_AVAILABLE:
            raise ImportError("OGB package not installed. Please install with 'pip install ogb'")
        
        print(f'[INFO]: Loading OGB dataset {args.dataset}')
        from ogb.graphproppred import PygGraphPropPredDataset
        dataset = PygGraphPropPredDataset(name=args.dataset)
        split_idx = dataset.get_idx_split()
        
        # Get dataset information
        if args.dataset == 'ogbg-molhiv':
            args.num_classes = 1  # Binary classification
            args.task_type = 'binary'
        elif args.dataset == 'ogbg-molpcba':
            args.num_classes = dataset.num_tasks  # Multi-label classification
            args.task_type = 'multilabel'
        
        # Get the first graph to check its structure
        first_graph = dataset[0]  # PyG graph object
        
        # Check node features
        if hasattr(first_graph, 'x') and first_graph.x is not None:
            args.num_features = first_graph.x.shape[1]
            print(f'[INFO]: Using PyG node features with {args.num_features} dimensions')
        else:
            args.num_features = 9 
            print(f'[INFO]: Using default node feature dimension of {args.num_features}')
        
        args.num_graphs = len(dataset)
        
        if args.sample_size != -1:
            print("[WARN]: Sampling not supported for OGB datasets. Using full dataset.")
        
        # For OGB datasets, we need to use the built-in collate function
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False)
        
        # For statistics
        training_set = dataset[split_idx["train"]]
        validation_set = dataset[split_idx["valid"]]
        test_set = dataset[split_idx["test"]]
        
    else:
        print('[INFO]: Path:', os.path.join('data',args.dataset))
        if args.dataset == 'FRANKENSTEIN':
            # First try to load the dataset
            dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset)
            
            # Check if dataset has node features by examining the first graph
            first_graph = dataset[0]
            print("[INFO]: Feature tensor shape:", first_graph.x.shape if hasattr(first_graph, 'x') and first_graph.x is not None else "No features")
            
            # Check if features exist and are not empty (zero-dimensional)
            has_valid_features = (hasattr(first_graph, 'x') and 
                                first_graph.x is not None and 
                                first_graph.x.shape[1] > 0)
            
            if not has_valid_features:
                print('[INFO]: FRANKENSTEIN dataset has empty feature tensors. Creating synthetic features...')
                
                # Create synthetic features using our utility function
                from utils import create_synthetic_features
                
                # Use a fixed number of synthetic features for simplicity
                num_synthetic_features = 10
                
                # Create synthetic features
                dataset = create_synthetic_features(dataset, num_features=num_synthetic_features, seed=args.seed)
                
                # Update the number of features
                args.num_features = num_synthetic_features
                print(f'[INFO]: Created {num_synthetic_features} synthetic features for each node')
            else:
                args.num_features = first_graph.x.shape[1]
                print(f'[INFO]: Using existing node features with {args.num_features} dimensions')
        else:
            dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset)
            if hasattr(dataset[0], 'x') and dataset[0].x is not None and dataset[0].x.shape[1] > 0:
                args.num_features = dataset[0].x.shape[1]
            else:
                from utils import create_synthetic_features
                num_synthetic_features = 10
                print(f'[INFO]: Creating {num_synthetic_features} synthetic features for {args.dataset}')
                dataset = create_synthetic_features(dataset, num_features=num_synthetic_features, seed=args.seed)
                args.num_features = num_synthetic_features
                
        args.task_type = 'classification'
        args.num_classes = dataset.num_classes
        args.num_graphs = len(dataset)

        if args.sample_size != -1:
            dataset = sample_dataset(dataset, args.sample_size, args.seed)
            
        # data spliting
        num_training = int(len(dataset)*0.8)
        num_val = int(len(dataset)*0.1)
        num_test = len(dataset) - (num_training+num_val)
        training_set, validation_set, test_set = random_split(dataset,[num_training,num_val,num_test])

        # dataloader
        train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # logger
    metric_name = "Average Precision" if args.dataset == 'ogbg-molpcba' else "ROC-AUC" if args.dataset == 'ogbg-molhiv' else "Accuracy"
    task_type = "Multi-label classification" if args.dataset == 'ogbg-molpcba' else "Binary classification" if args.dataset == 'ogbg-molhiv' else "Classification"
    
    print("""----Data Statistics----
          Dataset: %s
          Task type: %s
          Evaluation metric: %s
          # of graphs: %d
          # of classes/tasks: %d
          # of features: %d
          ---------------------
          Train samples: %d
          Validation samples: %d
          Test samples: %d"""%(
              args.dataset, task_type, metric_name, args.num_graphs, 
              args.num_classes, args.num_features, 
              len(training_set), len(validation_set), len(test_set)))
    
    # Model construction
    if args.test_only:
        print("\n[INFO]: Test only mode - skipping training")
        model = Net(args).to(args.device)
        # Attempt to load the best model
        try:
            print('\nLoading best model for testing...')
            checkpoint = torch.load('best_model.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            perf_key = 'val_perf' if args.dataset.startswith('ogbg-mol') else 'val_loss'
            perf_value = checkpoint.get(perf_key, 0.0)
            print(f"Loaded model from epoch {checkpoint['epoch']} with validation performance: {perf_value:.4f}")
        except Exception as e:
            print(f"[ERROR]: Could not load model: {e}")
            return
    elif args.hyptune == 1:
        model, stats = run_optimization(args, train_loader, val_loader)
    else:
        model_state_dict, stats = model_train(args, train_loader, val_loader)
        model = Net(args).to(args.device)
        model.load_state_dict(model_state_dict)
    
    # Testing the model
    if not args.test_only:
        print('\nLoading best model for testing...')
        checkpoint = torch.load('best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Detailed evaluation on test set
    print("\n-------- Detailed Test Evaluation --------")
    test_perf, test_loss = test(model, test_loader, args)
    print('---------------Test----------------')
    print(f'Test loss: {test_loss:.4f} | Test {metric_name}: {test_perf:.4f}')

    # Also evaluate on validation set for comparison
    print("\n-------- Validation Set Evaluation --------")
    val_perf, val_loss = test(model, val_loader, args)
    print('---------------Validation----------------')
    print(f'Validation loss: {val_loss:.4f} | Validation {metric_name}: {val_perf:.4f}')
    
    # Difference between validation and test performance
    perf_diff = abs(val_perf - test_perf)
    perf_pct = perf_diff / (abs(val_perf) + 1e-8) * 100  # Avoid division by zero
    print(f"\nValidation-Test gap: {perf_diff:.4f} ({perf_pct:.2f}% difference)")

    # Plotting the metrics (only if not in test_only mode)
    if not args.test_only:
        perf_label = "AP" if args.dataset == 'ogbg-molpcba' else "ROC-AUC" if args.dataset == 'ogbg-molhiv' else "Accuracy"
        losses = [stats['train_losses'], stats['val_losses']]
        metrics = [stats['train_perfs'], stats['val_perfs']]
        plotter(losses, metrics, y_label=perf_label)




if __name__ == '__main__':
    main(arg_parse())
