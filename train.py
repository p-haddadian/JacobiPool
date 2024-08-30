import argparse
import logging
import os
import optuna

import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric import utils
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

from networks import Net
from utils import EarlyStopping, ModelSaveCallback
from utils import plotter, sample_dataset


def arg_parse(args = None):
    parser = argparse.ArgumentParser(description='JacobiPool')
    parser.add_argument('--dataset', type=str, default='DD', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
    parser.add_argument('--epochs', type=int, default=1, help='maximum number of epochs')
    parser.add_argument('--seed', type=int, default=777, help='seed')
    parser.add_argument('--device', type=str, default='cuda', help='device selection: cuda or cpu')
    parser.add_argument('--batch_size', type=int, default=4 , help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--approx_func', type=str, default='jacobi', help='desired approximation function (e.g. jacobi, chebyshev)')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--num_hidden', type=int, default=32, help='hidden size')
    parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--num_heads', type=int, default=2, help="number of hidden attention heads")
    parser.add_argument("--hop_num", type=int, default=2, help="hop number")
    parser.add_argument("--p_norm", type=int, default=0.0, help="p_norm")
    parser.add_argument('--early_stop', default=False, help="indicates whether to use early stop or not")
    parser.add_argument('--patience', type=int, default=50, help='patience for earlystopping')
    parser.add_argument('--verbose', type=int, default=1, help='level of verbosity: 0: Just the important outputs, 1: Partial verbosity including model training per epoch, 2: Complete verbosity and printing all INFOs')
    parser.add_argument('--hyptune', type=bool, default=True, help='whether you want Optuna find the best hyperparameters')
    parser.add_argument('--sample_size', type=int, default=-1, help='if want to train on a subset of dataset, specify the number of samples')
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
        - accuracy, loss
    '''
    model.eval()
    correct = 0
    loss = 0
    loss_fcn = torch.nn.CrossEntropyLoss()
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim = 1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += loss_fcn(out, data.y).item()
    return correct / len(loader.dataset), loss / len(loader.dataset)

def model_train(args, train_loader, val_loader):
    model = Net(args).to(args.device)
    print(f'[INFO]: Model architecture:\n{model}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fcn = torch.nn.CrossEntropyLoss()

    min_loss = 1e10
    patience = 0

    stats = {}
    train_losses = list()
    val_losses = list()

    train_accs = list()
    val_accs = list()

    # Training the model
    for epoch in range(args.epochs):
        model.train()
        correct = 0
        loss_all =0
        for i, data in enumerate(train_loader):
            data = data.to(args.device)
            out = model(data)
            loss = loss_fcn(out, data.y)
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
            loss_all += loss.item()
            train_acc = correct / len(train_loader.dataset)
            if args.verbose == 2:
                print('Training Loss: {0:.4f}| Training Acc: {1:.4f}'.format(loss, train_acc))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        val_acc, val_loss = test(model, val_loader, args)
        train_loss = loss_all / len(train_loader.dataset)
        print('Epoch: {0} | Train Loss: {1:.4f} | Val Loss: {2:.4f} | Val Acc: {3:.4f}'.format(epoch, train_loss, val_loss, val_acc))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_loss < min_loss:
            torch.save(model.state_dict(), 'latest.pth')
            print('Model saved at epoch {}'.format(epoch))
            min_loss = val_loss
            patience = 0
        else:
            patience += 1
        
        if patience > args.patience:
            print('Maximum patience reached at epoch {} and val loss had no change'.format(epoch))
            break
    stats['train_losses'] = train_losses
    stats['val_losses'] = val_losses
    stats['train_accs'] = train_accs
    stats['val_accs'] = val_accs
    return model.state_dict(), stats

# Hyperparameter tunning based on Optuna
def objective(trial: optuna.Trial, args, train_loader, val_loader):
    # Hyperparametrs to tune
    args.num_hidden = trial.suggest_categorical('num_hidden', [32, 64])
    args.lr = trial.suggest_categorical('lr', [0.005, 0.001, 0.0005])
    args.weight_decay = trial.suggest_categorical('weight_decay', [0.0001, 0.00005])
    args.pooling_ratio = trial.suggest_categorical('pooling_ratio', [0.25, 0.35, 0.5])
    args.dropout_ratio = trial.suggest_categorical('dropout_ratio', [0.2])
    args.hop_num = trial.suggest_categorical('hop_num', [2, 3, 4])
    args.a = trial.suggest_float('a', -1.0, 2.0, step=0.5)
    args.b = trial.suggest_float('b', -1.0, 2.0, step=0.5)

    # Train the model
    model_state_dict, stats = model_train(args, train_loader, val_loader)

    score = torch.tensor(stats['val_accs']).mean().item()
    trial.set_user_attr('model_state_dict', model_state_dict)
    trial.set_user_attr('stats', stats)

    return score

# create and run the optuna study
def run_optimization(args, train_loader, val_loader):
    model_save_callback = ModelSaveCallback()

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, args, train_loader, val_loader), n_trials=10, callbacks=[model_save_callback])

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
            args.device = 'cuda:0'
        else:
            print('[WARN]: No cuda device available, cpu will be used')
            args.device = 'cpu'
            torch.manual_seed(args.seed)

    print(f'[INFO]: Used device: {args.device}')
    
    # loading the dataset
    print('[INFO]: Path:', os.path.join('data',args.dataset))
    dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset)

    # in case of sampling
    if args.sample_size != -1:
        dataset = sample_dataset(dataset, args.sample_size, args.seed)
        
    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features
    args.num_graphs = len(dataset)

    # data spliting
    num_training = int(len(dataset)*0.8)
    num_val = int(len(dataset)*0.1)
    num_test = len(dataset) - (num_training+num_val)
    training_set, validation_set, test_set = random_split(dataset,[num_training,num_val,num_test])

    # dataloader
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # logger
    print("""----Data Statistics----
          Dataset: %s
          # of graphs: %d
          # of classes: %d
          # of features: %d
          ---------------------
          Train samples: %d
          Validation samples: %d
          Test samples: %d"""%(args.dataset, args.num_graphs, args.num_classes, args.num_features, len(training_set), len(validation_set), len(test_set)))
    
    # Model construction
    if args.hyptune:
        model, stats = run_optimization(args, train_loader, val_loader)
    else:
        model_state_dict, stats = model_train(args, train_loader, val_loader)
        model = Net(args).to(args.device)
        model.load_state_dict(model_state_dict)
    
    # Testing the model
    model.load_state_dict(torch.load('latest.pth'))
    test_acc, test_loss = test(model, test_loader, args)
    print('---------------Test----------------')
    print('Test loss: {0:.4f} | Test Acc: {1:.4f}'.format(test_loss, test_acc))

    # Plotting the necessary metrics
    losses = [stats['train_losses'], stats['val_losses']]
    accs = [stats['train_accs'], stats['val_accs']]
    plotter(losses, accs)




if __name__ == '__main__':
    main(arg_parse())
