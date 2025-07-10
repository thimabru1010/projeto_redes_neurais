import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from sklearn.model_selection import train_test_split
from model import MLPClassifier
from base_experiment import BaseExperiment
import argparse
import os
from sklearn.metrics import accuracy_score, f1_score
def evaluate_on_test_set(model, criterion, device, output_dir, batch_size=256):
    """
    Avalia o modelo no conjunto de teste e imprime loss, accuracy e f1.
    """

    # Carregar dados de teste
    X_test = np.load('data/X_test.npy', allow_pickle=True)
    y_test = np.load('data/y_test.npy', allow_pickle=True)
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Tentar carregar o melhor modelo salvo
    best_model_path = None
    for fname in os.listdir(output_dir):
        if fname.startswith("best_model_epoch_") and fname.endswith(".pth"):
            best_model_path = os.path.join(output_dir, fname)
            break
    if best_model_path is not None:
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Best model not found, evaluating current model.")

    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)[:, 0]
            loss = criterion(outputs, y.float())
            running_loss += loss.item() * X.size(0)
            predictions = torch.sigmoid(outputs).squeeze()
            binary_predictions = (predictions > 0.5).float()
            all_predictions.extend(binary_predictions.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    test_loss = running_loss / len(test_loader.dataset) # type: ignore
    test_acc = accuracy_score(all_targets, all_predictions)
    test_f1 = f1_score(all_targets, all_predictions)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f} | Test F1: {test_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a MLP Classifier on COVID-19 dataset')
    parser.add_argument('--data_path', type=str, default='data/baseCovidRJTratado2_preprocessado.csv',
                        help='Path to the preprocessed dataset CSV file')
    parser.add_argument('--exp_name', type=str, default='exp1',
                        help='Name of the experiment for output folder')
    parser.add_argument('--n_hidden_layers', type=int, default=4,
                        help='List of hidden layer sizes for the MLP')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate for the MLP')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                        help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for training')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='L2 regularization weight (weight_decay) for optimizer')
    parser.add_argument('--layer_norm', type=str, default=None, choices=[None, 'batch', 'layer'],
                        help='Normalization layer to use in the MLP (batch or layer normalization)')
    args = parser.parse_args()
    
    # Hiperparâmetros
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Criar pasta de output para o experimento
    output_dir = f'data/experiments/{args.exp_name}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Experiment output directory: {output_dir}")

    # Carrega o dataset preprocessado
    # df = pd.read_csv('data/baseCovidRJTratado2_preprocessado3.csv')

    # # Separe X e y (ajuste o nome da coluna target conforme seu dataset)
    # target_col = 'Evolução do caso'
    # X = df.drop(columns=[target_col]).values.astype(np.float32)
    # y = df[target_col].values.astype(np.int64)
    X = np.load('data/X_train.npy', allow_pickle=True)
    y = np.load('data/y_train.npy', allow_pickle=True)
    
    input_dim = X.shape[1]  # Atualiza input_dim com base no dataset
    hidden_layers = []
    initial_dim = input_dim
    for _ in range(args.n_hidden_layers):
        layer_size = initial_dim // 2
        hidden_layers.append(layer_size)
        initial_dim = layer_size
    print(f"Input dimension: {input_dim}, Hidden layers: {hidden_layers}")

    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Torch datasets e dataloaders
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Instancia modelo, otimizador e função de perda
    model = MLPClassifier(input_dim=X.shape[1], hidden_layers=hidden_layers, output_dim=1, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss()  # Ajuste para classificação binária
    
    # Scheduler para decaimento da learning rate (CosineAnnealingLR)
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scheduler = ExponentialLR(optimizer, gamma=0.9) # gamma define o fator de decaimento

    # Experimento
    experiment = BaseExperiment(model, optimizer, criterion, device, output_dir)
    history = experiment.train(train_loader, val_loader, args.epochs, scheduler=scheduler)
    
    # Avaliar no conjunto de teste o modelo treinado
    evaluate_on_test_set(model, criterion, device, output_dir, batch_size=args.batch_size)