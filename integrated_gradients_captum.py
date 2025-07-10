from captum.attr import IntegratedGradients
import numpy as np
import pandas as pd
import torch
from model import MLPClassifier
import matplotlib.pyplot as plt

def mean_absolute_attributions(model, X_tensor, n_steps=50):
    """
    Calcula a média dos valores absolutos das atribuições de Integrated Gradients
    para todos os exemplos do conjunto X_tensor.
    
    Retorna: vetor (n_features,) com a média das atribuições absolutas.
    """
    ig = IntegratedGradients(model)
    all_attr = []
    for i in range(X_tensor.shape[0]):
        x = X_tensor[i:i+1]
        attributions, delta = ig.attribute(
            input_example,
            baselines=baseline,
            target=None,
            return_convergence_delta=True)
        all_attr.append(torch.abs(attributions))
    mean_attr = torch.stack(all_attr).mean(dim=0)
    return mean_attr

def plot_top_k_features(mean_attr, feature_names=None, top_k=10):
    """
    Plota as top-k features mais importantes com base nas atribuições médias.
    
    mean_attr: tensor (n_features,) com importâncias
    feature_names: lista opcional com nomes das features
    """
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(len(mean_attr))]

    df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_attr.cpu().numpy()
    })

    df_sorted = df.sort_values(by="importance", key=abs, ascending=False).head(top_k)
    df_sorted = df_sorted[::-1]  # inverte para o plot ficar com a mais importante no topo

    fig = plt.figure(figsize=(8, 5))
    plt.barh(df_sorted["feature"], df_sorted["importance"])
    plt.xlabel("Importância média (|Integrated Gradients|)")
    plt.title(f"Top {top_k} features mais relevantes")
    plt.tight_layout()
    plt.show()
    fig.savefig('data/experiments/exp3/top_k_features.png', dpi=300, bbox_inches='tight')
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # baseline - vetor de zeros (mesma dimensionalidade)
    X_test = torch.Tensor(np.load('data/X_test.npy', allow_pickle=True)).to(device)
    y_test = torch.Tensor(np.load('data/y_test.npy', allow_pickle=True)).to(device)
    baseline = torch.zeros_like(X_test[0]).unsqueeze(0)

    # -------------------
    # 1. CARREGANDO O MODELO
    # -------------------
    input_dim = X_test.shape[1]
    n_hidden_layers = 4  # ajuste conforme seu modelo
    hidden_layers = []
    initial_dim = input_dim
    for i in range(n_hidden_layers):
        layer_size = initial_dim // 2
        hidden_layers.append(layer_size)
        initial_dim = layer_size
    model = MLPClassifier(input_dim=input_dim, hidden_layers=hidden_layers, output_dim=1)  # ajuste conforme seu modelo
    model.load_state_dict(torch.load('data/experiments/exp3/best_model_epoch_36.pth', map_location='cpu'))  # ajuste o caminho do modelo
    model.eval()
    model = model.to(device)
    
    ig = IntegratedGradients(model)

    # Escolha um registro para explicar
    idx = 0
    input_example = X_test[idx].unsqueeze(0)          # shape (1, n_features)

    # target=None usa logit da cabeça única (classe positiva)
    attributions, delta = ig.attribute(
            input_example,
            baselines=baseline,
            target=None,
            return_convergence_delta=True
    )

    # --------------
    # 5. EXIBINDO A EXPLICAÇÃO
    # --------------
    # Load feature names if available
    df = pd.read_csv('data/baseCovidRJTratado2_preprocessado3.csv')
    feature_names = df.columns.tolist()
    if 'Evolução do caso' in feature_names:
        feature_names.remove('Evolução do caso')  # remove target column if exists
    
    # cols = [f"feat_{i}" for i in range(X_test.shape[1])]
    attr_series = pd.Series(attributions.detach().numpy().flatten(), index=feature_names)
    attr_series_sorted = attr_series.sort_values(key=np.abs, ascending=False)

    print("\n=== Integrated Gradients (maiores contribuições) ===")
    print(attr_series_sorted.head(10))

    print(f"\nConvergence delta (deve ficar perto de zero): {delta.item():.4e}")
    
    # calcula atribuições médias
    print("\nCalculando atribuições médias para todas as amostras...")
    mean_attr = mean_absolute_attributions(model, X_test, n_steps=100)
    
    # exibe gráfico das 10 variáveis mais importantes
    plot_top_k_features(mean_attr, feature_names=feature_names, top_k=10)