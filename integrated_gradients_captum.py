from captum.attr import IntegratedGradients
import numpy as np
import pandas as pd
import torch
from model import MLPClassifier
import matplotlib.pyplot as plt
import torch
import matplotlib.ticker as mticker

def save_top_k_features_by_class(model, X_test, y_test, feature_names, top_k=10, device='cpu'):
    """
    Salva gráficos das top-k features mais importantes para cada classe.
    """
    import os
    os.makedirs('data/experiments/exp6', exist_ok=True)
    classes = {0: 'Discharge', 1: 'Death'}
    for class_value, class_name in classes.items():
        idxs = (y_test == class_value).nonzero(as_tuple=True)[0]
        if len(idxs) == 0:
            print(f"Nenhuma amostra para a classe {class_name}")
            continue
        X_class = X_test[idxs]
        mean_attr = mean_absolute_attributions(model, X_class, n_steps=100, device=device)
        # Garante que mean_attr está na CPU
        if isinstance(mean_attr, torch.Tensor):
            mean_attr_np = mean_attr.detach().cpu().numpy()[0]
        else:
            mean_attr_np = mean_attr[0]
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_attr_np
        })
        df_sorted = df.sort_values(by="importance", key=abs, ascending=False).head(top_k)
        df_sorted = df_sorted[::-1]
        plt.figure(figsize=(8, 5))
        plt.barh(df_sorted["feature"], df_sorted["importance"])
        plt.xlabel("Average feature importance (Integrated Gradients)")
        plt.title(f"Top {top_k} most relevant features - Class: {class_name}")
        plt.tight_layout()
        fig_path = f"data/experiments/exp6/top_k_features_{class_name}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Figura salva em: {fig_path}")
        
        # Save top less important features
        df_sorted = df.sort_values(by="importance", key=abs, ascending=False).tail(top_k)
        df_sorted = df_sorted[::-1]
        plt.figure(figsize=(8, 5))
        plt.barh(df_sorted["feature"], df_sorted["importance"])
        plt.xlabel("Average feature importance (Integrated Gradients)")
        plt.title(f"Top {top_k} less relevant features - Class: {class_name}")
        plt.tight_layout()
        fig_path = f"data/experiments/exp6/last_k_features_{class_name}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Figura salva em: {fig_path}")
        
def find_neutral_input(model, X_pool, device='cpu', tol=0.05):
    """
    Procura no conjunto X_pool uma amostra cujo logit esteja mais próximo de 0,
    ou seja, cujo sigmoid esteja mais perto de 0.5 (saída neutra).
    
    Parâmetros:
        model: modelo PyTorch treinado
        X_pool: tensor de entrada (N, D)
        device: 'cpu' ou 'cuda'
        tol: tolerância aceitável no logit (ex: tol=0.05 => aceita logit entre -0.05 e 0.05)

    Retorna:
        A entrada que gera o logit mais próximo de 0
    """
    model.eval()
    X_pool = X_pool.to(device)
    model = model.to(device)

    with torch.no_grad():
        logits = model(X_pool).squeeze()
    
    if logits.ndim == 0:
        logits = logits.unsqueeze(0)

    # acha índice com logit mais próximo de 0
    idx = torch.argmin(torch.abs(logits))
    logit_value = logits[idx].item()

    if abs(logit_value) <= tol:
        print(f"Encontrado: logit = {logit_value:.4f}, sigmoid ≈ {torch.sigmoid(logits[idx]).item():.4f}")
    else:
        print(f"Nenhum logit suficientemente próximo de zero. Logit mais próximo: {logit_value:.4f}")

    return X_pool[idx]


def mean_absolute_attributions(model, X_tensor, n_steps=50, device='cpu'):
    """
    Calcula a média dos valores absolutos das atribuições de Integrated Gradients
    para todos os exemplos do conjunto X_tensor.
    
    Retorna: vetor (n_features,) com a média das atribuições absolutas.
    """
    ig = IntegratedGradients(model)
    all_attr = []
    for i in range(X_tensor.shape[0]):
        x = X_tensor[i:i+1].to(device)
        # baseline deve estar no mesmo device
        baseline = torch.zeros_like(x).to(device)
        attributions, delta = ig.attribute(
            x,
            baselines=baseline,
            target=None,
            return_convergence_delta=True)
        all_attr.append(torch.abs(attributions))
    mean_attr = torch.stack(all_attr).mean(dim=0)
    return mean_attr

def plot_top_k_features(mean_attr, feature_names=None, top_k=10, device='cpu'):
    """
    Plota as top-k features mais importantes com base nas atribuições médias.
    
    mean_attr: tensor (n_features,) com importâncias
    feature_names: lista opcional com nomes das features
    """
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(len(mean_attr))]

    # Garante que mean_attr está na CPU para uso com pandas/numpy/matplotlib
    if isinstance(mean_attr, torch.Tensor):
        mean_attr_np = mean_attr.detach().cpu().numpy()
    else:
        mean_attr_np = mean_attr

    df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_attr_np[0, :]
    })

    df_sorted = df.sort_values(by="importance", key=abs, ascending=False).head(top_k)
    df_sorted = df_sorted[::-1]  # inverte para o plot ficar com a mais importante no topo

    fig = plt.figure(figsize=(8, 5))
    plt.barh(df_sorted["feature"], df_sorted["importance"])
    plt.xlabel("Average feature importance (Integrated Gradients)")
    plt.title(f"Top {top_k} most relevant features")
    plt.tight_layout()
    # plt.show()
    fig.savefig('data/experiments/exp6/top_k_features.png', dpi=300, bbox_inches='tight')
    
    # Plot as less important features
    df_sorted = df.sort_values(by="importance", key=abs, ascending=False).tail(top_k)
    df_sorted = df_sorted[::-1]  # inverte para o plot ficar com a menos importante no topo
    fig = plt.figure(figsize=(8, 5))
    bars = plt.barh(df_sorted["feature"], df_sorted["importance"])
    plt.xlabel("Average feature importance (Integrated Gradients)")
    plt.title(f"Last {top_k} less relevant features")
    
    # Zoom no eixo x para valorizar as pequenas diferenças
    min_val = df_sorted["importance"].min()
    max_val = df_sorted["importance"].max()
    delta = (max_val - min_val) * 0.2 if (max_val - min_val) != 0 else 0.001
    plt.xlim(min_val, max_val + delta)

    # Adiciona os valores numéricos ao lado das barras
    for bar in bars:
        width = bar.get_width()
        plt.text(width + np.sign(width)*delta/3, 
                bar.get_y() + bar.get_height()/2,
                f'  {width:.3f}', va='center', fontsize=10)

    plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    plt.tight_layout()
    # plt.show()
    fig.savefig('data/experiments/exp6/last_k_features.png', dpi=300, bbox_inches='tight')
    
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
    model.load_state_dict(torch.load('data/experiments/exp6/best_model.pth', map_location='cpu'))  # ajuste o caminho do modelo
    model.eval()
    model = model.to(device)
    
    output = model(baseline.to(device))  # teste se o modelo está funcionando
    print(f"Modelo carregado. Exemplo de saída: {output.shape}")
    print(f"Exemplo de saída (logits): {output[0].item():.4f}")
    print(f"Exemplo de saída (sigmoid): {torch.sigmoid(output[0]).item():.4f}")
    print(output.mean())
    print(torch.sigmoid(torch.Tensor([0.0])))
    print(torch.sigmoid(torch.Tensor([0.5])))  # teste de sanidade
    baseline = find_neutral_input(model, X_test, device='cuda').unsqueeze(0)
    print(baseline.shape)
    
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
    # Test Loss: 0.0421 | Test Accuracy: 0.8036 | Test F1: 0.8500
    # Load feature names if available
    df = pd.read_csv('data/baseCovidRJTratado2_preprocessado3.csv')
    df.drop(columns=['Evolução do caso', 'Semana epidemiológica da alta ou óbito'], inplace=True)  # remove index column if exists
    feature_names = df.columns.tolist()
    if 'Evolução do caso' in feature_names:
        feature_names.remove('Evolução do caso')  # remove target column if exists
    
    # cols = [f"feat_{i}" for i in range(X_test.shape[1])]
    attr_series = pd.Series(attributions.detach().cpu().numpy().flatten(), index=feature_names)
    attr_series_sorted = attr_series.sort_values(key=np.abs, ascending=False)

    print("\n=== Integrated Gradients (maiores contribuições) ===")
    print(attr_series_sorted.head(10))

    print(f"\nConvergence delta (deve ficar perto de zero): {delta.item():.4e}")
    
    # calcula atribuições médias
    print("\nCalculando atribuições médias para todas as amostras...")
    mean_attr = mean_absolute_attributions(model, X_test, n_steps=100, device=device) # type: ignore
    
    print(len(feature_names))
    print(mean_attr.shape)
    # exibe gráfico das 10 variáveis mais importantes
    plot_top_k_features(mean_attr, feature_names=feature_names, top_k=10, device=device) # type: ignore
    
    # save_top_k_features_by_class(model, X_test, y_test, feature_names, top_k=10, device=device)