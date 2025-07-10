import torch
import numpy as np
import torch.nn as nn

##############################
# 1. FUNÇÃO INTEGRATED GRADIENTS
##############################
def integrated_gradients(model: nn.Module,
                         input_tensor: torch.Tensor,
                         baseline: torch.Tensor = None,
                         target_class: int = None,
                         n_steps: int = 50):
    """
    Calcula Integrated Gradients para um único exemplo.
    
    model ........ rede já treinada
    input_tensor . tensor shape (1, n_features)  – exemplo a explicar
    baseline ..... tensor shape (1, n_features)  – referência (default = zeros)
    target_class . int | None  – classe cujo logit será explicado (None = argmax)
    n_steps ....... int  – número de pontos da aproximação de Riemann
    """
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    baseline = baseline.to(input_tensor.device)

    # gera pontos interpolados baseline→input
    alphas = torch.linspace(0.0, 1.0, n_steps + 1, device=input_tensor.device).view(-1, 1)
    interpolated = baseline + alphas * (input_tensor - baseline)
    interpolated.requires_grad_(True)

    # forward em lote
    outputs = model(interpolated)          # (n_steps+1, n_classes ou 1)
    if outputs.dim() == 1:
        outputs = outputs.unsqueeze(1)

    if target_class is None:
        with torch.no_grad():
            target_class = model(input_tensor).argmax(dim=1).item()

    targets = outputs[:, target_class]

    # gradientes
    grads = torch.autograd.grad(
        targets,
        interpolated,
        grad_outputs=torch.ones_like(targets),
        create_graph=False,
        retain_graph=False
    )[0]                                   # (n_steps+1, n_features)

    avg_grads = grads.mean(dim=0)          # (n_features,)
    attributions = (input_tensor - baseline).squeeze(0) * avg_grads
    return attributions.detach()


##############################
# 2. CARREGANDO MODELO E DADOS
##############################

# ajuste os caminhos conforme seus arquivos
model_path = "model_trained.pth"   # modelo salvo com torch.save(model.state_dict(), ...)
X_test_path = "X_test.npy"         # dados de teste em numpy (N, n_features)

# carrega os dados para descobrir n_features
X_test = np.load(X_test_path)
n_features = X_test.shape[1]

# instancia modelo e carrega pesos
model = SimpleMLP(in_features=n_features)
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)

##############################
# 3. CHAMANDO INTEGRATED GRADIENTS
##############################
sample_idx = 0                          # qual linha do X_test queremos explicar
sample_tensor = torch.tensor(X_test[sample_idx:sample_idx+1], dtype=torch.float32)

attr = integrated_gradients(model, sample_tensor, n_steps=100)

print("Atribuições Integrated Gradients:")
print(attr.numpy())

