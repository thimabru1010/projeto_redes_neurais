import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss binária (usa logits, sem sigmoid)

        alpha: peso da classe positiva (0 ≤ alpha ≤ 1). Ex: 0.25 para classe minoritária
        gamma: fator de foco (> 0). Ex: 2.0 para penalizar exemplos fáceis
        reduction: 'mean' | 'sum' | 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: Tensor (N,) ou (N, 1) — saída do modelo sem sigmoid
        targets: Tensor (N,) ou (N, 1) — rótulos 0 ou 1
        """
        targets = targets.view(-1)
        logits = logits.view(-1)

        # aplica sigmoid para obter probabilidade
        probas = torch.sigmoid(logits)

        # focal loss por classe
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        p_t = probas * targets + (1 - probas) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # 'none'
