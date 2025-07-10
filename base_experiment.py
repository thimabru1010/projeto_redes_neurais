import torch
import os
from sklearn.metrics import accuracy_score, f1_score

class BaseExperiment:
    def __init__(self, model, optimizer, criterion, device, output_dir=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.output_dir = output_dir or "data/experiments/"
        
        # Criar o diretório de output se não existir
        os.makedirs(self.output_dir, exist_ok=True)

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X)[:, 0]
            loss = self.criterion(outputs, y.float())
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * X.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss

    def valid_one_epoch(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)[:, 0]
                loss = self.criterion(outputs, y.float())
                running_loss += loss.item() * X.size(0)
                
                # Aplicar sigmoid e converter para predições binárias
                predictions = torch.sigmoid(outputs).squeeze()
                binary_predictions = (predictions > 0.5).float()
                
                all_predictions.extend(binary_predictions.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        epoch_loss = running_loss / len(dataloader.dataset)
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions)
        
        return epoch_loss, accuracy, f1

    def train(self, train_loader, valid_loader, epochs, scheduler=None, early_stopping_patience=10, early_stopping_min_delta=0.0001):
        history = {'train_loss': [], 'valid_loss': [], 'valid_accuracy': [], 'valid_f1': []}
        
        # Early stopping variables
        best_valid_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader)
            valid_loss, valid_accuracy, valid_f1 = self.valid_one_epoch(valid_loader)
            
            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)
            history['valid_accuracy'].append(valid_accuracy)
            history['valid_f1'].append(valid_f1)
            
            # Update learning rate if scheduler is provided
            if scheduler is not None:
                scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Valid Loss: {valid_loss:.4f} - Valid Acc: {valid_accuracy:.4f} - Valid F1: {valid_f1:.4f} - LR: {current_lr:.6f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Valid Loss: {valid_loss:.4f} - Valid Acc: {valid_accuracy:.4f} - Valid F1: {valid_f1:.4f}")
            
            # Early stopping logic
            if valid_loss < best_valid_loss - early_stopping_min_delta:
                best_valid_loss = valid_loss
                patience_counter = 0
                # Save the best model state
                best_model_state = self.model.state_dict().copy()
                print(f"Validation loss improved. New best: {best_valid_loss:.4f}")
                
            else:
                patience_counter += 1
                print(f"No improvement in validation loss. Patience: {patience_counter}/{early_stopping_patience}")
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    print(f"Best validation loss: {best_valid_loss:.4f}")
                    print('Saving the best model state...')
                    model_path = os.path.join(self.output_dir, f"best_model_epoch_{epoch+1}.pth")
                    self.save_model(model_path, best_model_state)
                    break
        
        return history
    
    def save_model(self, path, model_state=None):
        """
        Salva o modelo treinado no caminho especificado.
        """
        if model_state is not None:
            torch.save(model_state, path)
        else:
            torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")