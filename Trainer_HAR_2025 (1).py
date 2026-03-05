import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class HAR_Trainer:
    def __init__(self, model, device, num_classes):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        
        # Multiple loss functions
        self.criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.criterion_focal = FocalLoss()
        self.criterion_contrastive = ContrastiveLoss()
        
    def setup_optimizer(self, lr=0.001):
        """Setup optimizer with different learning rates for different parts"""
        param_groups = [
            {'params': self.model.cnn_3d.parameters(), 'lr': lr},
            {'params': self.model.adaptive_gcn.parameters(), 'lr': lr * 0.1},
            {'params': self.model.bi_lstm.parameters(), 'lr': lr},
            {'params': self.model.stc_attention.parameters(), 'lr': lr},
            {'params': self.model.classifier.parameters(), 'lr': lr * 10}
        ]
        
        self.optimizer = optim.AdamW(param_groups, weight_decay=0.0001)
        
        # Cyclical learning rate
        self.scheduler = CyclicLR(
            self.optimizer, 
            base_lr=lr/10, 
            max_lr=lr, 
            step_size_up=2000,
            cycle_momentum=False
        )
    
    def contrastive_loss(self, features, labels, margin=1.0):
        """Contrastive loss for self-supervised learning"""
        batch_size = features.size(0)
        distances = torch.cdist(features, features, p=2)
        
        # Create label matrix
        label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
        
        positive_loss = (1 - label_matrix.float()) * distances.pow(2)
        negative_loss = label_matrix.float() * F.relu(margin - distances).pow(2)
        
        loss = (positive_loss + negative_loss).mean()
        return loss
    
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, labels) in enumerate(dataloader):
            data, labels = data.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            features = self.model.get_features(data)
            
            # Multiple loss components
            ce_loss = self.criterion_ce(outputs, labels)
            focal_loss = self.criterion_focal(outputs, labels)
            contrastive_loss = self.contrastive_loss(features, labels)
            
            # Combined loss with adaptive weighting
            loss = ce_loss + 0.5 * focal_loss + 0.1 * contrastive_loss

            
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(dataloader)} | '
                      f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%')

        return total_loss / len(dataloader), 100. * correct / total

    def gradient_penalty(self, real_data, labels, lambda_gp=10):
        """Gradient penalty for training stability"""
        batch_size = real_data.size(0)
        
        # Random interpolation between real and fake (for regularization)
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        interpolates = alpha * real_data + (1 - alpha) * (real_data + 0.1 * torch.randn_like(real_data))
        interpolates.requires_grad_(True)
        
        disc_interpolates = self.model(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return lambda_gp * gradient_penalty
    
    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in dataloader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100. * correct / total
        return accuracy, all_predictions, all_labels

class ContrastiveLoss(nn.Module):
    """Contrastive Loss for self-supervised learning"""
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive