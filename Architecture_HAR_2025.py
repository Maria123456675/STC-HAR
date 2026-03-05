import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SpatialTemporalChannelAttention(nn.Module):
    """Spatial-Temporal-Channel Attention Module"""
    def __init__(self, in_channels, num_joints, num_frames, reduction_ratio=16):
        super(SpatialTemporalChannelAttention, self).__init__()

        # Spatial Attention
        self.spatial_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, 1),
            nn.Sigmoid()
        )

        # Temporal Attention
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction_ratio,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction_ratio, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Channel Attention
        self.channel_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch, channels, frames, joints)
        batch_size, C, T, V = x.shape

        # Spatial Attention
        x_s = x.permute(0, 2, 3, 1)
        s_w = self.spatial_fc(x_s)
        x = x * s_w.permute(0, 3, 1, 2)

        # Temporal Attention
        t_w = self.temporal_conv(x.mean(-1))
        x = x * t_w.unsqueeze(-1)

        # Channel Attention
        c_w = self.channel_fc(x.mean(-1).mean(-1))
        x = x * c_w.unsqueeze(-1).unsqueeze(-1)

        return x

class AdaptiveGCNLayer(nn.Module):
    """Adaptive Graph Convolutional Layer"""
    def __init__(self, in_channels, out_channels, num_joints):
        super(AdaptiveGCNLayer, self).__init__()
        self.num_joints = num_joints
        
        # Learnable adjacency matrix
        self.base_adj = nn.Parameter(torch.eye(num_joints))
        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        
        # Adaptive weights
        self.adaptive_weights = nn.Sequential(
            nn.Linear(in_channels, num_joints * num_joints),
            nn.Tanh()
        )
        
    def forward(self, x, edge_index):
        # x shape: (batch * frames, joints, channels)
        BT, V, C = x.shape

      
        # Apply graph convolution
        delta = self.adaptive_weights(x.mean(1)).view(BT, V, V)
        adj = torch.softmax(delta, -1) + self.base_adj.unsqueeze(0)
        adj = adj / adj.sum(-1, keepdim=True).clamp(1e-6)
        x_out = torch.bmm(adj, self.W(x))
        BT2, V2, Co = x_out.shape
        x_out = self.bn(x_out.reshape(BT2 * V2, Co)).reshape(BT2, V2, Co)
        return F.relu(x_out)

class MultiHeadGraphAttention(nn.Module):
    """Multi-Head Graph Attention"""
    def __init__(self, in_channels, out_channels, num_heads=8):
        super(MultiHeadGraphAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        
        self.query = nn.Linear(in_channels, out_channels)
        self.key = nn.Linear(in_channels, out_channels)
        self.value = nn.Linear(in_channels, out_channels)
        self.output = nn.Linear(out_channels, out_channels)
        
    def forward(self, x, adjacency=None):
        # x shape: (batch, joints, channels)
        batch_size, V, C = x.shape
        
        Q = self.query(x).view(batch_size, V, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, V, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, V, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = torch.einsum('bqhd,bkhd->bhqk', Q, K) / math.sqrt(self.head_dim)
        
        # Apply adjacency mask if provided
        if adjacency is not None:
            scores = scores.masked_fill(adjacency.unsqueeze(0).unsqueeze(0) == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V)
        out = out.contiguous().view(batch_size, V, -1)
        
        return self.output(out)

class Skeleton3DCNN(nn.Module):
    """3D CNN for Spatio-temporal Feature Extraction"""
    def __init__(self, in_channels=3, feature_dim=64):
        super(Skeleton3DCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First 3D conv block
            nn.Conv3d(in_channels, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            # Second 3D conv block
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            # Third 3D conv block
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 1, 1))
        )
        
        self.fc = nn.Linear(256, feature_dim)
        
    def forward(self, x):
        # x shape: (batch, channels, frames, joints, 1)
        x = self.conv_layers(x)
        x = x.squeeze(-1).squeeze(-1)  # (batch, 256, frames)
        x = x.transpose(1, 2)  # (batch, frames, 256)
        x = self.fc(x)  # (batch, frames, feature_dim)
        return x

class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM for Temporal Modeling"""
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
        super(BidirectionalLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
    def forward(self, x):
        # x shape: (batch, frames, features)
        lstm_out, (hidden, cell) = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        
        # Combine bidirectional outputs
        forward_out = hidden[-2, :, :]  # Last forward layer
        backward_out = hidden[-1, :, :]  # Last backward layer
        combined = torch.cat([forward_out, backward_out], dim=1)
        
        return lstm_out, combined

class STCHAR(nn.Module):
    """Extended Multi-Stream Temporal-attention Adaptive GCN"""
    def __init__(self, num_classes, num_joints=25, num_frames=300, 
                 in_channels=3, hidden_dim=256, num_heads=8):
        super(STCHAR, self).__init__()
        
        self.num_joints = num_joints
        self.num_frames = num_frames
        
        # 3D-CNN Stream
        self.cnn_3d = Skeleton3DCNN(in_channels, hidden_dim)
        
        # Adaptive GCN Stream
        self.adaptive_gcn = AdaptiveGCNLayer(hidden_dim, hidden_dim, num_joints)
        self.multi_head_attention = MultiHeadGraphAttention(hidden_dim, hidden_dim, num_heads)
        
        # Attention Mechanisms
        self.stc_attention = SpatialTemporalChannelAttention(hidden_dim, num_joints, num_frames)
        
        # Bi-LSTM for temporal modeling
        self.bi_lstm = BidirectionalLSTM(hidden_dim, hidden_dim // 2)
        
        # Multi-stream fusion
        self.fusion_weights = nn.Parameter(torch.ones(3))  # CNN, GCN, LSTM weights
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize adjacency matrix (body connections)
        self.edge_index = self._create_body_edges(num_joints)
        
    def _create_body_edges(self, num_joints):
        """Create basic body skeleton edges"""
        # NTU-RGBD 25 joints connections
        edges = [
            (0,1), (1,2), (2,3), (3,4),              # Spine
            (1,5), (5,6), (6,7), (7,8),              # Left arm
            (1,9), (9,10), (10,11), (11,12),         # Right arm
            (0,13), (13,14), (14,15), (15,16),       # Left leg
            (0,17), (17,18), (18,19), (19,20),       # Right leg
            (2,21), (5,21), (9,21), (13,21), (17,21) # Additional connections
        ]
        
        edge_index = []
        for edge in edges:
            edge_index.append([edge[0], edge[1]])
            edge_index.append([edge[1], edge[0]])  # Undirected graph
            
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    def get_features(self, x):
        B, C, T, V = x.shape
        cnn_feat = self.cnn_3d(x.unsqueeze(-1))
        attn_in = cnn_feat.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, V)
        attn_feat = self.stc_attention(attn_in)
        gcn_in = attn_feat.permute(0, 2, 3, 1).contiguous().view(B * T, V, -1)
        gcn_out = self.adaptive_gcn(gcn_in).view(B, T, V, -1).mean(1)
        gcn_pooled = self.multi_head_attention(gcn_out).mean(1)
        _, lstm_f = self.bi_lstm(attn_feat.mean(-1).transpose(1, 2))
        cnn_pooled = cnn_feat.mean(1)
        w = F.softmax(self.fusion_weights, 0)
        return torch.cat([w[0] * cnn_pooled, w[1] * gcn_pooled, w[2] * lstm_f], 1)

    def forward(self, x):
        # x shape: (batch, channels, frames, joints)
        batch_size, C, T, V = x.shape
        
        # 3D-CNN Feature Extraction
        cnn_input = x.unsqueeze(-1)  # Add depth dimension
        cnn_features = self.cnn_3d(cnn_input)  # (batch, frames, hidden_dim)
        
        # Prepare for GCN
        gcn_input = cnn_features.transpose(1, 2)  # (batch, hidden_dim, frames)
        gcn_input = gcn_input.unsqueeze(-1)  # (batch, hidden_dim, frames, 1)
        gcn_input = gcn_input.expand(-1, -1, -1, V)  # (batch, hidden_dim, frames, joints)
        
        # Apply STC Attention
        attended_features = self.stc_attention(gcn_input)
        
        # Adaptive GCN Processing
        gcn_features = attended_features.permute(0, 2, 3, 1)  # (batch, frames, joints, hidden_dim)
        gcn_features = gcn_features.contiguous().view(batch_size * T, V, -1)
        
        # Apply adaptive GCN
        gcn_out = self.adaptive_gcn(gcn_features, self.edge_index)
        gcn_out = gcn_out.view(batch_size, T, V, -1)
        
        # Multi-head attention on joints
        gcn_out = gcn_out.mean(1)  # (batch, joints, hidden_dim)
        gcn_attended = self.multi_head_attention(gcn_out)
        gcn_pooled = gcn_attended.mean(1)  # (batch, hidden_dim)
        
        # Temporal modeling with Bi-LSTM
        temporal_features = attended_features.mean(-1).transpose(1, 2)  # (batch, frames, hidden_dim)
        lstm_features, lstm_final = self.bi_lstm(temporal_features)
        
        # CNN stream features
        cnn_pooled = cnn_features.mean(1)  # (batch, hidden_dim)
        
        # Multi-stream fusion with learned weights
        weights = F.softmax(self.fusion_weights, dim=0)
        fused_features = torch.cat([
            weights[0] * cnn_pooled,
            weights[1] * gcn_pooled, 
            weights[2] * lstm_final
        ], dim=1)
        
        # Classification
        output = self.classifier(fused_features)
        
        return output