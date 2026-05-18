import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, k, in_feats = 64, feats = 64):
        super().__init__()
        num_groups = next(g for g in [8, 4, 2, 1] if feats % g == 0)
        self.block = nn.Sequential(
            nn.Conv1d(in_feats, feats, k, padding="same"),
            nn.GroupNorm(num_groups, feats), nn.ReLU(),
            nn.Conv1d(feats, feats, k, padding="same"),
            nn.GroupNorm(num_groups, feats), nn.ReLU(),
            nn.MaxPool1d(2),
        )
    def forward(self, x): return self.block(x)

class MultiBranchCNNLSTM(nn.Module):
    def __init__(self, n_channels, n_classes, win_samples, net_cfg):
        super().__init__()

        kernel_list = net_cfg.scales_kernels

        self.n_channels = n_channels

        print(f"N-Branches : {len(kernel_list)}")

        n_branches = len(kernel_list)
        self.branches, self.lstms = nn.ModuleList(), nn.ModuleList()
        for b in range(len(kernel_list)):
            k   = kernel_list[b]
            conv_list = []
            for i in range(net_cfg.depth_per_branch):
                if i == 0:
                    conv_list.append(ConvBlock(k,
                                               in_feats=self.n_channels,
                                               feats=net_cfg.feat_depth
                                            )
                                    )
                else:
                    conv_list.append(ConvBlock(k,
                                               in_feats=net_cfg.feat_depth,
                                               feats=net_cfg.feat_depth
                                            )
                                    )

            cnn = nn.Sequential(*conv_list)
            self.branches.append(cnn)
            flat = cnn(torch.randn(1, self.n_channels, win_samples)).numel()
            self.lstms.append(nn.LSTM(flat, 
                                      net_cfg.lstm_hidden_size, 
                                      batch_first=True)
                                      )

        self.classifier = nn.Sequential(
            nn.Linear(n_branches*net_cfg.lstm_hidden_size, net_cfg.classifier_hidden_dim),
            nn.BatchNorm1d(net_cfg.classifier_hidden_dim), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(net_cfg.classifier_hidden_dim, n_classes),
        )

    def forward(self, x):                       # x: (B,T,C,L)
        B, T, _, L = x.shape
        branch_seq = []
        for cnn, lstm in zip(self.branches, self.lstms):
            feat = cnn(x.reshape(B*T, self.n_channels, L)).reshape(B, T, -1)
            hseq,_ = lstm(feat)                 # (B,T,H)
            branch_seq.append(hseq)
        h = torch.cat(branch_seq, dim=2)        # (B,T,4*H)
        return self.classifier(h.reshape(B*T, -1)).view(B, T, -1)    

    def get_activations(self,x):
        B, T, _, L = x.shape
        branch_seq = []

        for cnn, lstm in zip(self.branches, self.lstms):
            feat = cnn(x.reshape(B*T, self.n_channels, L)).reshape(B, T, -1)
            # hseq,_ = feat                 # (B,T,H)
            branch_seq.append(feat)
            
        h = torch.stack(branch_seq, dim=-1)       # (B,T,4*H)
        return h