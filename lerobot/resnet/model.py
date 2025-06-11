import torch
import torch.nn as nn
import torchvision.models as models

class SkillPlanner(nn.Module):
    def __init__(self, num_skills, max_len=5, embed_dim=512, num_layers=2):
        super().__init__()
        self.max_len = max_len
        self.num_skills = num_skills

        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # output: [B, 512, 1, 1]
        self.fc_embed = nn.Linear(512 * 2, embed_dim)  # head + flan feature → embedding

        self.skill_embedding = nn.Embedding(num_skills + 1, embed_dim)  # +1 for padding (-1)
        self.pos_embedding = nn.Parameter(torch.randn(max_len, embed_dim))

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(embed_dim, num_skills)

    def forward(self, head, flan, labels=None, mask=None):
        # Encode images
        B = head.size(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feat_head = self.encoder(head).squeeze(-1).squeeze(-1)  # [B, 512]
        feat_flan = self.encoder(flan).squeeze(-1).squeeze(-1)  # [B, 512]
        img_feat = torch.cat([feat_head, feat_flan], dim=1)     # [B, 1024]
        context = self.fc_embed(img_feat).unsqueeze(1)          # [B, 1, D]

        if labels is not None:
            labels_input = labels.clone()
            labels_input[labels_input == -1] = self.num_skills  # padding idx
            tgt_embed = self.skill_embedding(labels_input) + self.pos_embedding  # [B, T, D]
            T = self.max_len
            tgt_mask = torch.triu(torch.ones((T, T), device=device), diagonal=1).bool()  # [T, T]
            # tgt_mask = (labels != -1).unsqueeze(1).repeat(1, self.max_len, 1)  # [B, T, T]
            out = self.decoder(tgt_embed, context, tgt_mask)
            logits = self.output_layer(out)  # [B, T, num_skills]
            return logits
        else:
            # Inference: autoregressive
            outputs = []
            prev = torch.full((head.size(0), 1), self.num_skills, dtype=torch.long, device=head.device)  # Start token
            for t in range(self.max_len):
                embed = self.skill_embedding(prev) + self.pos_embedding[:prev.size(1)]
                out = self.decoder(embed, context)
                logits = self.output_layer(out[:, -1])  # [B, num_skills]
                pred = logits.argmax(dim=-1, keepdim=True)  # [B, 1]
                outputs.append(pred)
                prev = torch.cat([prev, pred], dim=1)
            return torch.cat(outputs, dim=1)  # [B, T]
