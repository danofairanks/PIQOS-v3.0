# tri_brain_robot_skin.py — Tri-Brain + Skin Extension (2025–2026 ready)
# Logic | Empathy | Instinct | Skin
# © 2025 Dano Fairbanks — Restricted Research License

import torch
import torch.nn as nn
import torch.nn.functional as F
from piqos_core import PIQOSCore


class TriBrainRobotSkin(nn.Module):
    def __init__(
        self,
        logic_features: int = 6,      # IMU, joints, etc.
        empathy_features: int = 48,   # face/pose/voice/breath
        instinct_features: int = 3,   # classic deep-brain
        skin_features: int = 24,      # pressure, temp, vibration, chemical, pain, stretch
        neurons: int = 144,
        gating_hidden: int = 96,
        device=None,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.neurons = neurons

        # === Four encoders → PIQOS latent space ===
        self.logic_encoder    = nn.Linear(logic_features,    neurons)
        self.empathy_encoder  = nn.Linear(empathy_features,  neurons)
        self.instinct_encoder = nn.Linear(instinct_features, neurons)
        self.skin_encoder     = nn.Linear(skin_features,     neurons)

        for enc in (self.logic_encoder, self.empathy_encoder,
                    self.instinct_encoder, self.skin_encoder):
            nn.init.xavier_uniform_(enc.weight)

        # === Four PIQOS hemispheres ===
        self.left_brain   = PIQOSCore(neurons=neurons).to(self.device)  # Logic
        self.right_brain  = PIQOSCore(neurons=neurons).to(self.device)  # Empathy
        self.deep_brain   = PIQOSCore(neurons=neurons).to(self.device)  # Instinct
        self.skin_brain   = PIQOSCore(neurons=neurons).to(self.device)  # Skin / Pain

        # === Gating now over 4 brains ===
        summary_dim = 32
        self.logic_summary    = nn.Sequential(nn.Linear(neurons, summary_dim), nn.ReLU())
        self.empathy_summary  = nn.Sequential(nn.Linear(neurons, summary_dim), nn.ReLU())
        self.instinct_summary = nn.Sequential(nn.Linear(neurons, summary_dim), nn.ReLU())
        self.skin_summary     = nn.Sequential(nn.Linear(neurons, summary_dim), nn.ReLU())

        gate_in = summary_dim * 4 + 4  # 4 summaries + 4 H values
        self.gating_net = nn.Sequential(
            nn.Linear(gate_in, gating_hidden),
            nn.ReLU(),
            nn.Linear(gating_hidden, 4),   # logits for [logic, empathy, instinct, skin]
            nn.Softmax(dim=-1)
        )

        # === Corpus callosum now fuses 4 brains ===
        self.corpus_callosum = nn.Linear(4, 1, bias=True)
        nn.init.constant_(self.corpus_callosum.weight, 0.25)
        nn.init.constant_(self.corpus_callosum.bias, 0.0)

        # === Policy head — now sees all 4 hebbian shadows (576 dims) ===
        policy_hidden = 256
        self.policy_proj = nn.Linear(4 * neurons, policy_hidden)
        self.policy_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(policy_hidden, policy_hidden // 2),
            nn.ReLU(),
            nn.Linear(policy_hidden // 2, 8)   # 8-d action (extra dims for haptic feedback, grasp force, etc.)
        )

        # move everything
        for mod in (self.logic_encoder, self.empathy_encoder, self.instinct_encoder, self.skin_encoder,
                    self.logic_summary, self.empathy_summary, self.instinct_summary, self.skin_summary,
                    self.gating_net, self.corpus_callosum, self.policy_proj, self.policy_head):
            mod.to(self.device)

        self.eval()

    def forward(self, logic_inputs, empathy_inputs, instinct_inputs, skin_inputs, return_policy=False):
        # Encode
        L = F.normalize(self.logic_encoder(logic_inputs.to(self.device)),    p=2, dim=1)
        R = F.normalize(self.empathy_encoder(empathy_inputs.to(self.device)),p=2, dim=1)
        D = F.normalize(self.instinct_encoder(instinct_inputs.to(self.device)),p=2, dim=1)
        S = F.normalize(self.skin_encoder(skin_inputs.to(self.device)),    p=2, dim=1)

        # Four hemispheres
        H_l, vl, al = self.left_brain(L)
        H_r, vr, ar = self.right_brain(R)
        H_d, vd, ad = self.deep_brain(D)
        H_s, vs, as_ = self.skin_brain(S)

        H_l = H_l.view(-1); H_r = H_r.view(-1); H_d = H_d.view(-1); H_s = H_s.view(-1)

        # Gating over 4 brains
        sL = self.logic_summary(L); sR = self.empathy_summary(R)
        sD = self.instinct_summary(D); sS = self.skin_summary(S)

        gating_input = torch.cat([
            sL, sR, sD, sS,
            H_l.unsqueeze(-1), H_r.unsqueeze(-1), H_d.unsqueeze(-1), H_s.unsqueeze(-1)
        ], dim=1)

        weights = self.gating_net(gating_input)          # (batch,4)
        H_stack = torch.stack([H_l, H_r, H_d, H_s], dim=1)
        H_gated = (weights * H_stack).sum(dim=1, keepdim=True)
        H_global = self.corpus_callosum(H_gated).view(-1)

        # Optional policy (reads all four hebbian shadows)
        policy_action = None
        if return_policy:
            hb_all = torch.cat([
                self.left_brain.hebbian.detach(),
                self.right_brain.hebbian.detach(),
                self.deep_brain.hebbian.detach(),
                self.skin_brain.hebbian.detach()
            ], dim=0).unsqueeze(0).repeat(logic_inputs.shape[0], 1).to(self.device)

            x = F.relu(self.policy_proj(hb_all))
            policy_action = torch.tanh(self.policy_head(x))   # bounded -1..1

        details = {
            "H_global": H_global,
            "H_left": H_l, "H_right": H_r, "H_deep": H_d, "H_skin": H_s,
            "gating_weights": weights,
        }

        if return_policy:
            return (H_global, details, policy_action)
        return (H_global, details)

    def suggest_action(self, details):
        gw = details["gating_weights"].mean(dim=0).cpu().numpy()
        idx = gw.argmax()
        labels = ["Logic — execute plan", "Empathy — protect human", "Instinct — avoid danger", "Skin — gentle touch / recoil"]
        return labels[idx]