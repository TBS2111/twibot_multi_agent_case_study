"""GBT-CMA agent: builds the signed weighted graph, then learns graph embeddings with
Cross-Modal Attention -> signed GNN -> Graph Transformer (gated fusion), trained with a
self-supervised signed link objective."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cross_modal_attention import CrossModalAttention
from models.signed_gnn import GNNTransformerEncoder
from preprocessing.features import build_signed_graph


ARCH = "cma-signedgnn-gt-gated"          # bump when the architecture changes (invalidates old checkpoints)


class GBTCMAModel(nn.Module):
    def __init__(self, dims, cfg):
        super().__init__()
        self.cma = CrossModalAttention(dims, cfg.hidden, cfg.heads, cfg.dropout)
        self.enc = GNNTransformerEncoder(cfg.hidden, cfg.emb_dim, cfg.heads,
                                         cfg.gnn_layers, cfg.gt_layers, cfg.dropout)

    def forward(self, mods, ei, ew):
        return self.enc(self.cma(mods), ei, ew)


def signed_link_loss(z, ei, w, margin, n_rand=2):
    z = F.normalize(z, dim=-1)
    c = (z[ei[0]] * z[ei[1]]).sum(-1)
    aw = w.abs()
    pos = w > 0
    l_pos = ((1 - c) * aw)[pos].sum() / (aw[pos].sum() + 1e-8)
    l_neg = (F.relu(c - margin) * aw)[~pos].sum() / (aw[~pos].sum() + 1e-8)
    n = z.size(0)
    r = torch.randint(0, n, (2, n * n_rand), device=z.device)
    l_rand = F.relu((z[r[0]] * z[r[1]]).sum(-1) - margin).mean()
    return l_pos + l_neg + 0.5 * l_rand


class GBTCMAAgent:
    name = "gbt_cma"

    def run(self, state):
        cfg = state["cfg"]
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        edges, w, sims = build_signed_graph(cfg, state["sem_emb"], state["url_mat"], state["interest_mat"],
                                            state["social_mat"], state["link_edges"], state["trust"])
        log = [f"[gbt_cma] signed graph: {len(edges)} edges "
               f"(+{int((w > 0).sum())} / -{int((w <= 0).sum())})"]

        dev = torch.device(cfg.device)
        mods = {"semantic": state["sem_emb"], "behavioral": state["beh_feat"], "trust_profile": state["trust_feat"]}
        mods = {k: torch.tensor(v, dtype=torch.float32, device=dev) for k, v in mods.items()}
        dims = {k: v.size(1) for k, v in mods.items()}
        e = torch.tensor(edges.T, device=dev)
        wt = torch.tensor(w, device=dev)
        ei = torch.cat([e, e.flip(0)], 1)
        ew = torch.cat([wt, wt])

        model = GBTCMAModel(dims, cfg).to(dev)
        ckpt = cfg.ckpt_dir / "gbt_cma.pt"
        loaded = False
        if cfg.mode == "infer" and ckpt.exists():
            sd = torch.load(ckpt, map_location=dev)
            if sd.get("dims") == dims and sd.get("arch") == ARCH:
                model.load_state_dict(sd["model"])
                loaded = True
                log.append("[gbt_cma] loaded checkpoint")
            else:
                log.append("[gbt_cma] checkpoint is from an older architecture -> retraining")
        if not loaded:
            log.append(f"[gbt_cma] CMA -> {cfg.gnn_layers}x signed GNN -> {cfg.gt_layers}x graph transformer "
                       f"({sum(p.numel() for p in model.parameters()):,} params)")
            opt = torch.optim.Adam(model.parameters(), lr=cfg.gbt_lr, weight_decay=cfg.gbt_weight_decay)
            model.train()
            for ep in range(cfg.gbt_epochs):
                opt.zero_grad()
                loss = signed_link_loss(model(mods, ei, ew), e, wt, cfg.neg_margin)
                loss.backward()
                opt.step()
                if ep % max(1, cfg.gbt_epochs // 5) == 0 or ep == cfg.gbt_epochs - 1:
                    log.append(f"[gbt_cma] epoch {ep:4d} loss {loss.item():.4f}")
            torch.save({"model": model.state_dict(), "dims": dims, "arch": ARCH}, ckpt)
        model.eval()
        with torch.no_grad():
            emb = F.normalize(model(mods, ei, ew), dim=-1).cpu().numpy()
        return {"edges": edges, "edge_w": w, "edge_sims": sims, "emb": emb, "log": log}
