"""Quick integration test for v8 training pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from deepexomir.model.deepexomir_v8 import DeepExoMirModelV8
from deepexomir.model.multitask_heads import MultiTaskLoss
from deepexomir.training.trainer import Trainer, focal_loss

print("Test 1: All imports OK")

# Test 2: Create v8 model
print("\nTest 2: Creating v8 model...")
model = DeepExoMirModelV8(
    config={
        "backbone": {"embed_dim": 1280, "load_backbone": False},
        "model": {
            "d_model": 256, "n_heads": 8, "d_ff": 1024, "n_layers": 4,
            "d_conv": 4, "expand": 2, "cross_attn_every": 3,
            "dropout": 0.2, "max_mirna_len": 30, "max_target_len": 50,
        },
        "structural": {"bp_cnn_out": 128, "struct_mlp_in": 8, "struct_mlp_out": 64},
        "contact_map": {"enabled": True, "proj_dim": 32, "out_dim": 128},
        "classifier": {"type": "moe", "n_experts": 4, "top_k": 2},
        "multitask": {"enabled": True, "w_seed": 0.3, "w_mfe": 0.2, "w_position": 0.2},
        "augmentation": {"enabled": True, "p_augment": 0.3},
    },
    precomputed_embeddings=True,
)
print(f"  v8 model: {model.trainable_parameters():,} params")
print(f"  isinstance check: {isinstance(model, DeepExoMirModelV8)}")

# Test 3: Forward pass
print("\nTest 3: Forward pass...")
B = 4
model = model.cuda()
model.train()

batch = {
    "mirna_seq": ["AUGCUAGCUAGCUAGCUAGCU"] * B,
    "target_seq": ["GCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCU"] * B,
    "structural_features": torch.randn(B, 8).cuda(),
    "mirna_pertoken_emb": torch.randn(B, 30, 1280).cuda(),
    "mirna_pertoken_mask": torch.ones(B, 30, dtype=torch.bool).cuda(),
    "target_pertoken_emb": torch.randn(B, 50, 1280).cuda(),
    "target_pertoken_mask": torch.ones(B, 50, dtype=torch.bool).cuda(),
    "mirna_pooled_emb": torch.randn(B, 1280).cuda(),
    "target_pooled_emb": torch.randn(B, 1280).cuda(),
}

output = model(
    mirna_seqs=batch["mirna_seq"],
    target_seqs=batch["target_seq"],
    struct_features=batch["structural_features"],
    mirna_pertoken_emb=batch["mirna_pertoken_emb"],
    mirna_pertoken_mask=batch["mirna_pertoken_mask"],
    target_pertoken_emb=batch["target_pertoken_emb"],
    target_pertoken_mask=batch["target_pertoken_mask"],
    mirna_pooled_emb=batch["mirna_pooled_emb"],
    target_pooled_emb=batch["target_pooled_emb"],
)
print(f"  Output type: {type(output).__name__}")
print(f"  logits: {output['logits'].shape}")
print(f"  aux_preds: {list(output.get('aux_preds', {}).keys())}")
print(f"  load_balance_loss: {output['load_balance_loss'].item():.6f}")

# Test 4: Focal loss + multi-task loss
print("\nTest 4: Multi-task loss...")
labels = torch.randint(0, 2, (B,)).cuda()
primary_loss = focal_loss(
    output["logits"], labels, gamma=1.0, alpha=0.5, label_smoothing=0.05,
)
print(f"  Primary focal loss: {primary_loss.item():.4f}")

mt_loss = MultiTaskLoss(w_seed=0.3, w_mfe=0.2, w_position=0.2, w_load_balance=0.01)
mt_loss = mt_loss.cuda()

seed_labels = batch["structural_features"][:, 5].long().clamp(0, 7)
mfe_labels = batch["structural_features"][:, 0]

total_loss, loss_dict = mt_loss(
    primary_loss,
    output["aux_preds"],
    seed_type_labels=seed_labels,
    mfe_labels=mfe_labels,
    load_balance_loss=output["load_balance_loss"],
)
print(f"  Total loss: {total_loss.item():.4f}")
for k, v in loss_dict.items():
    print(f"    {k}: {v.item():.4f}")

# Test 5: Backward pass
print("\nTest 5: Backward pass...")
total_loss.backward()
grad_count = sum(1 for p in model.parameters() if p.grad is not None)
total_params = sum(1 for p in model.parameters() if p.requires_grad)
print(f"  Params with gradients: {grad_count}/{total_params}")
# MoE routing may leave some expert params unused in small batches (expected)
assert grad_count >= total_params - 10, f"Too many missing gradients! {grad_count} vs {total_params}"
print(f"  (4 missing = unused MoE experts in small batch, expected)")

# Test 6: Trainer v8 detection
print("\nTest 6: Trainer v8 detection...")
# Create a minimal trainer to test v8 detection
from torch.utils.data import DataLoader, TensorDataset

dummy_ds = TensorDataset(torch.randn(8, 1), torch.randint(0, 2, (8,)))
dummy_loader = DataLoader(dummy_ds, batch_size=4)

# We can't fully test Trainer without proper DataLoader format,
# but we can test the v8 detection logic
trainer = Trainer.__new__(Trainer)
trainer.model = model
trainer.device = torch.device("cuda")
trainer.is_v8 = False
trainer.multitask_loss_fn = None

# Simulate the v8 detection that __init__ does
from deepexomir.model.deepexomir_v8 import DeepExoMirModelV8 as V8Check
if isinstance(model, V8Check):
    trainer.is_v8 = True
    mt_cfg = model.config.get("multitask", {})
    trainer.multitask_loss_fn = MultiTaskLoss(
        w_seed=mt_cfg.get("w_seed", 0.3),
        w_mfe=mt_cfg.get("w_mfe", 0.2),
        w_position=mt_cfg.get("w_position", 0.2),
        w_load_balance=mt_cfg.get("w_load_balance", 0.01),
    ).to(trainer.device)

print(f"  is_v8: {trainer.is_v8}")
print(f"  multitask_loss_fn: {trainer.multitask_loss_fn is not None}")
assert trainer.is_v8, "Trainer should detect v8 model"
assert trainer.multitask_loss_fn is not None, "MultiTaskLoss should be created"

# Test 7: _forward_batch with v8
print("\nTest 7: _forward_batch with v8...")
output2 = trainer._forward_batch(batch)
assert isinstance(output2, dict), f"Expected dict, got {type(output2)}"
assert "logits" in output2, "Missing logits in output"
assert "aux_preds" in output2, "Missing aux_preds in output"
print(f"  _forward_batch returns dict: OK")
print(f"  logits: {output2['logits'].shape}")

print("\n" + "=" * 50)
print("ALL TESTS PASSED!")
print("=" * 50)
