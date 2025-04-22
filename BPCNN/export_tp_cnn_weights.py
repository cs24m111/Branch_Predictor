import numpy as np
import torch

# === CONFIGURATION ===
pth_file = f"models/models_tpcnn/cnn_branch_predictor_epoch2_step3478190.pth"
output_header = "tp_cnn_weights.h"
table_size = 2048
T_size = 2048            # Index space for (IP << 1 + Dir) & 0xFF
num_filters = 32
history_len = 200
quant_threshold = 0.8   # same as self.q from model

# === Load checkpoint ===
checkpoint = torch.load(pth_file, map_location='cpu')

# === Extract Layer 1: conv1 weight → T[256][filters][2] ===
conv_weight = checkpoint.get("conv1.weight")
if conv_weight is None:
    for k in checkpoint:
        if "conv1.weight" in k:
            conv_weight = checkpoint[k]
            break
assert conv_weight is not None, "conv1.weight not found"

conv_weight = conv_weight.squeeze(-1).squeeze(1).numpy()  # shape: [filters, table_size]

# Create T by folding table_size into 256 via masking
T = np.zeros((T_size, num_filters, 2), dtype=np.uint8)
for i in range(table_size):
    folded_i = i & 0xFF
    for f in range(num_filters):
        w = conv_weight[f][i]
        if w > quant_threshold:
            T[folded_i][f] = [0, 1]
        elif w < -quant_threshold:
            T[folded_i][f] = [1, 1]
        else:
            T[folded_i][f] = [0, 0]

# === Extract Layer 2: fc weights → L2[200][filters][2] ===
fc_weight = checkpoint.get("fc.weight")
if fc_weight is None:
    for k in checkpoint:
        if "fc" in k and "weight" in k:
            fc_weight = checkpoint[k]
            break
assert fc_weight is not None, "fc.weight not found"

fc_weight = fc_weight.view(num_filters, history_len).t().numpy()  # shape: [200, 32]
L2 = np.zeros((history_len, num_filters, 2), dtype=np.uint8)

for i in range(history_len):
    for j in range(num_filters):
        val = fc_weight[i][j]
        if val > quant_threshold:
            L2[i][j] = [0, 1]  # +1
        elif val < -quant_threshold:
            L2[i][j] = [1, 1]  # -1
        else:
            L2[i][j] = [0, 0]  # 0

# === Extract threshold k for prediction
k = checkpoint.get("k", torch.tensor(0.5)).item()
threshold = int(k * 100)

# === Export to header
def export_to_header(T, L2, threshold, filename="tp_cnn_weights.h"):
    with open(filename, "w") as f:
        f.write("#ifndef TP_CNN_WEIGHTS_H\n#define TP_CNN_WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")

        f.write("const uint8_t T[2048][32][2] = {\n")
        for i in range(2048):
            f.write("  {\n")
            for j in range(32):
                s, v = T[i][j]
                f.write(f"    {{{s}, {v}}},\n")
            f.write("  },\n")
        f.write("};\n\n")

        f.write("const uint8_t L2[200][32][2] = {\n")
        for i in range(200):
            f.write("  {\n")
            for j in range(32):
                s, v = L2[i][j]
                f.write(f"    {{{s}, {v}}},\n")
            f.write("  },\n")
        f.write("};\n\n")

        f.write(f"const int CNN_THRESHOLD = {threshold};\n")
        f.write("#endif\n")

export_to_header(T, L2, threshold, output_header)
print(f"✅ Exported to {output_header}")