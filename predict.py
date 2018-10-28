import os
import torch


model_dir = "LFW_model_torch"
latest_model = "cnn_epoch1.pkl"

model = torch.load(os.path.join(model_dir, latest_model))
