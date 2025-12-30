import os
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import numpy as np

from models.model import StateSpaceModel
from models.decoder import TransposedConvDecoder
from dataset.agent_teacher_dataset import AgentTeacherDataset
import yaml
import pprint

# ========== Utility ==========
def denormalize(image_tensor):
    """Convert tensor image to [0, 255] range and numpy format."""
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(3, 1, 1)
    return (image_tensor * STD + MEAN).clamp(0, 1)

def save_image_grid(images, filepath):
    """Save image grid as a single file."""
    from torchvision.utils import save_image
    save_image(images, filepath, nrow=len(images))

# ========== Load Config ==========
with open('configs/metaworld_data.yaml', 'r') as f:
    config = yaml.safe_load(f)

data_config = config['data']
data_config['test_tasks'] = ["button-press-v2"]
data_config['novar'] = True
model_config = config['model']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========== Dataset & Loader ==========
val_dataset = AgentTeacherDataset(**data_config, mode='test')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

# ========== Load Model ==========
ss_model = StateSpaceModel(
    latent_dim=model_config['latent_dim'],
    out_dim=model_config['out_dim'],
    waypoints=model_config['waypoints'],
    sub_waypoints=model_config['sub_waypoints'],
    ent_head=model_config['ent_head'],
    num_ent_head=model_config['num_ent_head'],
    metaworld='metaworld' in data_config['agent_dir']
).to(device)

ss_model.forward_model.transformer.h[0].attn.use_sdpa=False

decoder = TransposedConvDecoder(
    observation_shape=(3, 224, 224),
    emb_dim=512*2,
    activation=nn.ReLU,
    depth=32,
    kernel_size=5,
    stride=3
).to(device)

# Load weights
model_path = 'checkpoints/downstream_train_awda/model_seq_alpha1_5.pt'
checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)['model_state_dict']
pretrained_dict = checkpoint
new_state_dict = {}
for k, v in pretrained_dict.items():
    new_key = k.replace("module.", "")  # Remove the "module." prefix
    new_state_dict[new_key] = v
ss_model.load_state_dict(new_state_dict)
print("Loaded pretrained weights for the ss_model.")


decoder_path = 'checkpoints/downstream_train_awda/decoder2.pt'
checkpoint = torch.load(decoder_path, map_location=torch.device('cpu'), weights_only=True)['model_state_dict']
pretrained_dict = checkpoint
new_state_dict = {}
for k, v in pretrained_dict.items():
    new_key = k.replace("module.", "")  # Remove the "module." prefix
    new_state_dict[new_key] = v
decoder.load_state_dict(new_state_dict)
print("Loaded pretrained weights for the decoder.")

ss_model.eval()
decoder.eval()

# ========== Inference & Save ==========
save_dir = 'trajectory_reconstructions'
os.makedirs(save_dir, exist_ok=True)

with torch.inference_mode():
    for idx, (expert_context, agent_traj, _) in enumerate(tqdm(val_loader, desc="Saving trajectories", total=len(val_loader))):
        context = expert_context['video'].to(device)  # (B, T1, 3, 224, 224)
        images = agent_traj['images'].to(device)      # (B, T2, 3, 224, 224)
        head_label = agent_traj['head_label'].to(device)

        # Get latent features
        out = ss_model(None, images, context, pre_train=False, ents=head_label)
        first_state = out['enc_features'][:,10:11]
        forward_states = out['forward_states']  # (B, T3, D)
        forward_states = torch.cat([first_state, forward_states], dim=1)
        # forward_states = out['enc_features']  # (B, T3, D)

        B, T3, D = forward_states.shape
        recon = decoder(forward_states.view(B*T3, D))  # (B*T3, 3, 224, 224)
        recon = denormalize(recon).cpu()

        # Save each frame
        for t in range(T3):
            img_tensor = recon[t]
            filename = f"sample{idx:03d}_t{t:02d}.png"
            filepath = os.path.join(save_dir, filename)
            img_pil = Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            img_pil.save(filepath)

        # Only do a few for demo
        # if idx >= 9:
        #     break

print(f"Images saved in: {save_dir}")
