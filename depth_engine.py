import torch
import timm
import numpy as np
from PIL import Image

# MiDaS comes pre-packaged in torch.hub so we go not need to download it ourselves
def load_model():
    model_type = "DPT_Large"  
    # we have option to use DPT_Large, DPT_Hybrid, MiDaS_small
    # baad mei we will compare all three
    
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()  
    # used inference mode, not training mode
    
    # Load the matching image transforms
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform  # matches DPT_Large
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    
    return midas, transform, device


def get_depth_map(image: Image.Image, model, transform, device):
    # Converting image numpy RGB, because MiDaS requires that
    img_rgb = np.array(image.convert("RGB"))
    
    # Apply MiDaS transforms (resize, normalize, to tensor)
    input_tensor = transform(img_rgb).to(device)
    
    with torch.no_grad():  # no gradient calculation needed for inference
        prediction = model(input_tensor)
        
        # Now we need to whatever was the input imiage to its original size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],  # this is for height and with
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    return depth_map