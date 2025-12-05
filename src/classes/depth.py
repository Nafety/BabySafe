import cv2
import numpy as np
import torch
from libs.MiDaS.midas.midas_net import MidasNet
from libs.MiDaS.midas.transforms import Resize

class MiDaSDepthEstimator:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model = MidasNet(model_path)
        self.model.to(self.device).eval()
        self.transform = Resize(
            384, 384, resize_target=None, keep_aspect_ratio=True
        )

    def predict(self, image):
        # Appliquer Resize pour garder le ratio
        img_input = self.transform({"image": image})["image"]

        # Convertir en torch tensor et permuter canaux
        img_input = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).to(self.device).float()

        # Ajouter padding pour multiples de 32
        _, _, H, W = img_input.shape
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        img_input = torch.nn.functional.pad(img_input, (0, pad_w, 0, pad_h), mode='constant', value=0)

        with torch.no_grad():
            depth = self.model(img_input)

        depth = depth.squeeze().cpu().numpy()

        # Si padding ajouté, retirer le padding pour correspondre à l'image originale
        if pad_h > 0 or pad_w > 0:
            depth = depth[:H, :W]

        # Normalisation
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        return depth



class DepthEstimator:
    def __init__(self, config=None):
        # utilise config centralisée pour init
        device = config.pipeline_cfg.get('device', "cuda")
        midas_path = config.depth_cfg.get('model_path')
        self.model = MiDaSDepthEstimator(model_path=midas_path,device=device)

    def estimate(self, image):
        return self.model.predict(image)
