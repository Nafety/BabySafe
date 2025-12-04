import cv2
from src.classes.detect import ObjectDetector  # YOLOv8-seg encapsulé
from src.classes.depth import DepthEstimator
from src.classes.scorer import DangerScorer
from src.configs.config import config
import glob
import os
import numpy as np

# ==========================================
# Pipeline principal simplifié
# ==========================================
class SafeSightPipeline:
    def __init__(self):
        self.detector = ObjectDetector(config)  # YOLOv8-seg
        self.depth_estimator = DepthEstimator(config)
        self.scorer = DangerScorer(config)

    def process_frame(self, frame):
        # YOLOv8-seg renvoie détections + masques
        detections, masks = self.detector.detect_and_segment(frame)

        # Estimation profondeur
        depth_map = self.depth_estimator.estimate(frame)

        # Scoring des dangers
        danger_report = self.scorer.score(detections, masks, depth_map)
        
        return detections, masks, danger_report

# ==========================================
# Exécution principale sur images statiques
# ==========================================
if __name__ == '__main__':
    pipeline = SafeSightPipeline()

    image_folder = "images"
    image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))

    os.makedirs("outputs", exist_ok=True)

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Impossible de lire {img_path}")
            continue

        detections, masks, report = pipeline.process_frame(frame)

        h, w = frame.shape[:2]

        if detections and masks:
            for det, mask in zip(detections, masks):
                # Redimensionne le masque à la taille originale de l'image
                mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

                # Trouve les contours du masque
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Dessine les contours verts
                cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        # Affichage texte danger
        print(f"Danger report for {img_path}:\n{report}\n")

        # Sauvegarde
        output_path = os.path.join("outputs", os.path.basename(img_path))
        cv2.imwrite(output_path, frame)
        print(f"Traitement terminé pour {img_path}, résultat sauvegardé dans {output_path}")
