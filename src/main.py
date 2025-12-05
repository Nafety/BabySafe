import cv2
import numpy as np
from src.classes.segment import ObjectDetector
from src.classes.depth import DepthEstimator
from src.classes.scorer import DangerScorer
from src.configs.config import config

# ==========================================
# Pipeline principal temps réel
# ==========================================
class SafeSightPipeline:
    def __init__(self):
        self.detector = ObjectDetector(config)
        self.depth_estimator = DepthEstimator(config)
        self.scorer = DangerScorer(config)
        # Nombre de frames entre chaque inférence LLM
        self.frame_skip = config.pipeline_cfg.get("frame_skip", 5)
        self.frame_counter = 0
        self.last_report = None

    def process_frame(self, frame):
        self.frame_counter += 1

        # Détection et segmentation
        detections, masks = self.detector.detect_and_segment(frame)

        # Estimation de la profondeur
        depth_map = self.depth_estimator.estimate(frame)

        # Scoring des dangers tous les "frame_skip" frames
        if self.frame_counter % self.frame_skip == 0:
            self.last_report = self.scorer.score(detections, masks, depth_map)

        return detections, masks, self.last_report

# ==========================================
# Interface caméra
# ==========================================
def main():
    pipeline = SafeSightPipeline()
    cap = cv2.VideoCapture(0)  # Caméra par défaut

    if not cap.isOpened():
        print("Impossible d'ouvrir la caméra")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        detections, masks, danger_report = pipeline.process_frame(frame)

        # Affichage des masques uniquement pour les objets dangereux
        if detections and masks and danger_report:
            for det, mask in zip(detections, masks):
                obj_label = det['label']
                # Vérifie si objet présent dans le danger report
                obj_info = next((o for o in danger_report.get("objects", []) if o["object"] == obj_label), None)
                if obj_info is None:
                    continue

                # Couleur selon danger
                danger_level = obj_info["danger"]
                if danger_level == "low":
                    color = (0, 255, 0)
                elif danger_level == "medium":
                    color = (0, 165, 255)
                else:
                    color = (0, 0, 255)

                # Redimensionne le masque à la taille originale de l'image
                mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, color, 2)

        # Affiche le danger report dans une fenêtre séparée
        report_text = ""
        if danger_report:
            for obj in danger_report.get("objects", []):
                report_text += f"{obj['object']}: {obj['danger']} - {obj['recommendation']}\n"
        report_panel = np.zeros((h, 300, 3), dtype=np.uint8)
        for i, line in enumerate(report_text.split("\n")):
            cv2.putText(report_panel, line, (5, 20 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Combine frame + report panel
        combined = np.hstack((frame, report_panel))
        cv2.imshow("SafeSight - Danger Detection", combined)

        key = cv2.waitKey(1)
        if key == 27:  # ESC pour quitter
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
