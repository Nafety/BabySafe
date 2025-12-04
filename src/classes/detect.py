from ultralytics import YOLO
import numpy as np

class YOLOObjectDetector:
    def __init__(self, model_path=None):
        self.model = YOLO(model_path)

    def detect_and_segment(self, image):
        """
        Retourne détections et masques binaires prêts à l'emploi (taille de l'image)
        """
        results = self.model.predict(image, verbose=False)
        detections = []
        masks = []

        for r in results:
            if r.boxes is None or r.masks is None:
                continue

            names = r.names
            mask_data = r.masks.data if r.masks.data is not None else []

            for i, (box, cls, conf) in enumerate(zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf)):
                cls_int = int(cls)
                detections.append({
                    'bbox': [int(x) for x in box],
                    'class': cls_int,
                    'label': names.get(cls_int, str(cls_int)),
                    'conf': float(conf)
                })

                # Masque binaire de la taille de l'image
                if len(mask_data) > i:
                    masks.append(mask_data[i].cpu().numpy().astype(np.uint8))
                else:
                    # fallback : masque vide
                    masks.append(np.zeros(image.shape[:2], dtype=np.uint8))

        return detections, masks


class ObjectDetector:
    def __init__(self, config=None):
        model_path = getattr(config, "model_path", None)
        self.model = YOLOObjectDetector(model_path=model_path)

    def detect_and_segment(self, image):
        return self.model.detect_and_segment(image)
