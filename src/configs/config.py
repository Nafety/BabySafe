import yaml
from src.classes.scorer import DangerScorer


class Config:
    _instance = None

    def __new__(cls, yaml_path="config.yml"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_yaml(yaml_path)
            cls._instance._initialize_models()  # crée directement YOLO, SAM, etc.
        return cls._instance

    def _load_yaml(self, path):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        self.pipeline_cfg = cfg.get("pipeline", {})
        self.models_cfg = cfg.get("models", {})
        self.classes = cfg.get("classes", [])
        self.llm_cfg = cfg.get("llm", {})

    def _initialize_models(self):
        device = self.pipeline_cfg.get("device", "cuda")
        # YOLO
        self.model_path = self.models_cfg['yolo']['path']
        # Depth
        self.midas_path = self.models_cfg['depth']['model_path']

# singleton à importer partout
config = Config()
