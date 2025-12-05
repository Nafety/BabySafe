import yaml
import os
from dotenv import load_dotenv
from src.classes.scorer import DangerScorer

class Config:
    _instance = None

    def __new__(cls, yaml_path="config.yml", env_path=".env"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_yaml(yaml_path)
            cls._instance._load_env(env_path)
        return cls._instance

    def _load_yaml(self, path):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        self.pipeline_cfg = cfg.get("pipeline", {})
        self.detection_cfg = cfg.get("detection", {})
        self.depth_cfg = cfg.get("depth", {})
        self.classes = cfg.get("classes", [])
        self.llm_cfg = cfg.get("llm", {})

    def _load_env(self, path):
        # charge le .env
        load_dotenv(dotenv_path=path)
        # récupère la clé Gemini API
        self.llm_cfg['api_key'] = os.getenv("GEMINI_API_KEY", None)

# singleton à importer partout
config = Config()
