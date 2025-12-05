from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json

# installer le SDK Gemini : pip install google‑genai
from google import genai
import json
import re

def parse_llm_json(text):
    # Supprime les ```json ... ``` ou """ ... """
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"```$", "", text)
    text = re.sub(r'^"""', '', text)
    text = re.sub(r'"""$', '', text)
    
    # Retire les retours à la ligne superflus
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print("Error parsing JSON from LLM:", e)
        print("Raw LLM output:", text)
        return None

class DangerScorerLLM:
    def __init__(self, api_key=None, model_name="gemini-2.5-flash", temperature=0.7):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        if self.api_key is None:
            raise ValueError("api_key required for Gemini API usage")
        self.client = genai.Client(api_key=self.api_key)

    def assess(self, detections, masks, depth_map):
        prompt = self._build_prompt(detections, masks, depth_map)
        print("Prompt sent to LLM:", prompt)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        text = response.text.strip()
        print("LLM Response:", text)
        try:
            result = parse_llm_json(text)
        except json.JSONDecodeError as e:
            print("Error parsing JSON from LLM:", e)
            return None

        return result

    def _build_prompt(self, detections, masks, depth_map):
        # Construire un prompt demandant du JSON valide
        prompt = {
            "objects": []
        }
        for det, mask in zip(detections, masks):
            x1, y1, x2, y2 = det['bbox']
            depth_region = depth_map[y1:y2, x1:x2]
            mean_depth = float(depth_region.mean()) if depth_region.size > 0 else 0.0
            prompt["objects"].append({
                "label": det['label'],
                "bbox": det['bbox'],
                "confidence": round(det['conf'], 2),
                "mean_depth": round(mean_depth, 3)
            })

        instruction = (
            "You are in charge of one or multiple babies and you are supervising them and make sure they are all safe:\n"
            "For each object in 'objects', output a JSON with exactly this schema:\n"
            "{\n"
            "  \"objects\": [\n"
            "    {\"label\": str, \"danger\": \"low\"|\"medium\"|\"high\", \"recommendation\": str},\n"
            "    ...\n"
            "  ]\n"
            "}\n"
            "Choose danger level taking into account mean_depth (closer to any baby = more dangerous) and object type.\n"
            "Return only the JSON, no extra text."
        )

        full_prompt = {
            "scene": prompt,
            "instruction": instruction
        }

        return json.dumps(full_prompt)


class DangerScorer:
    def __init__(self, config=None):
        model_name = config.llm_cfg.get('model_name')
        temperature = config.llm_cfg.get('temperature', 0.7)
        api_key = config.llm_cfg.get('api_key', None)
        
        self.llm = DangerScorerLLM(
            model_name=model_name,
            temperature=temperature,
            api_key=api_key
        )

    def score(self, detections, masks, depth_map):
        return self.llm.assess(detections, masks, depth_map)
