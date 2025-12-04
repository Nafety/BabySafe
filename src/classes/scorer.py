from transformers import AutoTokenizer, AutoModelForCausalLM

class DangerScorerLLM:
    def __init__(self, model_name=None, device=None, max_tokens=200, temperature=0.7):
        self.model_name = model_name
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Tokenizer et modèle causal
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)

        # Pour certains modèles comme GPT2
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def assess(self, detections, masks, depth_map):
        prompt = self._build_prompt(detections, masks, depth_map)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=self.temperature,
            do_sample=True,
            top_p=0.9
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # garder seulement après le marqueur
        if "=== RESPONSE START ===" in text:
            text = text.split("=== RESPONSE START ===")[-1].strip()

        print("LLM Prompt:\n", prompt)
        print("LLM Response:\n", text)
        return text

    def _build_prompt(self, detections, masks, depth_map):
        prompt = (
            "You are an assistant analyzing objects for child safety.\n"
            "Use object positions, confidence, and depth information to determine risk.\n"
            "Objects that are closer to the child are more dangerous if they are harmful.\n\n"
            "Objects in the scene:\n"
        )
        for det, mask in zip(detections, masks):
            x1, y1, x2, y2 = det['bbox']
            depth_region = depth_map[y1:y2, x1:x2]
            mean_depth = float(depth_region.mean()) if depth_region.size > 0 else 0.0
            prompt += (
                f"- {det['label']}, position {det['bbox']}, confidence {det['conf']:.2f}, "
                f"mean_depth {mean_depth:.2f}\n"
            )

        # Exemple concret à suivre
        prompt += (
            "\nOutput instructions:\n"
            "For each object, output ONLY in the following format, without repeating the prompt:\n"
            "- Object: <label>\n"
            "  Danger: <low/medium/high>\n"
            "  Recommendation: <one short sentence>\n\n"
            "Example input:\n"
            "- fork, position [30, 270, 140, 400], confidence 0.8, mean_depth 0.5\n"
            "Example output:\n"
            "- Object: fork\n"
            "  Danger: high\n"
            "  Recommendation: Keep out of reach of child\n\n"
            "Now analyze the objects above and output strictly in the same format.\n"
        )

        # MARQUEUR de début de réponse
        prompt += "\n=== RESPONSE START ===\n"

        return prompt

class DangerScorer:
    def __init__(self, config=None):
        model_name = config.llm_cfg.get('model_name')
        temperature = config.llm_cfg.get('temperature', 0.7)
        device = config.pipeline_cfg.get('device', "cuda")
        max_tokens = config.llm_cfg.get('max_tokens', 200)
        self.llm = DangerScorerLLM(
            model_name=model_name,
            device=device,
            max_tokens=max_tokens,
            temperature=temperature
        )

    def score(self, detections, masks, depth_map):
        return self.llm.assess(detections, masks, depth_map)
