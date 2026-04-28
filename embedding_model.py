import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Union
import numpy as np
from tqdm import tqdm


class QwenEmbeddingModel:

    def __init__(self, model_path: str, device: str = "cuda:0", max_length: int = 8192, logger=None):
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.logger = logger

        self._load_model()

    def _load_model(self):

        if self.logger:
            self.logger.info(f"load: {self.model_path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32
            ).to(self.device)

            self.model.eval()

            if self.logger:
                self.logger.info("succeed")

        except Exception as e:
            if self.logger:
                self.logger.error(f"fail: {str(e)}")
            raise

    def encode(self, texts: Union[str, List[str]],
               batch_size: int = 32) -> np.ndarray:  # 建议将默认 batch_size 调大以加快构建速度

        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="vector"):
            batch_texts = texts[i:i + batch_size]

            # 编码
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']

                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

                embeddings = sum_embeddings / sum_mask

            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

        embeddings = np.vstack(all_embeddings)
        return embeddings

    def get_embedding_dimension(self) -> int:

        return self.model.config.hidden_size