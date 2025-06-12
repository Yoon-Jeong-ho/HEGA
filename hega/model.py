import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
from typing import List


class HEGAModel:
    """Hybrid Embedding-to-Generation Architecture."""

    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1", l_cut: int = 16,
                 use_embedding_for_generation: bool = True, k: int = 4, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.l_cut = l_cut
        self.k = k
        self.use_embedding_for_generation = use_embedding_for_generation
        self.index: faiss.IndexFlatL2 | None = None
        self.texts: List[str] = []

    def _forward_until(self, hidden_states: torch.Tensor, end_layer: int) -> torch.Tensor:
        for i in range(end_layer):
            layer = self.model.model.layers[i]
            hidden_states = layer(hidden_states)[0]
        return hidden_states

    def encode(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        hidden_states = self.model.model.embed_tokens(inputs.input_ids)
        hidden_states = self._forward_until(hidden_states, self.l_cut)
        pooled = hidden_states.mean(dim=1)
        return pooled.detach().cpu()

    def index_texts(self, texts: List[str]):
        self.texts.extend(texts)
        embeddings = self.encode(texts).numpy()
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def retrieve(self, query: str) -> List[str]:
        if self.index is None:
            return []
        query_emb = self.encode([query]).numpy()
        distances, indices = self.index.search(query_emb, self.k)
        return [self.texts[i] for i in indices[0] if i < len(self.texts)]

    def _forward_from(self, hidden_states: torch.Tensor, start_layer: int) -> torch.Tensor:
        for i in range(start_layer, len(self.model.model.layers)):
            layer = self.model.model.layers[i]
            hidden_states = layer(hidden_states)[0]
        hidden_states = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)
        return logits

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        retrieved_texts = self.retrieve(prompt)
        context = prompt
        if retrieved_texts:
            context += "\n" + "\n".join(retrieved_texts)
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        if self.use_embedding_for_generation:
            hidden_states = self.model.model.embed_tokens(inputs.input_ids)
            hidden_states = self._forward_until(hidden_states, self.l_cut)
            logits = self._forward_from(hidden_states, self.l_cut)
            generated = self.model.generate(inputs_embeds=logits, max_new_tokens=max_new_tokens,
                                            pad_token_id=self.tokenizer.eos_token_id)
        else:
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                            pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)
