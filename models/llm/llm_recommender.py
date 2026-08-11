"""LLM-based recommender with LoRA/QLoRA support"""

from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)


class LLMRecommender(nn.Module):
    """
    LLM-based recommendation model with LoRA fine-tuning

    Supports:
    - Full fine-tuning
    - LoRA (Low-Rank Adaptation)
    - QLoRA (Quantized LoRA with 4-bit/8-bit)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Model configuration dict with keys:
                - name or model_name: HF model name/path
                - lora: LoRA config (r, alpha, dropout, target_modules)
                - quantization: "4bit", "8bit", or None
                - max_length: Maximum sequence length
                - device: Device to load model on
        """
        super().__init__()
        self.config = config
        self.model_name = config.get('model_name') or config.get('name')
        self.max_length = config.get('max_length', 512)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Setup quantization config
        quantization_config = self._get_quantization_config()

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if quantization_config is None else None
        )

        # Apply LoRA if configured
        if 'lora' in config and config['lora']:
            self._apply_lora()

        # Recommendation head (project LLM hidden states to scores)
        hidden_size = self.model.config.hidden_size
        self.rec_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization config for QLoRA"""
        quant_type = self.config.get('quantization', None)

        if quant_type == '4bit':
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quant_type == '8bit':
            return BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            return None

    def _apply_lora(self):
        """Apply LoRA to the model"""
        lora_config = self.config['lora']

        # Prepare model for k-bit training if quantized
        if self.config.get('quantization'):
            self.model = prepare_model_for_kbit_training(self.model)

        # LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('lora_alpha', 32),
            lora_dropout=lora_config.get('lora_dropout', 0.1),
            target_modules=lora_config.get('target_modules', ["q_proj", "v_proj"])
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size] (optional, for training)

        Returns:
            Dict with:
                - logits: [batch_size, 1] recommendation scores
                - loss: scalar (if labels provided)
                - hidden_states: [batch_size, hidden_size]
        """
        # Get LLM outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Get last hidden state of the last token (or mean pooling)
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]

        # Use last token representation
        last_token_idx = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled_output = hidden_states[batch_indices, last_token_idx]  # [batch_size, hidden_size]

        # Recommendation score
        logits = self.rec_head(pooled_output).squeeze(-1)  # [batch_size]

        result = {
            'logits': logits,
            'hidden_states': pooled_output
        }

        # Compute loss if labels provided
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            result['loss'] = loss

        return result

    def generate_prompt(
        self,
        user_id: int,
        item_id: int,
        graph_context: str = "",
        user_history: Optional[str] = None
    ) -> str:
        """
        Generate input prompt for the LLM

        Args:
            user_id: User ID
            item_id: Item ID
            graph_context: Serialized graph structure
            user_history: User interaction history

        Returns:
            Formatted prompt string
        """
        prompt = f"### Recommendation Task\n"
        prompt += f"User: {user_id}\n"
        prompt += f"Candidate Item: {item_id}\n"

        if user_history:
            prompt += f"User History: {user_history}\n"

        if graph_context:
            prompt += f"Graph Context:\n{graph_context}\n"

        prompt += f"Will user interact with this item? (yes/no): "

        return prompt

    def encode_batch(
        self,
        prompts: list,
        max_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize a batch of prompts

        Args:
            prompts: List of prompt strings
            max_length: Max sequence length

        Returns:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        """
        if max_length is None:
            max_length = self.max_length

        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        return encoded['input_ids'], encoded['attention_mask']

    def predict_batch(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        graph_contexts: Optional[list] = None
    ) -> torch.Tensor:
        """
        Predict scores for a batch of user-item pairs

        Args:
            user_ids: [batch_size]
            item_ids: [batch_size]
            graph_contexts: List of graph context strings

        Returns:
            scores: [batch_size]
        """
        # Generate prompts
        prompts = []
        for i in range(len(user_ids)):
            user_id = user_ids[i].item()
            item_id = item_ids[i].item()
            context = graph_contexts[i] if graph_contexts else ""

            prompt = self.generate_prompt(user_id, item_id, context)
            prompts.append(prompt)

        # Encode
        input_ids, attention_mask = self.encode_batch(prompts)
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        # Forward
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)

        return outputs['logits']

    def save_pretrained(self, save_path: str):
        """Save model and tokenizer"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        torch.save(self.rec_head.state_dict(), f"{save_path}/rec_head.pt")

    def load_pretrained(self, load_path: str):
        """Load model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        # Note: For LoRA models, use PeftModel.from_pretrained
        self.rec_head.load_state_dict(torch.load(f"{load_path}/rec_head.pt"))

    def recommend(
        self,
        prompt: str,
        top_k: int = 10,
        return_explanation: bool = False,
        temperature: float = 0.7,
        max_new_tokens: int = 256
    ) -> Dict[str, any]:
        """
        Generate recommendations based on a prompt

        Args:
            prompt: Input prompt describing user history and recommendation task
            top_k: Number of recommendations to return
            return_explanation: Whether to generate explanations
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dict with:
                - item_ids: List of recommended item IDs
                - scores: List of recommendation scores (optional)
                - explanation: Generated explanation (if return_explanation=True)
        """
        # Encode prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Parse recommendations from generated text
        # Expected format: "1. Item 123: description\n2. Item 456: description\n..."
        item_ids = self._parse_item_ids(generated_text, top_k)

        result = {'item_ids': item_ids}

        if return_explanation:
            result['explanation'] = generated_text

        return result

    def _parse_item_ids(self, generated_text: str, top_k: int) -> list:
        """
        Parse item IDs from generated recommendation text

        Args:
            generated_text: Generated text from LLM
            top_k: Maximum number of items to extract

        Returns:
            List of item IDs
        """
        import re

        item_ids = []

        # Try to find patterns like "Item 123", "item 456", "#123", etc.
        patterns = [
            r'[Ii]tem\s+(\d+)',  # "Item 123" or "item 123"
            r'#(\d+)',           # "#123"
            r'ID:\s*(\d+)',      # "ID: 123"
            r'^\s*(\d+)\.',      # "1. " at start of line (line number, not item ID)
        ]

        for pattern in patterns[:3]:  # Skip the last one (line numbers)
            matches = re.findall(pattern, generated_text)
            if matches:
                item_ids.extend([int(m) for m in matches])
                if len(item_ids) >= top_k:
                    break

        # If no items found, generate random recommendations as fallback
        if not item_ids:
            import random
            item_ids = random.sample(range(1000), min(top_k, 100))

        return item_ids[:top_k]
