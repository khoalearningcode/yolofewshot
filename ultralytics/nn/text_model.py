from abc import abstractmethod
import clip
import mobileclip
import torch.nn as nn
from ultralytics.utils.torch_utils import smart_inference_mode
import torch
from ultralytics.utils import LOGGER
from pathlib import Path
import os

class TextModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def tokenize(texts):
        pass
    
    @abstractmethod
    def encode_text(texts, dtype):
        pass

class CLIP(TextModel):
    def __init__(self, size, device):
        super().__init__()
        self.model = clip.load(size, device=device)[0]
        self.to(device)
        self.device = device
        self.eval()
    
    def tokenize(self, texts):
        return clip.tokenize(texts).to(self.device)
    
    @smart_inference_mode()
    def encode_text(self, texts, dtype=torch.float32):
        txt_feats = self.model.encode_text(texts).to(dtype)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        return txt_feats
        
class MobileCLIP(TextModel):
    
    config_size_map = {
        "s0": "s0",
        "s1": "s1",
        "s2": "s2",
        "b": "b",
        "blt": "b"
    }
    
    def __init__(self, size, device):
        super().__init__()
        config = self.config_size_map[size]
        
        # Try to find mobileclip weight file in multiple locations
        weight_name = f'mobileclip_{size}.pt'
        possible_paths = [
            # Relative to current working directory
            weight_name,
            # In yolofewshot root (from yolofewshot/)
            Path(__file__).parent.parent.parent / weight_name,
            # In ultralytics root (from yolofewshot/ultralytics/)
            Path(__file__).parent.parent.parent / 'yolofewshot' / weight_name,
            # Check if running from outside yolofewshot
            Path('yolofewshot') / weight_name,
        ]
        
        pretrained_path = None
        for path in possible_paths:
            if Path(path).exists():
                pretrained_path = str(Path(path).resolve())
                LOGGER.info(f"Found {weight_name} at {pretrained_path}")
                break
        
        # If not found, don't use pretrained weights
        if pretrained_path is None:
            LOGGER.warning(f"Could not find {weight_name}, loading model without pretrained weights")
            pretrained_path = None
        else:
            LOGGER.info(f"Loading {weight_name} from {pretrained_path}")
        
        self.model = mobileclip.create_model_and_transforms(f'mobileclip_{config}', pretrained=pretrained_path, device=device)[0]
        self.tokenizer = mobileclip.get_tokenizer(f'mobileclip_{config}')
        self.to(device)
        self.device = device
        self.eval()
    
    def tokenize(self, texts):
        text_tokens = self.tokenizer(texts).to(self.device)
        # max_len = text_tokens.argmax(dim=-1).max().item() + 1
        # text_tokens = text_tokens[..., :max_len]
        return text_tokens

    @smart_inference_mode()
    def encode_text(self, texts, dtype=torch.float32):
        text_features = self.model.encode_text(texts).to(dtype)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features

def build_text_model(variant, device=None):
    LOGGER.info(f"Build text model {variant}")
    base, size = variant.split(":")
    if base == 'clip':
        return CLIP(size, device)
    elif base == 'mobileclip':
        return MobileCLIP(size, device)
    else:
        print("Variant not found")
        assert(False)