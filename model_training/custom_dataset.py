# custom_dataset.py

import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class MultiImageJSONLDataset(Dataset):
    def __init__(self, jsonl_path, processor, base_image_dir, max_length=512):
        self.processor = processor
        self.base_image_dir = base_image_dir
        self.max_length = max_length

        # Load all examples
        import jsonlines
        with jsonlines.open(jsonl_path) as reader:
            self.data = list(reader)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Resolve image paths
        image_paths = [
            os.path.join(self.base_image_dir, os.path.normpath(img.replace("panels/", "")))
            for img in item["images"]
        ]

        # Load images and force RGB
        images = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")  # Force 3-channel RGB
                images.append(img)
            except Exception as e:
                raise ValueError(f"Error loading image {img_path}: {e}")

        # Build prompt
        prompt = ""
        for turn in item["conversations"]:
            if turn["from"] == "human":
                prompt += turn["value"] + "\n"
            else:
                response = turn["value"]


        encoding = self.processor(
            text=prompt,
            images=images,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            image_processor_kwargs={"size": {"shortest_edge": 384}}
        )

        # Ensure pixel_values is 3D [C, H, W]
        pixel_values = encoding["pixel_values"]

        if pixel_values.dim() == 2:  # [H, W]
            pixel_values = pixel_values.unsqueeze(0).repeat(3, 1, 1)  # → [3, H, W]
        elif pixel_values.dim() == 3 and pixel_values.shape[-1] == 3:  # [H, W, C]
            pixel_values = pixel_values.permute(2, 0, 1)  # → [C, H, W]

        # Tokenize response
        labels = self.processor.tokenizer(
            response,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "pixel_values": pixel_values,
            "labels": labels.flatten()
        }
