import torch
from PIL import Image
import numpy as np
import base64
import io
import requests
import json

class VisionLLMNode:
    """
    A versatile ComfyUI node that integrates with OpenAI-compatible LLMs.
    It can function as a standard text-generation node or as a multimodal
    vision analyzer if an image is provided.
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the node, separating required and optional inputs.
        """
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "base_url": ("STRING", {"multiline": False, "default": "https://api.openai.com/v1"}),
                "role_prompt": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "user_prompt": ("STRING", {"multiline": True, "default": "Describe this image in detail."}),
                "model": ("STRING", {"multiline": False, "default": "gpt-4o"}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "analyze"
    CATEGORY = "LLM"

    def pil_to_base64(self, image: Image.Image) -> str:
        """Converts a PIL Image to a Base64 encoded string."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def analyze(self, api_key: str, base_url: str, role_prompt: str, user_prompt: str, model: str, image: torch.Tensor = None):
        """
        Main function that handles the logic for both text and image analysis.
        It builds and sends a request to the LLM API and returns the response.
        """
        if not api_key:
            return ("Error: API key is missing. Please provide your API key in the node.",)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # System message is always included
        messages = [
            {"role": "system", "content": role_prompt}
        ]

        # Dynamically build the user message based on whether an image is provided
        if image is not None:
            # Convert the ComfyUI tensor to a PIL Image, then to Base64
            pil_image = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
            base64_image = self.pil_to_base64(pil_image)
            
            # Create a multimodal user message
            user_content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        else:
            # Create a text-only user message
            user_content = user_prompt

        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 2048  # Increased for potentially more detailed descriptions
        }

        try:
            response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=120) # Added timeout
            response.raise_for_status()
            result = response.json()
            
            # Safely access the response content
            if 'choices' in result and result['choices'] and 'message' in result['choices'] and 'content' in result['choices']['message']:
                return (result['choices']['message']['content'],)
            else:
                return (f"Error: Unexpected API response format. Response: {response.text}",)

        except requests.exceptions.RequestException as e:
            return (f"Error calling API: {e}",)
        except KeyError:
            return (f"Error parsing API response. Check the console for details. Response: {response.text}",)


# Mapping node class and display name for ComfyUI
NODE_CLASS_MAPPINGS = {
    "VisionLLMNode": VisionLLMNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisionLLMNode": "Vision LLM Analyzer"
}