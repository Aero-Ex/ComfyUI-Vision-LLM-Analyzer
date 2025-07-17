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

        messages = [{"role": "system", "content": role_prompt}]

        if image is not None:
            pil_image = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
            base64_image = self.pil_to_base64(pil_image)
            user_content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        else:
            user_content = user_prompt

        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 2048
        }

        try:
            response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()

            # --- BUG FIX: Correctly parse the JSON response ---
            # The 'choices' field is a list, so we access its first element [0].
            if 'choices' in result and result['choices']:
                first_choice = result['choices'][0]
                if 'message' in first_choice and 'content' in first_choice['message']:
                    content = first_choice['message']['content']
                    return (content,)  # Success! Return the content.

            # If the expected structure is not found, return a detailed error.
            return (f"Error: Could not find 'content' in the API response. Full response: {response.text}",)

        except requests.exceptions.RequestException as e:
            return (f"Error calling API: {e}",)
        except json.JSONDecodeError:
            return (f"Error: Failed to decode JSON from API response. Response text: {response.text}",)
        except IndexError:
             return (f"Error: 'choices' list in API response was empty. Full response: {response.text}",)
        except Exception as e:
            # Catch any other unexpected errors.
            return (f"An unexpected error occurred: {e}. Full response: {response.text}",)


# Mapping node class and display name for ComfyUI
NODE_CLASS_MAPPINGS = {
    "VisionLLMNode": VisionLLMNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisionLLMNode": "Vision LLM Analyzer"
}
