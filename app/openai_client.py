# import base64
# import json
# import os
# from typing import Any, Dict, List, Optional

# from openai import OpenAI


# class OpenAIClient:
#     """
#     Thin wrapper around OpenAI SDK to handle:
#     - client initialization via env var OPENAI_API_KEY
#     - calling vision model to extract structured features as JSON
#     - calling text model to generate pickup lines as JSON
#     """

#     def __init__(self, api_key: Optional[str] = None):
#         self.api_key =  os.getenv("OPENAI_API_KEY")
#         if not self.api_key:
#             raise RuntimeError("OPENAI_API_KEY is not set. Please export it or put it in your .env file.")
#         # Initialize client from environment to avoid constructor incompatibilities (e.g., unexpected 'proxies')
#         self.client = OpenAI(
#             api_key=self.api_key,
            
#         )
#         print(self.client)
#         # Default models; you can adjust as needed
#         self.vision_model = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
#         print(self.vision_model)
#         self.text_model = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")
#         print(self.text_model)


#     # def _to_data_url(image_bytes: bytes, mime_type: Optional[str] = None) -> str:
#     #     # Attempt a basic mime inference if none supplied
#     #     if not mime_type:
#     #         mime_type = "image/jpeg"
#     #     b64 = base64.b64encode(image_bytes).decode("utf-8")
#     #     return f"data:{mime_type};base64,{b64}"

#     def describe_image(self, image_bytes: bytes, mime_type: Optional[str] = None) -> Dict[str, Any]:
#         """
#         Calls a vision-capable model to describe an image and extract structured attributes.
#         Returns a dict with keys: description (str), attributes (List[str]), vibes (List[str]).
#         """
#         data_url = self._to_data_url(image_bytes, mime_type)
#         system = (
#             "You are a helpful assistant that analyzes dating profile photos. "
#             "Extract objective, respectful, non-judgmental details suitable for crafting pickup lines. "
#             "Keep it concise and avoid sensitive or personal data inferences."
#         )
#         user_text = (
#             "Analyze the image and output strict JSON with keys: "
#             "description (string), attributes (string[]), vibes (string[]). "
#             "attributes should be short keywords like 'smiling', 'beach', 'guitar', 'dog', 'sunset', 'glasses'. "
#             "vibes are 1-3 words like 'adventurous', 'artsy', 'bookish'."
#         )
#         resp = self.client.chat.completions.create(
#             model=self.vision_model,
#             response_format={"type": "json_object"},
#             temperature=0.5,
#             messages=[
#                 {"role": "system", "content": system},
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": user_text},
#                         {"type": "image_url", "image_url": {"url": data_url}},
#                     ],
#                 },
#             ],
#         )
#         content = resp.choices[0].message.content
#         try:
#             data = json.loads(content)
#         except Exception:
#             # Fallback to a safe schema if model didn't obey JSON strictly
#             data = {"description": content, "attributes": [], "vibes": []}
#         # Ensure basic shape
#         data.setdefault("description", "")
#         data.setdefault("attributes", [])
#         data.setdefault("vibes", [])
#         return data

#     def generate_pickup_lines(
#         self,
#         features: Dict[str, Any],
#         dataset_examples: Optional[List[str]] = None,
#         count: int = 5,
#         tone: Optional[str] = None,

#         temperature: Optional[float] = None,
#     ) -> Dict[str, Any]:
#         """
#         Generate pickup lines in JSON format using features and optional dataset as style examples.
#         Returns a dict with key: lines (List[str]).
#         """
#         desc = features.get("description", "")
#         attributes = features.get("attributes", [])
#         vibes = features.get("vibes", [])
#         examples_block = "\n\nEXAMPLE LINES (style cues only):\n" + "\n".join(dataset_examples or []) if dataset_examples else ""
#         tone_text = f"Desired tone: {tone}." if tone else ""

#         system = (
#             "You are indian boy who like to flirt with girls. You have experience of flirting with 1000+ girls."  
#             "Prefer playful teasing , one-liners suitable for opening messages."
#             "You will create pickupline from image desciption you are given."
#         )
#         user = (
#             "Using the provided photo features, produce strictly JSON: {\"lines\": string[]} with "
#             f"exactly {max(1, min(count, 20))} lines.\n"
#             "Guidelines:\n"
#             "- Tailor lines to the attributes and vibes.\n"
#             "- Keep it friendly and fun; avoid overflattery.\n"
#             "- Optionally, include light wordplay or puns.\n"
#             "- Avoid assumptions about identity, beliefs, or sensitive traits.\n"
#             f"{tone_text}\n\n"
#             f"FEATURES:\nDescription: {desc}\nAttributes: {attributes}\nVibes: {vibes}"
#             f"{examples_block}"
#         )

#         resp = self.client.chat.completions.create(
#             model= self.text_model,
#             response_format={"type": "json_object"},
#             temperature=temperature if temperature is not None else 0.9,
#             messages=[
#                 {"role": "system", "content": system},
#                 {"role": "user", "content": user},
#             ],
#         )
#         content = resp.choices[0].message.content
#         try:
#             data = json.loads(content)
#         except Exception:
#             # Basic fallback: split lines
#             data = {"lines": [l.strip("- ") for l in content.splitlines() if l.strip()]}
#         data.setdefault("lines", [])
#         return data

