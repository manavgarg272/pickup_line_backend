# Pickup Line Generator API

FastAPI service that:
- Describes dating profile images using OpenAI vision.
- Generates tailored pickup lines using features and optional style examples.

## Setup

1. Python 3.10+
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_key_here
# Optional: override default models
export OPENAI_VISION_MODEL=gpt-4o-mini
export OPENAI_TEXT_MODEL=gpt-4o-mini
```

## Run
```bash
uvicorn app.main:app --reload --port 8080
```

## Endpoints

### Health
```bash
curl -s http://localhost:8080/health | jq
```

### 1) Describe Image
Multipart upload of an image. Returns JSON features.
```bash
curl -s -X POST "http://localhost:8080/v1/describe-image" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/photo.jpg" | jq
```

Response shape:
```json
{
  "description": "person smiling with a dog at the beach",
  "attributes": ["smiling", "dog", "beach", "sunset"],
  "vibes": ["adventurous", "warm"]
}
```

### 2) Generate Pickup Lines from Features
Send features and optional dataset examples and tone.
```bash
curl -s -X POST "http://localhost:8080/v1/generate-pickuplines" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "description": "person smiling with a dog at the beach",
      "attributes": ["smiling", "dog", "beach", "sunset"],
      "vibes": ["adventurous", "warm"]
    },
    "dataset_examples": [
      "Do you have a map? I keep getting lost in your smile.",
      "Are you a camera? Every time I look at you, I smile."
    ],
    "count": 5,
    "tone": "witty"
  }' | jq
```

### 3) Generate Pickup Lines from Image (one-shot)
Upload image, optional dataset examples, count, tone.
```bash
curl -s -X POST "http://localhost:8080/v1/generate-pickuplines-from-image?count=5&tone=witty" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/photo.jpg" \
  -F 'dataset_examples=["We should probably take our dogs on a date.", "Sunset walks are better with company."]' | jq
```

Note: The `dataset_examples` field in the combined endpoint is an optional JSON array passed in the request body along with multipart; if your client library struggles with that, prefer calling the two-step flow.

## Notes
- Keep content safe and respectful. Avoid sensitive inferences.
- You can change model names via env vars without code changes.
- If you need stricter JSON, the SDK already requests `response_format: json_object`.
