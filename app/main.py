import io
import os
import json
from typing import List, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# from app.openai_client import OpenAIClient
from app.schemas import (
    # ImageDescription,
    # GenerateFromFeaturesRequest,
    # GenerateFromFeaturesResponse,
    # GenerateFromImageResponse,
    # HealthResponse,
    GraphGenerateRequest,
    GraphGenerateResponse,
)
from app.graph import build_pickup_graph

load_dotenv()
# Enable LangSmith tracing if env not already set
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "pickup-line")

app = FastAPI(title="Pickup Line Generator API", version="0.1.0")

# CORS (adjust origins for your frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://065b46f36fdf.ngrok-free.app",
        "https://main.d3jjj51vtinn6u.amplifyapp.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single OpenAI client instance
# try:
#     openai_client = OpenAIClient()
# except Exception as e:
#     # Delay failure to runtime endpoints, but log here
#     print(e)
#     openai_client = None
#     INIT_ERROR = str(e)
# else:
#     INIT_ERROR = None


# @app.get("/health", response_model=HealthResponse)
# async def health() -> HealthResponse:
#     if openai_client is None:
#         return HealthResponse(status="degraded", version=app.version)
#     return HealthResponse(status="ok", version=app.version,  model_vision=openai_client.vision_model)


# @app.post("/v1/describe-image", response_model=ImageDescription)
# async def describe_image(file: UploadFile = File(...)) -> ImageDescription:
#     if openai_client is None:
#         raise HTTPException(status_code=500, detail=f"Initialization error: {INIT_ERROR}")
#     # Validate content type
#     if not (file.content_type and file.content_type.startswith("image/")):
#         raise HTTPException(status_code=400, detail="Please upload an image file.")

#     data = await file.read()
#     if len(data) == 0:
#         raise HTTPException(status_code=400, detail="Empty file uploaded.")

#     features = openai_client.describe_image(data, mime_type=file.content_type)
#     return ImageDescription(**features)


# @app.post("/v1/generate-pickuplines", response_model=GenerateFromFeaturesResponse)
# async def generate_pickup_lines(payload: GenerateFromFeaturesRequest) -> GenerateFromFeaturesResponse:
#     if openai_client is None:
#         raise HTTPException(status_code=500, detail=f"Initialization error: {INIT_ERROR}")
#     result = openai_client.generate_pickup_lines(
#         features=payload.features.model_dump(),
#         dataset_examples=payload.dataset_examples,
#         count=payload.count,
#         tone=payload.tone,
#         temperature=payload.temperature,
#     )
#     return GenerateFromFeaturesResponse(lines=result.get("lines", []))


# @app.post("/v1/generate-pickuplines-from-image", response_model=GenerateFromImageResponse)
# async def generate_pickup_lines_from_image(
#     file: UploadFile = File(...),
#     count: int = 5,
#     tone: Optional[str] = Form(default=None),
#     dataset_examples: Optional[str] = Form(default=None, description="JSON array of strings e.g. [\"line1\", \"line2\"]"),
#     temperature: Optional[float] = Form(default=None, description="Sampling temperature for generation"),
# ) -> GenerateFromImageResponse:
#     if openai_client is None:
#         raise HTTPException(status_code=500, detail=f"Initialization error: {INIT_ERROR}")

#     if not (file.content_type and file.content_type.startswith("image/")):
#         raise HTTPException(status_code=400, detail="Please upload an image file.")

#     data = await file.read()
#     if len(data) == 0:
#         raise HTTPException(status_code=400, detail="Empty file uploaded.")

#     # Parse dataset_examples JSON string if provided
#     examples_list: Optional[List[str]] = None
#     if dataset_examples:
#         try:
#             parsed = json.loads(dataset_examples)
#             if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
#                 examples_list = parsed
#         except Exception:
#             # Ignore malformed examples
#             examples_list = None

#     features = openai_client.describe_image(data, mime_type=file.content_type)
#     gen = openai_client.generate_pickup_lines(
#         features=features,
#         dataset_examples=examples_list,
#         count=count,
#         tone=tone,
#         temperature=temperature,
#     )
#     return GenerateFromImageResponse(features=ImageDescription(**features), lines=gen.get("lines", []))


@app.post("/v1/generate-graph", response_model=GraphGenerateResponse)
async def generate_graph(payload: GraphGenerateRequest) -> GraphGenerateResponse:
    # Choose model/temperature: prefer explicit override; otherwise use a safe base model for LangGraph
    chosen_model ="ft:gpt-3.5-turbo-1106:manav::C8AMBoyU"
    chosen_temp = payload.temperature if payload.temperature is not None else 0.5
    print(chosen_model, chosen_temp)
    app_graph = build_pickup_graph(model=chosen_model, temperature=chosen_temp)
    state_in = {"features": payload.features.model_dump()}

    try:
        result = app_graph.invoke(
            state_in,
            config={
                "tags": ["graph", "from-features"],
                "metadata": {
                    "route": "/v1/generate-graph",
                    "model": chosen_model,
                    "temperature": chosen_temp,
                },
            },
        )
    except Exception as e:
        # Surface error to client for debugging
        raise HTTPException(status_code=500, detail=f"graph_error: {e}")

    outputs = result.get("outputs", {})
    ratings = result.get("ratings", {})
    best_label = result.get("best_label", "")
    best_line = result.get("best_line", outputs.get(best_label, ""))

    return GraphGenerateResponse(
        outputs=outputs,
        ratings=ratings,
        best_label=best_label,
        best_line=best_line,
    )


@app.post("/v1/generate-graph-from-image", response_model=GraphGenerateResponse)
async def generate_graph_from_image(
    file: UploadFile = File(...),
    model_text: Optional[str] = Form(default=None),
    temperature: Optional[float] = Form(default=None, description="Sampling temperature for generation"),
) -> GraphGenerateResponse:
    # Validate content type
    if not (file.content_type and file.content_type.startswith("image/")):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    chosen_model = model_text or os.getenv("OPENAI_TEXT_MODEL") or "gpt-4o-mini"
    chosen_temp = temperature if temperature is not None else 0.5

    app_graph = build_pickup_graph(model=chosen_model, temperature=chosen_temp)
    state_in = {
        "image_bytes": data,
        "mime_type": file.content_type,
        # Allow features passthrough if you later extend with extra form fields
        "features": {},
    }

    try:
        result = app_graph.invoke(
            state_in,
            config={
                "tags": ["graph", "from-image"],
                "metadata": {
                    "route": "/v1/generate-graph-from-image",
                    "model": chosen_model,
                    "temperature": chosen_temp,
                    "mime_type": file.content_type,
                },
            },
        )
    except Exception as e:
        # Surface error to client for debugging
        raise HTTPException(status_code=500, detail=f"graph_error: {e}")

    outputs = result.get("outputs", {})
    ratings = result.get("ratings", {})
    best_label = result.get("best_label", "")
    best_line = result.get("best_line", outputs.get(best_label, ""))

    return GraphGenerateResponse(
        outputs=outputs,
        ratings=ratings,
        best_label=best_label,
        best_line=best_line,
    )
