import io
import os
import json
from typing import List, Optional
from dotenv import load_dotenv
import razorpay

from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# from app.openai_client import OpenAIClient
from app.schemas import (
    GraphGenerateRequest,
    GraphGenerateResponse,
    RazorpayCreateOrderRequest,
    RazorpayOrderResponse,
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
        "https://www.flirtsparks.in",
        "https://flirtsparks.in",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# Razorpay: Create Order
@app.post("/v1/payments/razorpay/create-order", response_model=RazorpayOrderResponse)
async def create_razorpay_order(payload: RazorpayCreateOrderRequest) -> RazorpayOrderResponse:
    key_id = os.getenv("RAZORPAY_KEY_ID")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET")
    if not key_id or not key_secret:
        raise HTTPException(status_code=500, detail="Razorpay credentials are not configured. Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET")

    if payload.amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be greater than 0 (in the smallest currency unit)")

    try:
        client = razorpay.Client(auth=(key_id, key_secret))
        data = {
            "amount": payload.amount,  # amount in paise for INR
            "currency": payload.currency or "INR",
        }
        if payload.receipt:
            data["receipt"] = payload.receipt
        if payload.notes:
            data["notes"] = payload.notes

        order = client.order.create(data=data)
        # Build typed response including public key for frontend checkout
        return RazorpayOrderResponse(
            id=order.get("id"),
            amount=order.get("amount"),
            currency=order.get("currency"),
            status=order.get("status"),
            receipt=order.get("receipt"),
            created_at=order.get("created_at"),
            amount_paid=order.get("amount_paid"),
            amount_due=order.get("amount_due"),
            notes=order.get("notes") or {},
            key_id=key_id,
        )
    except razorpay.errors.BadRequestError as e:
        raise HTTPException(status_code=400, detail=f"Razorpay BadRequest: {e}")
    except razorpay.errors.ServerError as e:
        raise HTTPException(status_code=502, detail=f"Razorpay ServerError: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Razorpay error: {e}")


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
