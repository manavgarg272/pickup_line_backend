from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class ImageDescription(BaseModel):
    description: str = ""
    attributes: List[str] = Field(default_factory=list)
    vibes: List[str] = Field(default_factory=list)


class GenerateFromFeaturesRequest(BaseModel):
    # Allow fields starting with "model_" without warnings
    model_config = ConfigDict(protected_namespaces=())
    features: ImageDescription
    dataset_examples: Optional[List[str]] = None
    count: int = 5
    tone: Optional[str] = Field(default=None, description="e.g., witty, wholesome, bold, poetic")
    model_text: Optional[str] = Field(default=None, description="Override text model id (e.g., a fine-tuned model)")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Sampling temperature for generation")


class GenerateFromFeaturesResponse(BaseModel):
    lines: List[str]


class GenerateFromImageResponse(BaseModel):
    features: ImageDescription
    lines: List[str]


class HealthResponse(BaseModel):
    # Allow fields starting with "model_" without warnings
    model_config = ConfigDict(protected_namespaces=())
    status: str = "ok"
    version: str = "0.1.0"
    model_text: Optional[str] = None
    model_vision: Optional[str] = None
    error: Optional[str] = None


class GraphGenerateRequest(BaseModel):
    # Allow fields starting with "model_" without warnings
    model_config = ConfigDict(protected_namespaces=())
    features: ImageDescription
    model_text: Optional[str] = Field(default=None, description="Override text model id for graph")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)


class GraphGenerateResponse(BaseModel):
    outputs: Dict[str, str] = Field(description="One line output from each node: playful, witty, spicy, sweet")
    ratings: Dict[str, int] = Field(description="Rating 1-10 from the rater for each node label")
    best_label: str
    best_line: str


class RazorpayCreateOrderRequest(BaseModel):
    amount: int = Field(description="Amount in the smallest currency unit (e.g., paise for INR)")
    currency: str = Field(default="INR", description="ISO currency code, e.g., INR")
    receipt: Optional[str] = Field(default=None, description="Receipt identifier for the order")
    notes: Optional[Dict[str, Any]] = Field(default=None, description="Optional notes to attach to the order")


class RazorpayOrderResponse(BaseModel):
    id: str
    amount: int
    currency: str
    status: Optional[str] = None
    receipt: Optional[str] = None
    created_at: Optional[int] = None
    amount_paid: Optional[int] = None
    amount_due: Optional[int] = None
    notes: Dict[str, Any] = Field(default_factory=dict)
    key_id: str = Field(description="Public Razorpay key id to be used on frontend for checkout")
