from typing import Dict, Optional, TypedDict
from typing_extensions import Annotated
from operator import or_
import os
import base64

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


class GraphState(TypedDict, total=False):
    features: Dict
    # Allow concurrent updates from multiple branches by merging dicts
    outputs: Annotated[Dict[str, str], or_]
    ratings: Annotated[Dict[str, int], or_]
    best_label: str
    best_line: str
    # Track how many times each generator has produced a line
    attempts: Annotated[Dict[str, int], or_]
    # Optional inputs for vision description
    image_bytes: bytes
    mime_type: str


def _build_llm(model: Optional[str], temperature: Optional[float]) -> ChatOpenAI:
    # Prefer explicit model arg; otherwise read from env, then fallback to a safe default
    chosen_model = model or os.getenv("OPENAI_TEXT_MODEL") or "gpt-4o-mini"
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    organization = os.getenv("OPENAI_ORG") or os.getenv("OPENAI_ORGANIZATION")

    kwargs = {
        "model": chosen_model,
        "temperature": temperature if temperature is not None else 0.8,
    }
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    if organization:
        kwargs["organization"] = organization

    return ChatOpenAI(**kwargs)


def _to_data_url(image_bytes: bytes, mime_type: Optional[str] = None) -> str:
    if not mime_type:
        mime_type = "image/jpeg"
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def _describe_node(model: Optional[str], temperature: Optional[float]):
    """
    Vision description node. If state has non-empty features, it returns state unchanged.
    Else, if image_bytes is present, it calls a vision-capable model to extract features
    with keys: description (str), attributes (List[str]), vibes (List[str]).
    """
    vision_model = os.getenv("OPENAI_VISION_MODEL") or model or "gpt-4o-mini"
    llm = ChatOpenAI(
        model=vision_model,
        temperature=temperature if temperature is not None else 0.5,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        organization=os.getenv("OPENAI_ORG") or os.getenv("OPENAI_ORGANIZATION"),
    )

    system = (
        "You analyze dating profile photos and extract concise, respectful details suitable for crafting pickup lines. "
        "Avoid sensitive inferences. Respond in strict JSON."
    )
    user_text = (
        "Analyze the image and output strict JSON with keys: "
        "description (string), attributes (string[]), vibes (string[]). "
        "attributes are short keywords like 'smiling','beach','guitar','dog','sunset','glasses'. "
        "vibes are 1-3 words like 'adventurous','artsy','bookish'."
    )

    def node(state: GraphState) -> GraphState:
        # If features already provided or no image, pass-through.
        features = state.get("features") or {}
        if features:
            return {"features": features}

        image_bytes = state.get("image_bytes")
        if not image_bytes:
            # Nothing to do; leave features empty
            return {"features": {}}

        data_url = _to_data_url(image_bytes, state.get("mime_type"))
        # Prepare multimodal message
        messages = [
            ("system", system),
            ("user", [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]),
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        resp = (prompt | llm).invoke({})
        content = getattr(resp, "content", resp)

        # Try parse JSON; fallback to safe structure
        import json
        try:
            data = json.loads(content)
        except Exception:
            data = {"description": str(content or ""), "attributes": [], "vibes": []}
        data.setdefault("description", "")
        data.setdefault("attributes", [])
        data.setdefault("vibes", [])
        return {"features": data}

    return node


def _make_gen_node(label: str, style_instruction: str, model: Optional[str], temperature: Optional[float]):
    llm = _build_llm(model, temperature)
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You write a single, short pickup line in the following style: {style_instruction}.\n"
                    "Keep it respectful and flirty, avoid creepy or offensive content.\n"
                    "Please use simple english\n"
                    "You can add humor in pickup line.\n"
                    "Only output the pickup line text, no quotes, no JSON."),
        ("user", "Features: {features}")
    ])
    chain = prompt | llm | StrOutputParser()

    def node(state: GraphState) -> GraphState:
        features = state.get("features", {})
        line = chain.invoke({"features": features})
        outputs = dict(state.get("outputs", {}))
        outputs[label] = (line or "").strip()
        attempts = dict(state.get("attempts", {}))
        attempts[label] = attempts.get(label, 0) + 1
        return {"outputs": outputs, "attempts": attempts}

    return node


def _rater_node(model: Optional[str], temperature: Optional[float]):
    llm = _build_llm(model, temperature)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a woman reading dating app openers.\n"
                    "Rate each line from 1-10 on attractiveness, charm, and respect.\n"
                    "Return a JSON object with keys 'ratings' (map of label to number), 'best_label' (string), and 'best_line' (string)."),
        ("user", "Lines to rate (JSON): {outputs}")
    ])
    chain = prompt | llm

    def node(state: GraphState) -> GraphState:
        outputs = state.get("outputs", {})
        resp = chain.invoke({"outputs": outputs})
        # Try to parse model JSON response; fallback to simple heuristic
        import json
        ratings, best_label, best_line = {}, "", ""
        try:
            data = json.loads(getattr(resp, "content", resp))
            ratings = {k: int(v) for k, v in data.get("ratings", {}).items()}
            best_label = data.get("best_label") or ""
            best_line = data.get("best_line") or (outputs.get(best_label) if best_label in outputs else "")
        except Exception:
            # Fallback: choose longest non-empty line
            non_empty = [(k, v) for k, v in outputs.items() if v]
            if non_empty:
                best_label, best_line = max(non_empty, key=lambda kv: len(kv[1]))
            ratings = {k: 7 for k in outputs.keys()}  # neutral default
        return {
            "ratings": ratings,
            "best_label": best_label,
            "best_line": best_line,
        }

    return node


def build_pickup_graph(model: Optional[str] = None, temperature: Optional[float] = None):
    g = StateGraph(GraphState)
    # Nodes
    g.add_node("describe", _describe_node(model, temperature))
    g.add_node("playful", _make_gen_node("playful", "playful, cheeky, light banter", model, temperature))
    g.add_node("witty", _make_gen_node("witty", "clever, wordplay, subtle humor", model, temperature))
    g.add_node("spicy", _make_gen_node("spicy", "bold, flirty, a tiny bit spicy but respectful", model, temperature))
    g.add_node("sweet", _make_gen_node("sweet", "wholesome, kind, cute", model, temperature))
    g.add_node("rate", _rater_node(model, temperature))

    # Start with describe, then fan out to the four generators
    g.add_edge(START, "describe")
    g.add_edge("describe", "playful")
    g.add_edge("describe", "witty")
    g.add_edge("describe", "spicy")
    g.add_edge("describe", "sweet")

    # Join at rater
    g.add_edge("playful", "rate")
    g.add_edge("witty", "rate")
    g.add_edge("spicy", "rate")
    g.add_edge("sweet", "rate")

    # Conditional loop: if a targeted label is below threshold and hasn't exceeded attempts, retry it
    THRESHOLD = 8
    MAX_ATTEMPTS_PER_LABEL = 2  # initial + 1 retry

    def retry_condition(state: GraphState) -> str:
        ratings = state.get("ratings", {}) or {}
        attempts = state.get("attempts", {}) or {}
        candidates = []
        for label in ("playful", "witty", "spicy"):
            score = 0
            try:
                score = int(ratings.get(label, 0))
            except Exception:
                score = 0
            if score < THRESHOLD and attempts.get(label, 0) < MAX_ATTEMPTS_PER_LABEL:
                candidates.append((score, label))
        if not candidates:
            return "done"
        # Pick the lowest scoring candidate to retry next
        candidates.sort(key=lambda x: x[0])
        _, label = candidates[0]
        return f"retry_{label}"

    g.add_conditional_edges(
        "rate",
        retry_condition,
        {
            "retry_playful": "playful",
            "retry_witty": "witty",
            "retry_spicy": "spicy",
            "done": END,
        },
    )

    # Note: final transition to END handled by conditional above when "done"

    return g.compile()
