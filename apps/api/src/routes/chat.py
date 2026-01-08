"""
Chat route handler.

This module handles the /chat endpoint which processes tutoring requests
and returns appropriate responses based on the selected mode and policy.
"""

from fastapi import APIRouter, HTTPException

from ..models import get_model_adapter
from ..policy import TutorPolicy
from ..schemas import ChatRequest, ChatResponse, DebugInfo

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a tutoring chat request.

    This endpoint:
    1. Checks if the request is on-topic (math-related)
    2. Applies appropriate tutoring policy based on mode
    3. Returns a helpful response without revealing answers (if configured)
    """
    policy = TutorPolicy()

    # Check for off-topic content
    if policy.is_off_topic(request.question):
        return ChatResponse(
            response=policy.get_refusal_message(),
            refusal=True,
            debug=DebugInfo(selected_policy="off_topic_refusal"),
        )

    # Check if student is asking for direct answer when not allowed
    if request.dont_reveal_answer and policy.is_asking_for_answer(request.question):
        return ChatResponse(
            response=policy.get_no_reveal_message(request.mode),
            refusal=False,
            debug=DebugInfo(selected_policy="no_reveal_redirect"),
        )

    # Get the model adapter and generate response
    adapter = get_model_adapter()

    try:
        response_text, policy_used = await adapter.generate_response(
            question=request.question,
            attempt=request.attempt,
            mode=request.mode,
            grade=request.grade,
            dont_reveal_answer=request.dont_reveal_answer,
            topic_tags=request.topic_tags,
        )

        return ChatResponse(
            response=response_text,
            refusal=False,
            debug=DebugInfo(selected_policy=policy_used),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

