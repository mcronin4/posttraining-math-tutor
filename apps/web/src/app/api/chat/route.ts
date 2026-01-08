import { NextRequest, NextResponse } from "next/server";
import type { ChatRequest, ChatResponse } from "@/types";

/**
 * API Route: POST /api/chat
 *
 * Forwards chat requests to the FastAPI backend.
 */

const API_URL = process.env.API_URL || "http://localhost:8000";

export async function POST(request: NextRequest) {
  try {
    const body: ChatRequest = await request.json();

    // Forward request to FastAPI backend
    const response = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Backend error:", errorText);
      return NextResponse.json(
        { error: "Failed to get response from tutor" },
        { status: response.status }
      );
    }

    const data: ChatResponse = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("API route error:", error);

    // If backend is unavailable, return a friendly error
    if (error instanceof TypeError && error.message.includes("fetch")) {
      return NextResponse.json(
        {
          response:
            "The tutor service is not available right now. Make sure the API server is running on port 8000.",
          refusal: false,
          debug: { selected_policy: "error_fallback" },
        },
        { status: 200 }
      );
    }

    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}

