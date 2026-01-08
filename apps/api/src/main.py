"""
Math Tutor API - Main FastAPI Application

This is the entry point for the FastAPI backend. It provides the /chat endpoint
for the tutoring interface and handles request routing.
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import chat
from .models import get_model_adapter

# Load environment variables
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    # Startup: initialize model adapter
    print("🚀 Starting Math Tutor API...")
    adapter = get_model_adapter()
    print(f"📚 Using model adapter: {adapter.__class__.__name__}")
    yield
    # Shutdown
    print("👋 Shutting down Math Tutor API...")


# Create FastAPI application
app = FastAPI(
    title="Math Tutor API",
    description="AI-powered math tutoring assistant for Ontario K-12 curriculum",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)


@app.get("/")
async def root():
    """Root endpoint - API status check."""
    return {
        "name": "Math Tutor API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

