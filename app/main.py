"""
Application entry point for the Real Estate Conversation BI API.

This module initializes the FastAPI application, configures global
middleware such as CORS, sets up shared application state, and
registers all API route modules.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import chat

# ---------------------------------------------------------
# Create FastAPI application instance
# ---------------------------------------------------------

app = FastAPI(title="Real Estate Converstaion BI API")


# ---------------------------------------------------------
# Configure CORS middleware
# Allows cross-origin requests for frontend integration
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.user_chats = {}
# user id key and active chat id
app.state.chat_history = {}
# chat id and respective chat_history
app.state.last_sql_queries = {}
# # last chat id and sql query 

# ---------------------------------------------------------
# Register API routers
# ---------------------------------------------------------
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])

