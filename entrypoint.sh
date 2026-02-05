#!/bin/bash
# AI-HR-Agent Container Entrypoint Script
# Handles initialization and startup of the sandboxed environment

set -e

# ========================================
# Color codes for output
# ========================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ========================================
# Logging functions
# ========================================
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ========================================
# Startup Banner
# ========================================
echo ""
echo "=========================================="
echo "   AI-HR-Agent Docker Sandbox"
echo "=========================================="
echo ""

# ========================================
# Environment Validation
# ========================================
log_info "Validating environment..."

# Check for required API keys
if [ -z "$OPENAI_API_KEY" ]; then
    log_warning "OPENAI_API_KEY is not set. The agent will not function properly."
    log_warning "Set it in your .env file or pass via -e OPENAI_API_KEY=your_key"
else
    log_success "OPENAI_API_KEY is configured"
fi

if [ -z "$TAVILY_API_KEY" ]; then
    log_warning "TAVILY_API_KEY is not set. Market research will use mock data."
else
    log_success "TAVILY_API_KEY is configured"
fi

# ========================================
# Directory Setup
# ========================================
log_info "Setting up directories..."

# Create output directory if it doesn't exist
mkdir -p /app/output
mkdir -p /app/notebooks
mkdir -p /app/.cache

log_success "Directories ready"

# ========================================
# Python Environment Check
# ========================================
log_info "Checking Python environment..."

python --version
log_info "Installed packages:"
pip list --format=freeze | grep -E "(langchain|langgraph|openai|gradio|spacy)" || true

# Verify spaCy model is available
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy model: en_core_web_sm loaded')" 2>/dev/null || {
    log_warning "spaCy model not found. Downloading..."
    python -m spacy download en_core_web_sm
}

log_success "Python environment ready"

# ========================================
# Network Check (optional)
# ========================================
log_info "Checking network connectivity..."

if curl -s --max-time 5 https://api.openai.com > /dev/null 2>&1; then
    log_success "OpenAI API reachable"
else
    log_warning "Cannot reach OpenAI API. Check network settings."
fi

# ========================================
# Display Configuration
# ========================================
echo ""
echo "=========================================="
echo "   Configuration Summary"
echo "=========================================="
echo "  Gradio Host: ${GRADIO_SERVER_NAME:-0.0.0.0}"
echo "  Gradio Port: ${GRADIO_SERVER_PORT:-7860}"
echo "  OpenAI Model: ${OPENAI_MODEL:-default}"
echo "  Debug Mode: ${DEBUG_MODE:-false}"
echo "=========================================="
echo ""

# ========================================
# Execute Command
# ========================================
log_info "Starting application..."
echo ""

# Execute the passed command (or default CMD)
exec "$@"
