#!/bin/bash
# Docker Build and Run Script for Financial AI Platform

set -e  # Exit on error

echo "🐳 Financial AI Platform - Docker Setup"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

print_status "Docker is installed"

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_warning "Docker Compose not found. Using 'docker compose' instead."
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Parse command line arguments
COMMAND=${1:-"build"}

case $COMMAND in
    build)
        echo ""
        echo "📦 Building Docker image..."
        docker build -t financial-ai:latest .
        print_status "Docker image built successfully"
        ;;
        
    run)
        echo ""
        echo "🚀 Starting Financial AI Dashboard..."
        docker run -d \
            --name financial-ai-dashboard \
            -p 8501:8501 \
            -v "$(pwd)/data:/app/data" \
            -v "$(pwd)/logs:/app/logs" \
            -v "$(pwd)/config:/app/config" \
            financial-ai:latest
        
        print_status "Dashboard started on http://localhost:8501"
        echo ""
        echo "📊 To view logs: docker logs -f financial-ai-dashboard"
        echo "🛑 To stop: docker stop financial-ai-dashboard"
        ;;
        
    up)
        echo ""
        echo "🚀 Starting services with Docker Compose..."
        $DOCKER_COMPOSE up -d
        print_status "Services started"
        echo ""
        echo "📊 Main Dashboard: http://localhost:8501"
        echo "📈 YoY Analysis: http://localhost:8502 (if enabled)"
        echo ""
        echo "📋 To view logs: $DOCKER_COMPOSE logs -f"
        echo "🛑 To stop: $DOCKER_COMPOSE down"
        ;;
        
    yoy)
        echo ""
        echo "🚀 Starting with YoY Analysis service..."
        $DOCKER_COMPOSE --profile yoy up -d
        print_status "Services started with YoY Analysis"
        echo ""
        echo "📊 Main Dashboard: http://localhost:8501"
        echo "📈 YoY Analysis: http://localhost:8502"
        ;;
        
    stop)
        echo ""
        echo "🛑 Stopping services..."
        $DOCKER_COMPOSE down
        docker stop financial-ai-dashboard 2>/dev/null || true
        docker rm financial-ai-dashboard 2>/dev/null || true
        print_status "Services stopped"
        ;;
        
    logs)
        echo ""
        echo "📋 Showing logs..."
        if [ -n "$(docker ps -q -f name=financial-ai-dashboard)" ]; then
            docker logs -f financial-ai-dashboard
        else
            $DOCKER_COMPOSE logs -f
        fi
        ;;
        
    clean)
        echo ""
        echo "🧹 Cleaning up Docker resources..."
        $DOCKER_COMPOSE down -v
        docker stop financial-ai-dashboard 2>/dev/null || true
        docker rm financial-ai-dashboard 2>/dev/null || true
        docker rmi financial-ai:latest 2>/dev/null || true
        print_status "Cleanup complete"
        ;;
        
    shell)
        echo ""
        echo "🐚 Opening shell in container..."
        docker exec -it financial-ai-dashboard /bin/bash
        ;;
        
    rebuild)
        echo ""
        echo "🔄 Rebuilding and restarting..."
        $DOCKER_COMPOSE down
        docker build --no-cache -t financial-ai:latest .
        $DOCKER_COMPOSE up -d
        print_status "Rebuild complete"
        ;;
        
    *)
        echo "Usage: $0 {build|run|up|yoy|stop|logs|clean|shell|rebuild}"
        echo ""
        echo "Commands:"
        echo "  build    - Build Docker image"
        echo "  run      - Run dashboard with docker run"
        echo "  up       - Start services with docker-compose"
        echo "  yoy      - Start with YoY Analysis service"
        echo "  stop     - Stop all services"
        echo "  logs     - Show container logs"
        echo "  clean    - Remove all containers and images"
        echo "  shell    - Open shell in running container"
        echo "  rebuild  - Rebuild and restart services"
        exit 1
        ;;
esac

echo ""
