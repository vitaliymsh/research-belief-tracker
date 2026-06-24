.PHONY: setup run clean

# Setup python environment and install dependencies
setup:
	@echo "=== Checking Python Environment ==="
	@if [ ! -d ".venv" ]; then \
		echo "Creating virtual environment..."; \
		uv venv --python 3.8.20; \
	else \
		echo "Virtual environment (.venv) already exists. Skipping."; \
	fi
	@echo "\n=== Syncing Dependencies from requirements.txt ==="
	uv pip install -r requirements.txt
	@echo "\n=== Setup verification complete! Use 'make run' to launch. ==="

# Run the FastAPI server
run:
	@echo "Killing any stale python, uv, or app.py processes..."
	-pkill -9 -u vmashkov -f "python|uv run|app.py" 2>/dev/null || true
	@echo "Killing any process on port 8000..."
	-echo vmashkov | sudo -S fuser -k 8000/tcp 2>/dev/null || true
	@echo "Restarting camera daemon..."
	@echo vmashkov | sudo -S systemctl restart nvargus-daemon
	GST_DEBUG=3 uv run python src/app.py

# Remove the virtual environment and cloned tracking repository
clean:
	rm -rf .venv
	rm -rf belief_tracking
	rm -rf belief-tracking
