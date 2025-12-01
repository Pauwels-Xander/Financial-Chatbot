.PHONY: setup start backend frontend env

# 1. Create .env if missing
env:
	@if [ ! -f .env ]; then \
		echo "Creating .env from .env.example..."; \
		cp .env.example .env; \
	else \
		echo ".env already exists."; \
	fi

# 2. Install all dependencies (single top-level requirements.txt)
setup: env
	@echo "Installing project dependencies..."
	pip install -r requirements.txt
	@echo "Setup complete!"

# 3. Run backend API (FastAPI)
backend:
	cd backend && uvicorn main:app --reload

# 4. Run frontend (Streamlit)
frontend:
	cd frontend && streamlit run app.py

# 5. Start both backend and frontend
start:
	@echo "Starting backend and frontend..."
	cd backend && uvicorn main:app --reload & \
	cd frontend && streamlit run app.py
