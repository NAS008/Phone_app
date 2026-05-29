FROM python:3.11-slim
WORKDIR /app/backend
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY shared/ /app/shared/
COPY backend/ .
CMD ["python", "app_phone.py"]
