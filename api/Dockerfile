FROM python:3.10-slim

WORKDIR /app
COPY ../requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Add wait-for script from ops
COPY ../ops/wait-for.sh /wait-for.sh

COPY . /app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host","0.0.0.0","--port","8000"]
