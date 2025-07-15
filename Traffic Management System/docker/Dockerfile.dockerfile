FROM python:3.9-slim

WORKDIR /app

COPY src/api/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY src/api/ .

CMD ["python", "app.py"]