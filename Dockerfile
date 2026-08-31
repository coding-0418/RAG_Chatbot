FROM python:3.12-slim

WORKDIR /app

COPY mf_faq_chatbot/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY mf_faq_chatbot/ ./mf_faq_chatbot

WORKDIR /app/mf_faq_chatbot

ENV PYTHONUNBUFFERED=1
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
