FROM python:3.9-slim
WORKDIR /app
COPY . /app

RUN pip install spacy
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download ja_core_news_sm

RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]