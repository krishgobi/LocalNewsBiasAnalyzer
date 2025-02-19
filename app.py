from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Create FastAPI app
app = FastAPI()

# Enable CORS (Allows frontend to communicate with backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Predefined keywords for bias classification
political_keywords = {
    "Republican": ["conservative", "right-wing", "GOP", "Trump", "MAGA"],
    "Democrat": ["liberal", "left-wing", "Biden", "progressive", "Democratic"],
    "Independent": ["moderate", "centrist", "independent"],
    "Libertarian": ["libertarian", "free market", "small government"],
    "Newsmaker": ["politician", "government official"]
}

ideological_keywords = {
    "Organization": ["think tank", "policy institute", "advocacy group"],
    "Journalist": ["reporter", "correspondent", "news agency"],
    "Activist": ["protest", "activism", "social justice", "rights movement"],
    "Columnist": ["opinion piece", "editorial", "op-ed"]
}

# Convert dictionary to keyword list
political_labels, political_words = zip(*[(k, " ".join(v)) for k, v in political_keywords.items()])
ideological_labels, ideological_words = zip(*[(k, " ".join(v)) for k, v in ideological_keywords.items()])

# TF-IDF vectorizer for text similarity
vectorizer = TfidfVectorizer()
political_matrix = vectorizer.fit_transform(political_words)
ideological_matrix = vectorizer.fit_transform(ideological_words)

class NewsInput(BaseModel):
    text: str

def classify_news(text):
    """Classify text as political or ideological bias based on keyword similarity."""
    text_vector = vectorizer.transform([text])

    # Compute cosine similarity
    political_scores = cosine_similarity(text_vector, political_matrix).flatten()
    ideological_scores = cosine_similarity(text_vector, ideological_matrix).flatten()

    # Get best matches
    political_match = political_labels[political_scores.argmax()]
    ideological_match = ideological_labels[ideological_scores.argmax()]

    # Determine final classification
    if max(political_scores) > max(ideological_scores):
        return {"bias": "Political", "category": political_match}
    else:
        return {"bias": "Ideological", "category": ideological_match}

@app.get("/", response_class=HTMLResponse)
async def home():
    return open("static/index.html").read()

@app.post("/predict")
async def predict_bias(news: NewsInput):
    result = classify_news(news.text)
    return {"classification": result}
