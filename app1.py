from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# FAQ data
questions = [
    "What is AI?",
    "What is machine learning?",
    "What is Python?",
    "How are you?",
    "What is deep learning?"
]

answers = [
    "AI is Artificial Intelligence.",
    "Machine learning is a subset of AI.",
    "Python is a programming language.",
    "I am fine, thank you!",
    "Deep learning is a part of machine learning using neural networks."
]

# Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    user_input = ""

    if request.method == "POST":
        user_input = request.form["message"]

        # Convert user input into vector
        user_vec = vectorizer.transform([user_input])

        # Compute similarity
        similarity = cosine_similarity(user_vec, X)

        # Get best match index
        index_max = similarity.argmax()

        # Smart response check
        if similarity[0][index_max] < 0.3:
            response = "Sorry, I don't understand your question."
        else:
            response = answers[index_max]

    return render_template("index.html", response=response, user_input=user_input)

if __name__ == "__main__":
    app.run(debug=True)
