from flask import Flask, render_template, request
from chatbot_model import predict_intent

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_input = request.json.get("message")
    response = predict_intent(user_input)
    return response

# ⬇️ADD THIS
if __name__ == "__main__":
    app.run(debug=True)
