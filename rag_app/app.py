from flask import Flask, request, jsonify, render_template
from rag_pipeline import build_rag_chain

app = Flask(__name__)
rag_chain = build_rag_chain()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        response = rag_chain.invoke(question)
        return jsonify({"question": question, "answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
