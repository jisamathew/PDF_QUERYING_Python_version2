from flask import Flask, request, jsonify
from transformers import CLIPProcessor, CLIPModel, pipeline
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'
# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def get_text_features(text):
    inputs = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
    query_embedding = clip_model.get_text_features(**inputs).squeeze().detach().numpy().tolist()
    return query_embedding

def answer_question(question, context):
    qa_model = pipeline('question-answering', model='deepset/roberta-base-squad2')
    answer = qa_model(question=question, context=context)
   
    # deepset/tinyroberta-squad2
    return answer['answer']
@app.route('/')
def hello_world():
    return 'Hello, Welcome to our Sample Flask API!'

@app.route('/embed', methods=['POST'])
def embed_text():
    data = request.get_json()
    text = data.get('text', '')
    embedding = get_text_features(text)
    return jsonify(embedding)

@app.route('/qa', methods=['POST'])
def qa():
    data = request.get_json()
    question = data.get('question', '')
    context = data.get('context', '')
    answer = answer_question(question, context)
    return jsonify({'answer': answer})
    
