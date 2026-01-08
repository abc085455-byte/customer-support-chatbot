from flask import Flask, render_template, request, jsonify, session
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re
import os
from datetime import datetime                  
import pickle

app = Flask(__name__)
app.secret_key = 'sk-.............'
# OpenAI Config
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'sk-.......')
USE_OPENAI = True

# Model save/load directory
MODEL_DIR = "saved_model"
os.makedirs(MODEL_DIR, exist_ok=True)

class AdvancedCustomerSupportChatbot:
    def __init__(self):
        self.conversation_history = {}

        # Check if saved model exists
        vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
        vectors_path = os.path.join(MODEL_DIR, "question_vectors.pkl")
        df_path = os.path.join(MODEL_DIR, "dataset_df.pkl")
        kb_path = os.path.join(MODEL_DIR, "knowledge_base.pkl")

        if all(os.path.exists(p) for p in [vectorizer_path, vectors_path, df_path, kb_path]):
            print("🤖 Loading saved model...")
            self.df = pd.read_pickle(df_path)
            with open(vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            with open(vectors_path, "rb") as f:
                self.question_vectors = pickle.load(f)
            with open(kb_path, "rb") as f:
                self.knowledge_base = pickle.load(f)
            print("✅ Saved model loaded successfully!")
        else:
            print("🤖 Loading dataset from HuggingFace...")
            dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
            self.df = pd.DataFrame(dataset['train'])
            print(f"✅ Dataset loaded: {len(self.df)} examples")

            # TF-IDF
            print("🔧 Training TF-IDF vectorizer...")
            self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2), min_df=2)
            self.question_vectors = self.vectorizer.fit_transform(self.df['instruction'])

            # Knowledge base
            self.knowledge_base = self._create_knowledge_base()

            # Save everything
            self.save_model()
            print("✅ Model, vectors & dataset saved for future runs!")

    def save_model(self):
        """Save TF-IDF model, vectors, dataset, knowledge base"""
        with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(MODEL_DIR, "question_vectors.pkl"), "wb") as f:
            pickle.dump(self.question_vectors, f)
        self.df.to_pickle(os.path.join(MODEL_DIR, "dataset_df.pkl"))
        with open(os.path.join(MODEL_DIR, "knowledge_base.pkl"), "wb") as f:
            pickle.dump(self.knowledge_base, f)

    def _create_knowledge_base(self):
        kb = {}
        for intent in self.df['intent'].unique():
            intent_data = self.df[self.df['intent'] == intent]
            kb[intent] = {
                'examples': intent_data['instruction'].tolist()[:5],
                'responses': intent_data['response'].tolist()[:5],
                'category': intent_data['category'].iloc[0]
            }
        return kb

    def clean_text(self, text):
        text = re.sub(r'\{\{Order Number\}\}', 'your order number', text)
        text = re.sub(r'\{\{Online Company Portal Info\}\}', 'our website', text)
        text = re.sub(r'\{\{Customer Support Hours\}\}', 'business hours (9 AM - 6 PM)', text)
        text = re.sub(r'\{\{Customer Support Phone Number\}\}', '1-800-SUPPORT', text)
        text = re.sub(r'\{\{Website URL\}\}', 'www.company.com', text)
        text = re.sub(r'\{\{.*?\}\}', '', text)
        return text.strip()

    def get_context_from_history(self, session_id):
        if session_id in self.conversation_history:
            history = self.conversation_history[session_id]
            return history[-3:] if len(history) > 3 else history
        return []

    def update_history(self, session_id, user_msg, bot_response):
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        self.conversation_history[session_id].append({
            'user': user_msg,
            'bot': bot_response,
            'timestamp': datetime.now().isoformat()
        })
        if len(self.conversation_history[session_id]) > 10:
            self.conversation_history[session_id] = self.conversation_history[session_id][-10:]

    def get_tfidf_response(self, user_query, top_k=3):
        query_vector = self.vectorizer.transform([user_query])
        similarities = cosine_similarity(query_vector, self.question_vectors).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                match = self.df.iloc[idx]
                results.append({
                    'response': self.clean_text(match['response']),
                    'category': match['category'],
                    'intent': match['intent'],
                    'confidence': float(similarities[idx])
                })
        return results[0] if results else None

    def get_response(self, user_query, session_id=None):
        context = self.get_context_from_history(session_id) if session_id else []
        tfidf_result = self.get_tfidf_response(user_query)

        if tfidf_result:
            result = {**tfidf_result, 'enhanced': False, 'method': 'TF-IDF'}
            if session_id:
                self.update_history(session_id, user_query, tfidf_result['response'])
            return result

        default_response = """I apologize, but I'm not sure I understand your question completely. 
I can help you with:
• Order cancellations and modifications
• Tracking your orders
• Refund and return requests
• Account login and management issues
• Payment problems
• Delivery information and options

Could you please rephrase your question or let me know which area you need help with?"""

        return {
            'response': default_response,
            'category': 'GENERAL',
            'intent': 'unknown',
            'confidence': 0.0,
            'enhanced': False,
            'method': 'Fallback'
        }

    def get_stats(self):
        return {
            'total_examples': len(self.df),
            'categories': self.df['category'].nunique(),
            'intents': self.df['intent'].nunique(),
            'category_distribution': self.df['category'].value_counts().to_dict(),
            'intent_distribution': self.df['intent'].value_counts().to_dict()[:10],
            'openai_enabled': USE_OPENAI
        }

# Initialize chatbot
print("="*70)
print("Initializing Advanced Customer Support Chatbot...")
print("="*70)
bot = AdvancedCustomerSupportChatbot()

@app.route('/')
def home():
    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    session_id = session.get('session_id', 'default')
    result = bot.get_response(user_message, session_id)
    return jsonify(result)

@app.route('/stats', methods=['GET'])
def stats():
    return jsonify(bot.get_stats())

@app.route('/history', methods=['GET'])
def history():
    session_id = session.get('session_id', 'default')
    history = bot.get_context_from_history(session_id)
    return jsonify({'history': history})

@app.route('/clear', methods=['POST'])
def clear_history():
    session_id = session.get('session_id', 'default')
    if session_id in bot.conversation_history:
        bot.conversation_history[session_id] = []
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Starting Advanced Flask Server...")
    print("🌐 Open your browser: http://127.0.0.1:5000")
    print(f"🤖 OpenAI Integration: {'✅ Enabled' if USE_OPENAI else '❌ Disabled'}")
    print("="*70 + "\n")
    app.run(debug=True, port=5000, threaded=True)