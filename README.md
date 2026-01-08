
# Advanced Customer Support Chatbot

## Overview
This is a **Customer Support Chatbot** built in Python using **TF-IDF and cosine similarity** for semantic matching.  
It is designed to assist users with common customer support queries such as:

- Order cancellations
- Order tracking
- Refunds and returns
- Account management
- Payment issues
- Delivery information

The chatbot uses a publicly available **customer support training dataset** from HuggingFace to understand queries and provide accurate responses.

---

## Features

- **TF-IDF Vectorization:** Converts user queries and support instructions into numeric vectors.
- **Cosine Similarity Matching:** Finds the most relevant responses based on semantic similarity.
- **Interactive Console Chat:** Chat with the bot directly in the terminal.
- **Category & Intent Detection:** Shows the category and intent of each response (for debugging or analysis).
- **Fallback Responses:** Handles unknown queries gracefully with helpful suggestions.

---

## Dataset

The chatbot uses the **Bitext Customer Support LLM Chatbot Training Dataset**:

- HuggingFace Dataset URL: [bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- Contains:
  - `instruction`: Sample user queries
  - `response`: Appropriate responses
  - `intent`: Intent label of the query
  - `category`: Broad category for the query

---

## Requirements

- Python 3.9+
- Libraries:

```bash
pip install pandas numpy scikit-learn datasets


---

##  Features

- AI-based responses to customer queries
- Easy integration with web interface
- API key stored securely using `.env` file
- Professional structure for deployment
- Prevents secrets from being exposed on GitHub

---



---

##  Setup Instructions

### 1️⃣ Clone Repository
```bash
git clone https://github.com/abc085455-byte/customer-support-chatbot.git
cd customer-support-chatbot
````

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### 3️⃣ Activate Virtual Environment

* **Windows:**

```bash
venv\Scripts\activate
```

* **Linux / MacOS:**

```bash
source venv/bin/activate
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 5️⃣ Setup `.env` File

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-XXXXXXXXXXXX
```

> Replace `sk-XXXXXXXXXXXX` with your OpenAI API key.
> Make sure `.env` is listed in `.gitignore` (it is by default).

### 6️⃣ Run the Application

```bash
python app.py
```

* Open browser at `http://127.0.0.1:5000`
* Start chatting with the AI customer support bot.

---

