# Thesis: Number-oriented text steganography for Dutch financial articles
This repository contains a Python-based implementation for embedding secret messages in text using a steganographic approach. The project focuses on comparing two methods: modifying existing financial articles and generating fake financial articles using ChatGPT.

## Features

- **Encoding**: Convert secret messages into octal numbers, calculate (M, N) pairs, and embed them into financial articles.
- **AI Integration**: Generate economic articles with ChatGPT, ensuring logical placement of numbers.
- **Metrics Evaluation**: Assess performance based on perplexity and payload capacity.

## Installation

### Prerequisites
- Python 3.8 or higher
- API key from OpenAI: https://openai.com/api/
- Dataset of NOS articles: https://www.kaggle.com/datasets/maxscheijen/dutch-news-articles

## Usage

### Preprocessing
Use `preprocessing.py` to prepare financial articles for modification. This script extracts the number of digits in each article to determine suitability for embedding secret messages.

```python
from preprocessing import preprocess
articles_dict = preprocess("dutch-news-articles.csv")
```

### Encoding
Encode a secret message into an article using `encoding.py`:

```python
from encoding import encode
original_article, original_perplexity, modified_article, modified_perplexity, payload_capacity = encode("Geluk")
```

### AI-Generated Articles
Generate and validate economic articles using AI models with `encode_with_ai.py`:

```python
from encode_with_ai import ai_encoded
ai_article, attempts = ai_encoded("Geluk", "gpt-4o", 0.3)
```

### Performance Evaluation
Compare the performance of the two methods using `results.py`:

```python
from results import performance_on_input
performance_on_input()
```

## File Structure

- `preprocessing.py`: Prepares financial articles for steganographic embedding.
- `encoding.py`: Core logic for encoding and modifying existing articles.
- `encode_with_ai.py`: Generates articles using AI models and validates the results.
- `results.py`: Compares the performance of the corpus-based and AI-based methods.


For more details, refer to the documentation in each script file.
