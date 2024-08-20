# Gen AI: News Research Tool 📈

![Gen AI: News Research Tool Screenshot](path_to_your_image) <!-- Replace with the URL to your screenshot -->

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Overview
**Gen AI: News Research Tool** is a web-based application built using [Streamlit](https://streamlit.io/), designed to help users process news articles and research data using Google Generative AI embeddings and FAISS for vector indexing. This tool can take URLs of news articles, process them, and then allow users to ask relevant questions based on the information extracted from the content.

## Features
- Supports multiple news article URLs for data extraction.
- Embeds content using **Google Generative AI** embeddings.
- Uses **FAISS** (Facebook AI Similarity Search) for efficient text retrieval and indexing.
- Allows users to ask questions and get answers directly based on processed data.
- Displays sources for the answers provided.

## Installation

### Prerequisites
Before you can run this tool, you need to have the following installed:
- Python 3.8 or above
- Streamlit
- FAISS
- Google Generative AI API access (Google Cloud setup)

### Step-by-Step Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/gen-ai-news-research-tool.git
    cd gen-ai-news-research-tool
    ```

2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Install NLTK data:
    ```bash
    python -c "import nltk; nltk.download('all')"
    ```

4. Set up your Google API key:
    - In your project directory, create a `.env` file or directly export it in your terminal:
      ```bash
      export GOOGLE_API_KEY='your-google-api-key'
      ```

## Usage

1. To run the Streamlit app, navigate to the project directory and run the following command:
    ```bash
    streamlit run app.py
    ```

2. Enter the URLs you want to process into the sidebar form. You can input up to 3 URLs at a time.

3. Click **Process URLs** to start extracting and indexing the data.

4. After processing, enter your question in the text box, and the system will provide answers based on the data extracted.

5. The sources of the answers will also be displayed, helping you verify the information.

## Dependencies

The key dependencies for this project are:
- [Streamlit](https://streamlit.io/) for the user interface.
- [LangChain](https://github.com/hwchase17/langchain) for text processing.
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector searches.
- [Google Generative AI](https://cloud.google.com/genai) for embeddings.
- [NLTK](https://www.nltk.org/) for text preprocessing.

You can find all the dependencies listed in the `requirements.txt` file.

## Configuration

Make sure to configure your **Google API key** as an environment variable:
```bash
export GOOGLE_API_KEY='your-google-api-key'
