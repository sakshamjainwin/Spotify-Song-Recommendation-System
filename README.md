# Spotify Song Recommender System

## Overview
This project develops a **Spotify Song Recommender System** that personalizes song recommendations based on **audio features, sentiment analysis, lyrics embeddings, and user preferences**. The system leverages **Spotify metadata, song lyrics, sentiment scores, and machine learning techniques** to optimize playlist consistency and similarity scores.

## Features
- **Audio Feature Similarity**: Uses Spotify metadata like **danceability, energy, key, loudness, mode, speechiness, etc.**
- **Lyrics Embedding Similarity**: Computes **semantic similarity** between song lyrics using **BERT embeddings**.
- **Sentiment Analysis**: Analyzes the **emotional tone** of lyrics using an **emotion classifier (joy, sadness, anger, fear, etc.).**
- **Popularity Normalization**: Adjusts similarity based on **track popularity** to balance mainstream and niche song recommendations.
- **Hyperparameter Optimization**: Tunes weights for different similarity measures to achieve the best recommendation accuracy.
- **Threshold-Based Evaluation**: Computes accuracy by adjusting the **similarity threshold** and evaluating the **playlist coherence**.

## Files and Structure
### **1. Dataset and User Data**
- **`User data (Testing data).json`**: Contains user playlist data for **testing the recommender system**.
- **`Spotify Song Ids (Training).json`**: Stores **Spotify song IDs** used for training and recommendation.
- **`Processed_Songs_Data.csv`**: Final **preprocessed dataset** with all relevant song attributes and embeddings.
- **`Song Recommender System Documentation.pdf`**: **Detailed methodology** and explanation of the recommender system.

### **2. Notebooks**
- **`EDA.ipynb`**: Exploratory Data Analysis (**EDA**) of **Spotify metadata, lyrics, and embeddings**.
- **`Lyrics embedding + sentiment analysis.ipynb`**: Generates **lyrics embeddings**, **sentiment analysis**, and **topic modeling**.
- **`spotify scraping.ipynb`**: Code for **scraping song metadata** from Spotify.
- **`lyrics scraping.ipynb`**: **Scrapes song lyrics** for analysis.
- **`final similarity scores.ipynb`**: Computes **song similarity scores**, **hyperparameter tuning**, and final recommendations.

## Methodology
### **1. Data Collection & Preprocessing**
- **Spotify Metadata**: Extracted from Spotify API, including **popularity, danceability, energy, tempo, mode, etc.**
- **Lyrics Scraping**: Lyrics data collected using **web scraping techniques**.
- **Lyrics Preprocessing**:
  - Removed metadata and annotations (**e.g., verse markers, translations, and unnecessary symbols**).
  - Detected song language and filtered non-English content where necessary.

### **2. Feature Engineering**
- **Lyrics Embeddings**:
  - Generated using **BERT-based models** (`bert-base-multilingual-cased`).
  - Converted lyrics into **768-dimensional embeddings**.
- **Sentiment Analysis**:
  - Used `j-hartmann/emotion-english-distilroberta-base` to predict **emotion scores** (joy, sadness, anger, etc.).
- **Topic Modeling**:
  - Identified dominant themes in song lyrics to **group similar songs**.

### **3. Similarity Computation**
The system calculates **song similarity** based on:
- **Popularity Difference (`absolute_difference_similarity`)**
- **Audio Feature Similarity (`combined_audio_similarity`)**
- **Lyrics Language Matching (`lyrics_language_similarity`)**
- **Sentiment Similarity (`sentiment_similarity_combined`)**
- **Lyrics Semantic Similarity (`lyrics_embedding_similarity`)**

### **4. Hyperparameter Tuning & Evaluation**
- **Hyperparameters Optimized:**
  - `weight_popularity`, `weight_audio`, `weight_language`, `weight_sentiment`, `weight_lyrics_embedding`.
  - `alpha_audio` and `alpha_sentiment` (balance between cosine similarity and Euclidean distance).
- **Evaluation:**
  - Compared the **last song** in a playlist with all **previous songs** to measure playlist coherence.
  - Tested various **similarity thresholds (0 to 1.0)** and computed **accuracy scores**.
  - Visualized results using a **Threshold vs. Accuracy plot**.

## Results & Performance
- The recommender system **successfully predicts** user song preferences by **matching songs with similar attributes**.
- The **optimized similarity function** ensures **diversity** while maintaining **playlist consistency**.
- **Hyperparameter tuning improves recommendation accuracy**, balancing **audio, lyrics, and sentiment factors**.

## Setup & Usage
### **Installation**
Ensure you have the required Python libraries installed:
```bash
pip install pandas numpy scikit-learn tqdm transformers langdetect torch matplotlib spotipy
```
### **Running the Code**
1. **Run `spotify scraping.ipynb`** to scrape Spotify metadata.
2. **Run `lyrics scraping.ipynb`** to collect lyrics data.
3. **Execute `Lyrics embedding + sentiment analysis.ipynb`** to generate embeddings and sentiment scores.
4. **Run `final similarity scores.ipynb`** to compute song similarity and evaluate recommendations.
5. **Analyze `EDA.ipynb`** for exploratory insights.

### **Visualization**
- The system generates **plots and tables** showcasing **playlist similarity scores** and **threshold-based accuracy**.

## Future Improvements
- **Enhance topic modeling** to cluster songs with **similar themes**.
- **Improve real-time recommendations** using **neural embeddings and deep learning models**.
- **Integrate with Spotify API** to provide **live personalized recommendations**.

---
### **Contributors & Acknowledgments**
Special thanks to **Spotify API, Hugging Face Transformers, and Scikit-Learn** for enabling the analysis.
