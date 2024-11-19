# Fake_news_detection_project
A machine learning project for detecting fake news using TF-IDF and classification models like Logistic Regression and Random Forest, featuring an interactive Streamlit app for real-time predictions 


**Description**  
This project leverages machine learning and natural language processing (NLP) techniques to detect and classify news articles as fake or real. By preprocessing text data, extracting features using TF-IDF, and training multiple classifiers, it provides a robust framework for combating misinformation. The project also features an interactive Streamlit app for real-time news classification.

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Features](#features)
4. [Dataset](#dataset)
5. [Technologies Used](#technologies-used)
6. [How to Run](#how-to-run)
7. [Results](#results)
8. [Future Scope](#future-scope)
9. [Contributors](#contributors)
10. [License](#license)

---

## **Introduction**

Fake news has become a significant challenge in the digital era, impacting public opinion and decision-making. This project aims to detect fake news articles using machine learning models trained on textual data. It preprocesses raw text, extracts key features using TF-IDF, and trains models like Logistic Regression, Naive Bayes, and Random Forest to achieve high classification accuracy.

---

## **Project Structure**

```plaintext
Fake_News_Detection_Project/
├── data/
│   ├── Fake.csv                     # Dataset of fake news
│   ├── True.csv                     # Dataset of real news
│
├── notebooks/
│   ├── Fake_News_Detection.ipynb    # Jupyter Notebook for model training
│
├── models/
│   ├── tfidf_vectorizer.pkl         # Saved TF-IDF vectorizer
│   ├── fake_news_model.pkl          # Saved classification model
│
├── app/
│   ├── app.py                       # Streamlit app for testing
│
├── requirements.txt                 # Dependencies for the project
├── README.md                        # Project documentation
└── LICENSE                          # License file
```

---

## **Features**

- **Data Preprocessing**:
  - Removed stopwords, punctuations, and special characters.
  - Tokenized and lemmatized the text for better feature extraction.
- **Feature Extraction**:
  - Applied TF-IDF (Term Frequency-Inverse Document Frequency) to transform text into numerical vectors.
- **Model Training**:
  - Trained Logistic Regression, Random Forest, and Naive Bayes models.
  - Performed hyperparameter tuning for improved performance.
- **Interactive Application**:
  - Developed a Streamlit app for real-time fake news detection, allowing users to test articles.

---

## **Dataset**

- **Source**: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)  
- **Size**: Approximately 40,000 news articles labeled as **Fake** or **Real**.  
- **Key Columns**:
  - `title`: Headline of the news article.
  - `text`: Full text of the news article.
  - `label`: Classification as `fake` or `real`.

---

## **Technologies Used**

- **Programming Language**: Python  
- **Libraries**:
  - NLP: NLTK, Scikit-learn
  - Data Manipulation: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn
  - Deployment: Streamlit

---

## **How to Run**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Fake_News_Detection_Project.git
   cd Fake_News_Detection_Project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model (if necessary):
   ```bash
   jupyter notebook notebooks/Fake_News_Detection.ipynb
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app/app.py
   ```

---

## **Results**

- **Accuracy**: Achieved 95% accuracy using Random Forest Classifier.
- **Precision and Recall**: Evaluated the model using precision, recall, and F1-score metrics to ensure robust performance.
- **TF-IDF Effectiveness**: TF-IDF provided excellent text representation, enabling high classification accuracy.

**Visualization**: The Streamlit app includes user-friendly visualizations of model performance and data distributions.

---

## **Future Scope**

1. **Expand Dataset**:
   - Integrate global news sources for a broader dataset.
2. **Advanced Models**:
   - Explore deep learning models like LSTMs or transformers (e.g., BERT).
3. **Real-Time Detection**:
   - Implement live news scraping and real-time classification.
4. **Multilingual Analysis**:
   - Extend capabilities to analyze non-English news articles.

---

## **Contributors**

- **Rishikesh**  
- LinkedIn: www.linkedin.com/in/rishikesh-a12090285
- Email: rishikesh23@kgpian.iitkgp.ac.in


---

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
