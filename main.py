import random
import nltk
from nltk.corpus import movie_reviews, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Загрузка необходимых ресурсов NLTK (если еще не загружены)
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text, stop_words, lemmatizer):
    # Токенизация
    tokens = word_tokenize(text.lower())
    # Удаление стоп-слов и знаков препинания
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    # Лемматизация
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(lemmas)

def main():
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Загрузка данных и меток
    documents = [(movie_reviews.raw(fileid), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]

    # Предобработка текстов
    texts = [preprocess_text(doc, stop_words, lemmatizer) for doc, _ in documents]
    labels = [label for _, label in documents]

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels)

    # Векторизация 
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Обучение модели логистической регрессии
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    # Оценка точности
    y_pred = clf.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Вывод 3 случайных примеров с предсказаниями
    print("\nПримеры предсказаний:")
    samples = random.sample(list(zip(X_test, y_test, y_pred)), 3)
    for i, (text, true_label, pred_label) in enumerate(samples, 1):
        print(f"\nПример {i}:")
        print(f"Текст (обработанный): {text}")
        print(f"Истинный класс: {true_label}")
        print(f"Предсказанный класс: {pred_label}")

if __name__ == "__main__":
    main()
