import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import pandas as pd
import requests
import googletrans
from googletrans import Translator
import pygwalker as pyg
import streamlit.components.v1 as components

# Load model and tokenizer
model = load_model('my_model.h5')

with open('tokenizer.json', 'r', encoding='utf-8') as json_file:
    tokenizer_json = json.load(json_file)
    tokenizer = tokenizer_from_json(tokenizer_json)

# Function to preprocess input text
def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=250)  # Replace your_max_sequence_length with the actual value
    return padded_sequences

# Function to predict emotion
def predict_emotion(text):
    preprocessed_text = preprocess_text(text)
    prediction = model.predict(preprocessed_text)
    return prediction
def classify_sentiment(score):
    if score > 2:
        return "Cực kỳ tích cực"
    elif 0.5 <= score <= 2:
        return "Tích cực bình thường"
    elif 0 < score < 0.5:
        return "Tích cực yếu"
    elif score == 0:
        return "Trung tính"
    elif -0.5 <= score < 0:
        return "Tiêu cực yếu"
    elif -2 <= score < -0.5:
        return "Tiêu cực trung bình"
    else:
        return "Cực kỳ tiêu cực"
def Text():
    # Streamlit app

    # Người dùng chọn ngôn ngữ nguồn
    source_lang = st.selectbox("Chọn ngôn ngữ nguồn", get_supported_languages(), index=0)
    # Input text area
    input_text1 = st.text_area("Enter text:", "")
    # Người dùng chọn ngôn ngữ đích (tiếng Anh)
    target_lang = "en"

    # Các từ và điểm tương ứng
    word_scores = {
        'tôi rất thích': 0.5,
        'tôi rất ghét': -0.5,
        'tôi rất tức giận': -0.5,
        'tôi yêu nó': 0.5,
        'ghét': -0.5,
        'ghê tởm': -0.6,
        'hate': -0.5,
        'fuck': -0.5,
        'covid': -0.3,
        'COVID-19': -0.3,
        'bitch': -0.3,
        'covid-19': -0.3,
        'Coronavirus': -0.3

    }

    # Button to submit
    if st.button("Submit"):
        if input_text1:
            # Thực hiện dịch
            translation_text = translate_text(input_text1, source_lang, target_lang)
            # Make prediction
            prediction = predict_emotion(translation_text)

            # Gán điểm cho mỗi cảm xúc
            emotion_scores = {
                'joy': 0.2, 'love': 2, 'trust': 0.3, 'surprise': 0.2,
                'anticipation': 0.3, 'optimism': 1, 'neutral': 0,
                'anger': -2, 'disgust': -2, 'sadness': -1, 'fear': -2,
                'pessimism': -1.5
            }

            total_score = 0

            # Kiểm tra và cộng/trừ điểm dựa trên từng trường hợp
            for word, score in word_scores.items():
                if word in input_text1:
                    total_score += score * input_text1.count(word)

            # Display result for probabilities greater than 0.4
            #st.write("Predicted Emotions with Probabilities > 0.4:")
            for i, emotion in enumerate(
                    ['joy', 'love', 'trust', 'surprise', 'anticipation', 'optimism', 'neutral', 'anger', 'disgust',
                     'sadness', 'fear', 'pessimism']):
                if prediction[0][i] > 0.5:
                    st.write(f"{emotion}: {prediction[0][i]} (Score: {emotion_scores[emotion]})")
                    total_score += emotion_scores[emotion]

            # Display total score
            st.write("\nTotal Score:", total_score)

            # Classify the sentiment based on total score
            sentiment_class = classify_sentiment(total_score)
            st.write("Sentiment Classification:", sentiment_class)

            # Display tokens
            #st.write("\nTokens:")
            tokens = tokenizer.texts_to_sequences([translation_text])[0]
            #st.write(tokens)
        else:
            st.warning("Please enter some text.")

def textCSV(input_text, source_lang, target_lang):
    # Các từ và điểm tương ứng
    word_scores = {
        'tôi rất thích': 0.5,
        'tôi rất ghét': -0.5,
        'tôi rất tức giận': -0.5,
        'tôi yêu nó': 0.5,
    }

    # Thực hiện dịch
    translation_text = translate_text(input_text, source_lang, target_lang)
    # Make prediction
    prediction = predict_emotion(translation_text)

    # Gán điểm cho mỗi cảm xúc
    emotion_scores = {
        'joy': 1, 'love': 2, 'trust': 1, 'surprise': 0.5,
        'anticipation': 0.5, 'optimism': 1, 'neutral': 0,
        'anger': -2, 'disgust': -2, 'sadness': -1, 'fear': -2,
        'pessimism': -1.5
    }

    total_score = 0

    # Kiểm tra và cộng/trừ điểm dựa trên từng trường hợp
    for word, score in word_scores.items():
        if word in input_text:
            total_score += score * input_text.count(word)

    # Tính điểm cho các cảm xúc
    for i, emotion in enumerate(
            ['joy', 'love', 'trust', 'surprise', 'anticipation', 'optimism', 'neutral', 'anger', 'disgust',
             'sadness', 'fear', 'pessimism']):
        if prediction[0][i] > 0.4:
            total_score += emotion_scores[emotion]

    # Classify the sentiment based on total score
    sentiment_class = classify_sentiment(total_score)

    return sentiment_class

def TQDataSet():
    # Load data
    df = pd.read_csv('nlp_train.csv')
    st.write("## Đánh Giá Tổng Quan về Dữ Liệu")
    st.write(" Dữ liệu")
    st.write(df)
    st.write(f"Số dòng: {df.shape[0]}")
    st.write("### Thông Tin Về Các Cột")
    st.write(f"Số cột: {df.shape[1]}")
    st.write("Tên các cột:")
    st.write(df.columns)
def PTCsv():
    st.title("CSV Translator")

    # Người dùng nhập file CSV
    uploaded_file = st.file_uploader("Chọn một file CSV", type=["csv"])

    if uploaded_file is not None:
        # Đọc dữ liệu từ file CSV
        df = pd.read_csv(uploaded_file)

        # Hiển thị nội dung của file CSV
        st.write("Nội dung file CSV:")
        st.write(df)

        # Người dùng chọn cột chứa văn bản cần dịch
        text_column = st.selectbox("Chọn cột chứa văn bản cần dịch", df.columns)

        # Người dùng chọn ngôn ngữ nguồn
        source_lang = st.selectbox("Chọn ngôn ngữ nguồn", get_supported_languages(), index=0)

        # Người dùng chọn ngôn ngữ đích (tiếng Anh)
        target_lang = "en"

        # Dịch từng câu và tạo DataFrame mới
        translate_dataframe(df, text_column, source_lang, target_lang)
def dichCSV():
    st.title("CSV Translator")

    # Người dùng nhập file CSV
    uploaded_file = st.file_uploader("Chọn một file CSV", type=["csv"])

    if uploaded_file is not None:
        # Đọc dữ liệu từ file CSV
        df = pd.read_csv(uploaded_file)

        # Hiển thị nội dung của file CSV
        st.write("Nội dung file CSV:")
        st.write(df)

        # Người dùng chọn cột chứa văn bản cần dịch
        text_column = st.selectbox("Chọn cột chứa văn bản cần dịch", df.columns)

        # Người dùng chọn ngôn ngữ nguồn
        source_lang = st.selectbox("Chọn ngôn ngữ nguồn", get_supported_languages(), index=0)

        # Người dùng chọn ngôn ngữ đích (tiếng Anh)
        target_lang = "en"

        # Dịch từng câu và tạo DataFrame mới
        translate_dataframe(df, text_column, source_lang, target_lang)
def translate_dataframe(dataframe, text_column, source_lang, target_lang):
    # Sao chép DataFrame để tránh ảnh hưởng đến DataFrame gốc
    df = dataframe.copy()

    # Tạo cột 'KetQua'
    df['KetQua'] = df[text_column].apply(lambda x: textCSV(str(x),source_lang,target_lang))

    # Hiển thị bảng và kết quả
    st.title('Dữ liệu từ CSV')
    st.dataframe(df[[text_column, 'KetQua']])

    # Hiển thị kết quả
    st.title('Kết Quả')
    st.write(df['KetQua'].value_counts())


def ggdich():
    st.title("Multi-Language to English Translator")

    # Người dùng chọn ngôn ngữ nguồn
    source_lang = st.selectbox("Chọn ngôn ngữ nguồn", get_supported_languages(), index=0)
    # Người dùng nhập văn bản cần dịch
    input_text = st.text_area("Nhập văn bản cần dịch", "")

    # Người dùng chọn ngôn ngữ đích (tiếng Anh)
    target_lang = "en"

    if st.button("Dịch"):
        if input_text:
            # Thực hiện dịch
            translation = translate_text(input_text, source_lang, target_lang)

            # Hiển thị kết quả dịch
            st.success(f"Kết quả dịch: {translation}")
            st.write(translation)
        else:
            st.warning("Vui lòng nhập văn bản cần dịch.")

def translate_text(text, source_lang, target_lang):
    # Sử dụng thư viện Translator để dịch văn bản
    translator = Translator()
    translation = translator.translate(text, src=source_lang, dest=target_lang)
    return translation.text
def get_supported_languages():
    # Danh sách ngôn ngữ được hỗ trợ
    languages = ['vi','en', 'zh-CN', 'ko', 'ja', 'fr', 'de']
    return languages

def TQH():
    st.title("CSV Translator")

    # Người dùng nhập file CSV
    uploaded_file = st.file_uploader("Chọn một file CSV", type=["csv"])

    if uploaded_file is not None:
        # Đọc dữ liệu từ file CSV
        df = pd.read_csv(uploaded_file)

        pyg_html = pyg.walk(df, return_html=True)

        components.html(pyg_html, width=1000, height=1000, scrolling=True)
def main():
    st.title("Emotion Analysis App")

    # Thêm thanh bảng chọn bên trái
    sidebar_options = ["Phân tích đoạn Text", "Tổng quan về dataset","Phân tích theo file CSV","Trực quan hóa","google dịch"]
    selected_option = st.sidebar.selectbox("Dữ liệu", sidebar_options)

    # Khởi tạo session_state nếu chưa có
    if 'data' not in st.session_state:
        st.session_state.data = None

    if selected_option == "Phân tích đoạn Text":
       Text()

    if selected_option == "Tổng quan về dataset":
        TQDataSet()

    if selected_option == "Phân tích theo file CSV":
        PTCsv()
    if selected_option == "Trực quan hóa":
        TQH()
    if selected_option == "google dịch":
        ggdich()
    if selected_option == "dịch CSV":
        dichCSV()
if __name__ == "__main__":
    main()