import requests
from googletrans import Translator
from pythainlp.tokenize import word_tokenize, sent_tokenize
import re
from transformers import pipeline
import json
import streamlit as st

# โหลดโมเดล Abstractive Summarization
summarizer = pipeline("summarization", model="csebuetnlp/mT5_multilingual_XLSum")

# สร้างออบเจ็กต์ Translator
translator = Translator()

# JSON mapping คำศัพท์
word_mapping = {
    "Python": "Python (ภาษาโปรแกรม)",
    "Computer": "คอมพิวเตอร์",
    "Artificial Intelligence": "ปัญญาประดิษฐ์",
    "Machine Learning": "การเรียนรู้ของเครื่อง",
    "Algorithm": "ขั้นตอนวิธี",
    "Application": "แอปพลิเคชัน",
    "Augmented Reality": "ความจริงเสริม",
    "Automation": "ระบบอัตโนมัติ",
    "Big Data": "ข้อมูลขนาดใหญ่",
    "Blockchain": "บล็อกเชน",
    # ... (คงคำศัพท์อื่น ๆ ไว้เหมือนเดิม)
}

def get_wikipedia_definition(term, lang="en"):
    """ดึงข้อมูลจาก Wikipedia"""
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": term,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        pages = data.get("query", {}).get("pages", {})

        for page in pages.values():
            if "extract" in page:
                return page["extract"]

    return None


def translate_to_thai(text):
    """แปลข้อความเป็นภาษาไทย"""
    translation = translator.translate(text, src='en', dest='th')
    return translation.text


def clean_text(text):
    """จัดการข้อความให้อ่านง่าย และลบข้อความในวงเล็บ"""
    text = re.sub(r"\(.*?\)", "", text)  # ลบข้อความในวงเล็บ
    tokens = word_tokenize(text, engine="newmm")  # ตัดคำไทย
    return " ".join(tokens)  # รวมคำใหม่ให้เป็นประโยคที่อ่านง่าย


def summarize_text(text, num_sentences=2):
    """สรุปเนื้อหาเป็นภาษาไทย"""
    sentences = sent_tokenize(text)
    extractive_summary = " ".join(sentences[:num_sentences])  # ดึง 2 ประโยคแรก

    # ใช้ Abstractive Summarization
    abstractive_summary = summarizer(extractive_summary, max_length=50, min_length=15, do_sample=False)[0][
        'summary_text']

    return abstractive_summary


def process_word(english_word):
    """ค้นหาคำอธิบาย แปล และสรุปเนื้อหา"""

    # เช็คว่าใน word_mapping มีคำนี้หรือไม่
    if english_word in word_mapping:
        mapped_translation = word_mapping[english_word]

        # ค้นหาข้อมูลจาก Wikipedia ภาษาอังกฤษ
        en_definition = get_wikipedia_definition(english_word, lang="en")

        # ค้นหาข้อมูลจาก Wikipedia ภาษาไทย (จาก word_mapping)
        th_definition = get_wikipedia_definition(mapped_translation, lang="th")
        if not th_definition:
            th_definition = "ไม่เจอ"

        # แปลและสรุปจากภาษาอังกฤษ
        thai_translation = translate_to_thai(en_definition) if en_definition else "ไม่เจอ"
        cleaned_definition = clean_text(thai_translation)
        summarized_definition1 = summarize_text(cleaned_definition) if en_definition else "ไม่เจอ"

        # สรุปจากภาษาไทย (ถ้ามี)
        summarized_definition2 = summarize_text(th_definition) if th_definition != "ไม่เจอ" else "ไม่เจอ"

        return {
            "คำค้นหา": english_word,
            "คำแปลจาก word_mapping": mapped_translation,
            "คำอธิบายจาก Wikipediaข้อมูลภาษาอังกฤษ": en_definition if en_definition else "ไม่เจอ",
            "คำอธิบายจาก Wikipediaภาษาไทย": th_definition,
            "คำแปลภาษาไทยก่อนสรุป": cleaned_definition if en_definition else "ไม่เจอ",
            "คำแปลภาษาไทยหลังสรุป1": summarized_definition1,
            "คำแปลภาษาไทยหลังสรุป2": summarized_definition2
        }

    else:
        # ค้นหาจากภาษาอังกฤษโดยตรง
        en_definition = get_wikipedia_definition(english_word, lang="en")
        if not en_definition:
            return {"error": "ไม่พบข้อมูลใน Wikipedia"}

        # แปลและสรุปจากภาษาอังกฤษ
        thai_translation = translate_to_thai(en_definition)
        cleaned_definition = clean_text(thai_translation)
        summarized_definition = summarize_text(cleaned_definition)

        return {
            "คำค้นหา": english_word,
            "คำแปล": translate_to_thai(english_word),
            "คำอธิบายจาก Wikipediaข้อมูลภาษาอังกฤษ": en_definition,
            "คำแปลภาษาWikipediaข้อมูลภาษาอังกฤษ": cleaned_definition,
            "คำแปลภาษาไทยหลังสรุป1": summarized_definition,
            "คำแปลภาษาไทยหลังสรุป2": "ไม่สามารถสรุปได้ (ไม่มีข้อมูลจาก Wikipedia ภาษาไทย)"
        }


# ใช้ Streamlit สำหรับรับ input และแสดงผล
st.title("NLP Word Processor")
st.write("กรุณาใส่คำภาษาอังกฤษที่ต้องการค้นหา")

english_word = st.text_input("คำภาษาอังกฤษ", "")

if st.button("ค้นหา"):
    if english_word:
        result = process_word(english_word)
        st.json(result)
    else:
        st.warning("กรุณาใส่คำที่ต้องการค้นหา")