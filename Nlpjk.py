import streamlit as st
import requests
from googletrans import Translator
from pythainlp.tokenize import word_tokenize, sent_tokenize
import re
from transformers import pipeline, AutoTokenizer
import json

# โหลดโมเดล Abstractive Summarization
@st.cache_resource
def load_summarizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
        return pipeline("summarization", model="csebuetnlp/mT5_multilingual_XLSum", tokenizer=tokenizer)
    except Exception as e:
        st.warning(f"⚠️ โหลดโมเดล mT5 ไม่สำเร็จ: {e}")
        try:
            return pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as e_bart:
            st.error(f"❌ โหลดโมเดล BART ไม่สำเร็จ: {e_bart}")
            return None

summarizer = load_summarizer()

@st.cache_resource
def load_translator():
    try:
        return Translator()
    except Exception as e:
        st.error(f"❌ ไม่สามารถโหลด Translator ได้: {e}")
        return None

translator = load_translator()

def get_wikipedia_definition(term, lang="en"):
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": term,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if "extract" in page:
                return page["extract"]
    except:
        return None

def translate_to_thai(text):
    if translator:
        try:
            return translator.translate(text, src='en', dest='th').text
        except:
            return None
    return None

def clean_text(text):
    text = re.sub(r"\(.*?\)", "", text)
    try:
        tokens = word_tokenize(text, engine="newmm")
        return " ".join(tokens)
    except:
        return text

def summarize_text(text, num_sentences=2):
    try:
        sentences = sent_tokenize(text)
        extractive_summary = " ".join(sentences[:num_sentences])
    except:
        return text
    if summarizer:
        try:
            result = summarizer(extractive_summary, max_length=50, min_length=15, do_sample=False)
            return result[0]['summary_text']
        except:
            return None
    return None

def process_word(english_word):
    definition = get_wikipedia_definition(english_word, lang="en")
    if not definition:
        return {"error": "ไม่พบข้อมูลใน Wikipedia"}

    thai_translation = translate_to_thai(definition)
    if not thai_translation:
        return {"error": "ไม่สามารถแปลข้อมูลเป็นภาษาไทยได้"}

    cleaned_definition = clean_text(thai_translation)
    summarized_definition = summarize_text(cleaned_definition)

    return {
        "คำค้นหา": english_word,
        "คำแปล": translate_to_thai(english_word),
        "คำอธิบายภาษาอังกฤษ": definition,
        "คำแปลภาษาไทยก่อนสรุป": cleaned_definition,
        "คำแปลภาษาไทยหลังสรุป": summarized_definition if summarized_definition else "ไม่สามารถสรุปข้อความได้"
    }

# ส่วนของ Streamlit UI
st.title("🌐 สรุปคำศัพท์จาก Wikipedia เป็นภาษาไทย")
st.markdown("กรอกคำศัพท์ภาษาอังกฤษเพื่อดูคำอธิบาย แปล และสรุป")

word_input = st.text_input("🔍 คำศัพท์ภาษาอังกฤษ", "")

if st.button("วิเคราะห์คำนี้"):
    if word_input.strip():
        with st.spinner("⏳ กำลังประมวลผล..."):
            result = process_word(word_input.strip())
        if "error" in result:
            st.error(result["error"])
        else:
            st.success("✅ วิเคราะห์สำเร็จ")
            st.markdown(f"### 🔤 คำค้นหา: `{result['คำค้นหา']}`")
            st.markdown(f"**🔁 คำแปล:** {result['คำแปล']}")
            st.markdown("### 📚 คำอธิบายภาษาอังกฤษ")
            st.write(result["คำอธิบายภาษาอังกฤษ"])
            st.markdown("### 📝 คำแปลภาษาไทย (ก่อนสรุป)")
            st.write(result["คำแปลภาษาไทยก่อนสรุป"])
            st.markdown("### ✨ คำแปลภาษาไทย (หลังสรุป)")
            st.write(result["คำแปลภาษาไทยหลังสรุป"])
    else:
        st.warning("⚠️ กรุณากรอกคำศัพท์ก่อนกดปุ่ม")
