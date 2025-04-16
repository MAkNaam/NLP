import streamlit as st
import requests
from googletrans import Translator
from pythainlp.tokenize import word_tokenize, sent_tokenize
import re
from transformers import pipeline, AutoTokenizer
import json
import os

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

# โหลดคำศัพท์จาก JSON
@st.cache_resource
def load_glossary():
    glossary_data = {
    "glossary": [
        {
            "en": "Artificial Intelligence (AI)",
            "th": "ประมวลผล (เอไอ)",
            "description": "ศาสตร์มุ่งเน้นเครื่องระบบให้สามารถทำงาน ห์ และได้ผลโดยไม่ต้องเขียนโปรแกรม"
        },
        {
            "en": "Machine Learning (ML)",
            "th": "การเรียนรู้ของเครื่อง (เอ็มแอล)",
            "description": "แขนงหนึ่งของ AI ที่ทำให้เครื่องสามารถเรียนรู้จากข้อมูลและปรับตัวเองได้โดยไม่ต้องเขียนโปรแกรมโดยตรง"
        },
        {
            "en": "Deep Learning (DL)",
            "th": "การเรียนรู้ลึก (แอล)",
            "description": "แขนงหนึ่งของ Machine Learning ที่ใช้โครงข่ายประสาทหลายชั้น เรียนรู้ข้อมูลซับซ้อน เช่น การจำรูปภาพ"
        },
        {
            "en": "Natural Language Processing (NLP)",
            "th": "การประมวลผลภาษา (เอ็นแอล)",
            "description": "ทำให้เครื่องสามารถเข้าใจ ห์ และตอบสนองต่อภาษาของมนุษย์ได้อย่าง"
        },
        {
            "en": "Algorithm",
            "th": "อัลกอริทึม",
            "description": "    "
        },
        {
            "en": "Data",
            "th": "ข้อมูล",
            "description": "ข้อเท็จจริง ตัวเลข ข้อความ ที่เก็บรวบรวมไว้เพื่อใช้ในการวิเคราะห์ ประมวลผล"
        },
        {
            "en": "Cloud Computing",
            "th": "การประมวลผลแบบคลาวด์",
            "description": "การให้บริการเครื่อง ฐานข้อมูล ผ่านอินเทอร์เน็ต"
        },
        {
            "en": "API (Application Programming Interface)",
            "th": "เอไอ (ส่วนต่อประสานโปรแกรมประยุกต์)",
            "description": "ของคำสั่ง กฎ และเครื่องมือช่วยให้แอปพลิเคชันต่าง ๆ สามารถสื่อสารและแลกเปลี่ยนข้อมูลได้"
        },
        {
            "en": "Blockchain",
            "th": "บล็อกเชน",
            "description": "เก็บข้อมูลแบบกระจายย์ ทำให้ข้อมูลไม่สามารถแก้ไขย้อนหลังได้ และตรวจสอบได้"
        },
        {
            "en": "Internet of Things (IoT)",
            "th": "อินเทอร์เน็ตของสรรพสิ่ง (ไอโอที)",
            "description": "เครื่องมือต่าง ๆ ที่เชื่อมต่อกันผ่านอินเทอร์เน็ต สามารถแลกเปลี่ยนข้อมูลและทำงานร่วมกัน"
        },
        {
            "en": "Cybersecurity",
            "th": "ความปลอดภัยทางไซเบอร์",
            "description": "การปกป้องระบบเครือข่าย ข้อมูล และทรัพยากรจากโจมตี การเข้าถึงโดยไม่ได้รับอนุญาต และการโจมตีทางไซเบอร์"
        },
        {
            "en": "Virtual Reality (VR)",
            "th": "ความเสมือน (อาร์)",
            "description": "จำลองสภาพแวดล้อมเสมือนมาให้สามารถโต้ตอบได้ผ่านอุปกรณ์เฉพาะ"
        },
        {
            "en": "Augmented Reality (AR)",
            "th": "ความเสริม (เออาร์)",
            "description": "นำข้อมูลเช่นภาพมาแสดงซ้อนบนโลก เอาเพื่อเพิ่มประสบการณ์ใช้งาน"
        },
        {
            "en": "Big Data",
            "th": "ข้อมูลขนาดใหญ่",
            "description": "ข้อมูลจำนวนมหาศาล ซับซ้อน และเปลี่ยนแปลงตลอดเวลา ซึ่งยากต่อการประมวลผลด้วยเครื่องคอมพิวเตอร์"
        },
        {
            "en": "Database",
            "th": "ฐานข้อมูล",
            "description": "ระบบที่ใช้เก็บและจัดข้อมูลอย่างเป็นระบบ เอาให้ง่ายต่อการค้นหาและประมวลผลข้อมูล"
        },
        {
            "en": "Edge Computing",
            "th": "เอดจ์คอมพิวเตอร์",
            "description": "การประมวลผลข้อมูลที่ใกล้แหล่งกำเนิดข้อมูล เอาเพื่อลดเวลาแฝงและเพิ่มความเร็วในการตอบสนอง"
        },
        {
            "en": "Quantum Computing",
            "th": "คอมพิวเตอร์ควอนตัม",
            "description": "การคำนวณที่ใช้ควอนตัมในการประมวลผลข้อมูล เร็วกว่าเครื่องคอมพิวเตอร์"
        },
        {
            "en": "5G",
            "th": "อินเทอร์เน็ต 5G",
            "description": "การสื่อสารไร้สายที่เร็ว ความหน่วงต่ำ สามารถเชื่อมต่ออุปกรณ์จำนวนมาก"
        }
    ]
}

    
    # ข้อมูลลงไฟล์ JSON ถ้าไม่
    glossary_file = "tech_glossary.json"
    if not os.path.exists(glossary_file):
        with open(glossary_file, "w", encoding="utf-8") as f:
            json.dump(glossary_data, f, ensure_ascii=False, indent=2)
    else:
        # อ่านข้อมูลจากไฟล์
        try:
            with open(glossary_file, "r", encoding="utf-8") as f:
                glossary_data = json.load(f)
        except Exception as e:
            st.error(f"ไม่สามารถอ่านไฟล์ glossary: {e}")
    
    return glossary_data

# ค้นหาคำศัพท์ในคลังคำศัพท์
def find_in_glossary(term):
    glossary = load_glossary()
    term_lower = term.lower()
    
    for item in glossary["glossary"]:
        if term_lower in item["en"].lower():
            return {
                "คำค้นหา": item["en"],
                "คำแปล": item["th"],
                "คำอธิบาย": item["description"],
                "คำแปลภาษาไทย": item["description"],
                "คำอธิบาย": item["description"],
                "source": "glossary"
            }
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
            if "extract" in page and page["extract"].strip():
                return page["extract"]
        return None  # No extract found or empty extract
    except requests.exceptions.RequestException as e:
        st.error(f"ข้อผิดพลาดในการดึงข้อมูลจาก Wikipedia: {e}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"ข้อผิดพลาดในการถอด JSON จาก Wikipedia: {e}")
        return None

def translate_to_thai(text):
    if translator:
        try:
            return translator.translate(text, src='en', dest='th').text
        except:
            return None
    return None

def clean_text(text):
    text = re.sub(r"\(.*?\)", "", text)  # ลบข้อความในวงเล็บ
    try:
        tokens = word_tokenize(text, engine="newmm")  # ตัดคำด้วย PyThaiNLP
        cleaned_tokens = [token.strip() for token in tokens if token.strip()]  # ลบช่องว่างที่ไม่จำเป็น
        return "".join(cleaned_tokens)  # รวมคำโดยไม่เว้นช่องว่าง
    except:
        return text

def summarize_text(text, num_sentences=10):
    try:
        sentences = sent_tokenize(text)
        extractive_summary = " ".join(sentences[:num_sentences])
    except:
        return text
    if summarizer:
        try:
            result = summarizer(extractive_summary, max_length=400, min_length=15, do_sample=False)
            return result[0]['summary_text']
        except:
            return None
    return None

def process_word(english_word):
    # ค้นหาคำศัพท์ในคลังคำศัพท์ก่อน
    glossary_result = find_in_glossary(english_word)
    if glossary_result:
        return glossary_result

    definition = get_wikipedia_definition(english_word, lang="en")
    if not definition:
        return {"error": "ไม่พบข้อมูลใน Wikipedia"}

    # ทำความสะอาดเนื้อหาก่อน
    clean_text_EG = clean_text(definition)  # ทำความสะอาดข้อความ
    summarized_definitions = summarize_text(clean_text_EG)
    if not summarized_definitions:
        return {"error": "ไม่สามารถสรุปข้อความได้"}
    
    summarized_definition = translate_to_thai(summarized_definitions)  # ทำความสะอาดข้อความแล้ว

    # แปลข้อความแล้วเป็นภาษาไทย
    thai_translation = translate_to_thai(definition)
    if not thai_translation:
        return {"error": "ไม่สามารถแปลข้อมูลเป็นภาษาไทยได้"}

    # เว้นวรรคคำแปลภาษาไทยตามภาษาไทย
    try:
        tokenized_translation = " ".join(word_tokenize(thai_translation, engine="newmm"))
    except:
        tokenized_translation = thai_translation  # หากข้อผิดพลาด ให้ใช้ข้อความ

    return {
        "คำค้นหา": english_word,
        "คำแปล": translate_to_thai(english_word),
        "คำค้นหาDF": definition,
        "คำแปลภาษาไทย": tokenized_translation,
        "คำอธิบาย": summarized_definition
    }

# ส่วนของ Streamlit UI
st.title("🌐 TechTerm Translator TH")
st.markdown(f"### 🌐 แปลคำศัพท์เทคโนโลยีเป็นภาษาไทย")
st.markdown("กรอกคำศัพท์เพื่อดูคำแปล และคำอธิบาย")

word_input = st.text_input("🔍 คำศัพท์:  ", "")

if st.button("🔍 ค้นหา"):
    if word_input.strip():
        with st.spinner("⏳ ประมวลผล..."):
            result = process_word(word_input.strip())
        if "error" in result:
            st.error(result["error"])
        else:
            st.success("✅ ค้นหาสำเร็จ")
            
            # แสดงแหล่งข้อมูล
            if result.get("source") == "glossary":
                st.info("📚 ข้อมูลจากคลังคำศัพท์")
                st.markdown(f"### 🔤 คำค้นหา: `{result['คำค้นหา']}`")
                st.markdown(f"**🔁 คำแปล:** {result['คำแปล']}")
                st.markdown("### 📚 คำอธิบาย")
                st.write(result["คำอธิบาย"])
            else:
                st.info("🌐 ข้อมูลจาก Wikipedia")
                st.markdown(f"### 🔤 คำค้นหา: `{result['คำค้นหา']}`")
                st.markdown(f"**🔁 คำแปล:** {result['คำแปล']}")
                st.markdown("### 📚 คำอธิบาย")
                st.write(result["คำค้นหาDF"])
                st.markdown("### 📝 คำแปลภาษาไทย")
                st.write(result["คำแปลภาษาไทย"])
                st.markdown("### ✨ คำอธิบาย")
                st.write(result["คำอธิบาย"])
    else:
        st.warning("⚠️ กรอกคำศัพท์ก่อนกดปุ่ม")

# ส่วนการคลังคำศัพท์
with st.expander("🔧 คลังคำศัพท์"):
    st.write("คุณสามารถเพิ่มคำศัพท์ใหม่ลงในคลังคำศัพท์ได้")
    
    new_en = st.text_input("คำศัพท์", "")
    new_th = st.text_input("คำแปลภาษาไทย", "")
    new_desc = st.text_area("คำอธิบาย", "")
    
    if st.button("เพิ่มคำศัพท์"):
        if new_en and new_th and new_desc:
            glossary = load_glossary()
            
            # ตรวจสอบว่าคำศัพท์3อยู่แล้ว3ไม่
            term_exists = False
            for item in glossary["glossary"]:
                if new_en.lower() == item["en"].lower():
                    term_exists = True
                    break
            
            if term_exists:
                st.warning(f"⚠️ คำศัพท์ '{new_en}'อยู่ในคลังคำศัพท์แล้ว ไม่สามารถเพิ่มซ้ำได้")
            else:
                # เพิ่มคำศัพท์ใหม่
                glossary["glossary"].append({
                    "en": new_en,
                    "th": new_th,
                    "description": new_desc
                })
                
                # บันทึกลงไฟล์
                try:
                    with open("tech_glossary.json", "w", encoding="utf-8") as f:
                        json.dump(glossary, f, ensure_ascii=False, indent=2)
                    st.success("✅ เพิ่มคำศัพท์สำเร็จ")
                except Exception as e:
                    st.error(f"❌ ไม่สามารถเพิ่มเข้าคลังคำศัพท์: {e}")
        else:
            st.warning("⚠️ กรอกข้อมูลให้ครบช่อง")

