import streamlit as st
import base64
import io
import folium
from streamlit_folium import st_folium
import pandas as pd
from datetime import datetime
from transformers import pipeline
from deep_translator import GoogleTranslator
from gtts import gTTS
import json
import time

try:
    import speech_recognition as sr
    _google_speech_available = True
except ModuleNotFoundError:
    _google_speech_available = False

@st.cache_resource
def load_sentiment_pipeline():
    try:
        return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    except:
        st.warning("Hugging Face model not available, using keyword fallback.")
        return None

@st.cache_resource
def load_whisper():
    try:
        import whisper
        return whisper.load_model("small")
    except Exception as e:
        st.error("Whisper could not be loaded. Using Google Speech.")
        return None

sentiment_pipe = load_sentiment_pipeline()

st.set_page_config(page_title="SamaajhVaani", layout="wide")

st.markdown("""
<style>
    .risk-high { color: white; background-color: #D32F2F; padding: 0.5em; border-radius: 10px; font-weight: bold; text-align: center; }
    .risk-medium { color: black; background-color: #FFA000; padding: 0.5em; border-radius: 10px; font-weight: bold; text-align: center; }
    .risk-low { color: black; background-color: #388E3C; padding: 0.5em; border-radius: 10px; font-weight: bold; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("SamaajhVaani – 1092 AI Assistant")
st.caption("Verified Understanding Before Action")

language_options = ["English", "Kannada", "Hindi", "Auto-detect"]
language = st.selectbox("🌐 Language", language_options, key="lang")
if "detected_language" not in st.session_state:
    st.session_state.detected_language = None
if language == "Auto-detect" and st.session_state.detected_language is not None:
    language = st.session_state.detected_language
    st.session_state.detected_language = None

lang_code = {"English": "en", "Kannada": "kn", "Hindi": "hi"}.get(language, "en")

asr_engine = st.selectbox("🎛️ ASR Engine", ["Google Web Speech (fast)", "Whisper (offline, accurate)"])
if asr_engine == "Whisper (offline, accurate)":
    st.caption("Whisper loads on first use – may take a few seconds.")
actual_engine = "whisper" if "Whisper" in asr_engine else "google"

def speak(text, lang):
    tts = gTTS(text=text, lang=lang)
    mp3 = io.BytesIO()
    tts.write_to_fp(mp3)
    return mp3.getvalue()

def autoplay_audio(audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    md = f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3">'
    st.markdown(md, unsafe_allow_html=True)

phrases = {
    "English": {
        "greeting": "1092 helpline. Do you need help? Please speak now.",
        "confirm": lambda issue: f"I understood: {issue}. Is this correct? Say yes or no.",
        "retry": "I'm sorry, I didn't get that. Please tell me your problem again.",
        "escalate": "Transferring you to a human agent. Please hold.",
        "help_dispatched": "Thank you. Help is on the way. Please stay on the line."
    },
    "Kannada": {
        "greeting": "1092 ಸಹಾಯವಾಣಿ. ನಿಮಗೆ ಸಹಾಯ ಬೇಕೇ? ದಯವಿಟ್ಟು ಹೇಳಿ.",
        "confirm": lambda issue: f"ನಾನು ಅರ್ಥ ಮಾಡಿಕೊಂಡಂತೆ: {issue}. ಇದು ಸರಿಯೆ? ಹೌದು ಅಥವಾ ಇಲ್ಲ ಎಂದು ಹೇಳಿ.",
        "retry": "ಕ್ಷಮಿಸಿ, ನನಗೆ ಸರಿಯಾಗಿ ಅರ್ಥವಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಸಮಸ್ಯೆಯನ್ನು ಮತ್ತೆ ಹೇಳಿ.",
        "escalate": "ನಿಮ್ಮ ಸಮಸ್ಯೆಯನ್ನು ಮಾನವ ಸಹಾಯಕರಿಗೆ ವರ್ಗಾಯಿಸಲಾಗುತ್ತಿದೆ. ದಯವಿಟ್ಟು ಸಾಲಿನಲ್ಲಿರಿ.",
        "help_dispatched": "ಧನ್ಯವಾದಗಳು. ಸಹಾಯ ಬರುತ್ತಿದೆ. ದಯವಿಟ್ಟು ಸಾಲಿನಲ್ಲಿರಿ."
    },
    "Hindi": {
        "greeting": "1092 हेल्पलाइन। क्या आपको मदद चाहिए? कृपया बोलिए।",
        "confirm": lambda issue: f"मैंने समझा: {issue}. क्या यह सही है? हाँ या ना कहें।",
        "retry": "क्षमा करें, मुझे ठीक से समझ नहीं आया। कृपया अपनी समस्या फिर से बताएं।",
        "escalate": "आपकी समस्या मानव सहायक को स्थानांतरित की जा रही है। कृपया लाइन में रहें।",
        "help_dispatched": "धन्यवाद। मदद आ रही है। कृपया लाइन में रहें।"
    }
}

def detect_sentiment_keywords(text, lang):
    keywords = {
        "English": {
            "distress": ["help","stuck","bleeding","accident","dying","emergency"],
            "fear": ["scared","afraid","terrified","follow","stalk"],
            "anger": ["angry","furious","upset","mad"],
            "confusion": ["confused","don't know","unknown","where"],
            "urgency": ["quick","fast","now","immediately","urgent","hurry"]
        },
        "Kannada": {
            "distress": ["ಸಹಾಯ","ರಕ್ತ","ಅಪಘಾತ","ಸಾಯು"],
            "fear": ["ಭಯ","ಹೆದರಿಕೆ","ಹಿಂಬಾಲಿಸು","ಹಿಂಬಾಲಿಕೆ"],
            "anger": ["ಕೋಪ","ಸಿಟ್ಟು"],
            "confusion": ["ಗೊಂದಲ","ಗೊತ್ತಿಲ್ಲ"],
            "urgency": ["ಬೇಗ","ತಡಮಾಡಬೇಡಿ","ಈಗ","ತಕ್ಷಣ"]
        },
        "Hindi": {
            "distress": ["मदद","बचाओ","खून","दुर्घटना"],
            "fear": ["डर","डरा","पीछा","सहमा"],
            "anger": ["गुस्सा","नाराज","चिल्ला"],
            "confusion": ["उलझन","पता नहीं"],
            "urgency": ["जल्दी","तेज़","अभी","तुरंत"]
        }
    }
    text_lower = text.lower()
    scores = {}
    for emotion, words in keywords[lang].items():
        score = sum(1 for w in words if w in text_lower)
        if score > 0:
            scores[emotion] = score
    if not scores:
        return ("Calm/Neutral", "gray", "⚪", 0.5, {})
    top = max(scores, key=scores.get)
    emoji_map = {
        "distress": ("Distress", "red", "🔴"),
        "fear": ("Fear", "orange", "🟠"),
        "anger": ("Anger", "gold", "🟡"),
        "confusion": ("Confusion", "purple", "🟣"),
        "urgency": ("Urgency", "blue", "🔵")
    }
    label, color, emoji = emoji_map.get(top, ("Calm/Neutral", "gray", "⚪"))
    max_score = max(scores.values())
    confidence = min(0.7, max_score / 5.0)
    return label, color, emoji, confidence, scores

def detect_sentiment(text, lang):
    label_map = {
        "sadness": ("Distress", "red", "🔴"),
        "fear": ("Fear", "orange", "🟠"),
        "anger": ("Anger", "gold", "🟡"),
        "joy": ("Calm/Neutral", "gray", "⚪"),
        "love": ("Calm/Neutral", "gray", "⚪"),
        "surprise": ("Calm/Neutral", "gray", "⚪"),
        "neutral": ("Calm/Neutral", "gray", "⚪")
    }

    if lang == "English":
        text_to_analyze = text
    else:
        try:
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            text_to_analyze = translated
        except:
            return detect_sentiment_keywords(text, lang)

    if sentiment_pipe is None:
        return detect_sentiment_keywords(text, lang)

    try:
        result = sentiment_pipe(text_to_analyze)
        if not result or not isinstance(result, list) or len(result) == 0:
            return detect_sentiment_keywords(text, lang)
        all_scores = result[0]
        scores = {item['label']: item['score'] for item in all_scores}
        top_item = all_scores[0]
        label = top_item['label']
        score = top_item['score']
        label_data = label_map.get(label, ("Calm/Neutral", "gray", "⚪"))
        return label_data[0], label_data[1], label_data[2], score, scores
    except Exception as e:
        st.warning(f"HF error: {e}")
        return detect_sentiment_keywords(text, lang)

def detect_dialect(text, lang):
    if lang == "Kannada":
        dialects = {
            "North Karnataka": {"patterns": ["ಹಿಂಬಾಲಿಸ್ತಾ","ಬರ್ತಾ","ಮಾಡ್ತಾ","ಕೊಡ್ತಾ","ಹೋಗ್ತಾ","ಬರುತ್ತಾಳೆ","ಮಾಡ್ತಾನೆ"], "standard": "ಹಿಂಬಾಲಿಸುತ್ತಿದ್ದಾರೆ / ಮಾಡುತ್ತಿದ್ದಾನೆ"},
            "Coastal Karnataka": {"patterns": ["ಪೋಯಿ","ಬರ್ಪೆ","ಮಲ್ಪುವೆ","ಉಂಡೆ","ಪೋಪಿನಿ"], "standard": "ಹೋಗು / ಬರುತ್ತಿದ್ದೇನೆ / ಊಟ"},
            "Old Mysore": {"patterns": ["ಮಾಡ್ತೀನಿ","ಬರ್ತೀನಿ","ಹೋಗ್ತೀನಿ","ಕೊಡ್ತೀನಿ"], "standard": "ಮಾಡುತ್ತೇನೆ / ಬರುತ್ತೇನೆ"},
            "Hubli-Dharwad": {"patterns": ["ಅವ್ನು","ಇವ್ನು","ಬ್ಯಾಡ"], "standard": "ಅವನು / ಇವನು / ಬೇಡ"},
            "Gulbarga": {"patterns": ["ಹೋತೀನಿ","ಬರತೀನಿ"], "standard": "ಹೋಗ್ತೀನಿ / ಬರ್ತೀನಿ"}
        }
        matched = None
        best_count = 0
        for region, data in dialects.items():
            count = sum(1 for p in data["patterns"] if p in text)
            if count > best_count:
                best_count = count
                matched = f"{region} → standard: {data['standard']} (matched {count} phrases)"
        return matched if matched else None

    elif lang == "Hindi":
        patterns = {
            "Mumbai Hindi": {"patterns": ["इधर","उधर","अपुन","भिडू","चपरगंजू"], "standard": "इधर / वहाँ / हम / दोस्त"},
            "Bhojpuri": {"patterns": ["हम","रउआ","का हो"], "standard": "मैं / आप / क्या हो रहा है"}
        }
        best_count = 0
        result = None
        for region, data in patterns.items():
            count = sum(1 for p in data["patterns"] if p in text)
            if count > best_count:
                best_count = count
                result = f"{region} → standard: {data['standard']} (matched {count} phrases)"
        return result

    elif lang == "English":
        patterns = {
            "Indian English": {"patterns": ["kindly do the needful","passing out","prepone","out of station","revert back"], "standard": "Please complete / graduate / move earlier / out of town / reply"},
            "African American Vernacular": {"patterns": ["finna","y'all","he be","ain't","imma"], "standard": "fixing to / you all / he is / am not / I am going to"},
            "British English": {"patterns": ["bloke","loo","chuffed","knackered","gobsmacked"], "standard": "man / toilet / pleased / exhausted / amazed"},
            "Scottish": {"patterns": ["aye","wee","lass","ken","outwith"], "standard": "yes / small / girl / know / outside"}
        }
        best_count = 0
        result = None
        for region, data in patterns.items():
            count = sum(1 for p in data["patterns"] if p in text.lower())
            if count > best_count:
                best_count = count
                result = f"{region} → standard: {data['standard']} (matched {count} phrases)"
        return result
    return None

def detect_formality(text, lang):
    formal_indicators = {
        "English": ["please", "kindly", "sir", "madam", "request", "respectfully"],
        "Kannada": ["ದಯವಿಟ್ಟು", "ಕೃಪೆ", "ಸ್ವಾಮಿ", "ಅಮ್ಮ", "ಮಹಾಶಯ"],
        "Hindi": ["कृपया", "श्रीमान", "श्रीमती", "महोदय", "सविनय"]
    }
    casual_indicators = {
        "English": ["hey", "yeah", "gonna", "wanna", "dude", "bro"],
        "Kannada": ["ಏ", "ಮಗ", "ಲೇ", "ಮಾರಾಯ"],
        "Hindi": ["अरे", "यार", "अबे", "भाई"]
    }
    text_lower = text.lower()
    formal_score = sum(1 for w in formal_indicators[lang] if w in text_lower)
    casual_score = sum(1 for w in casual_indicators[lang] if w in text_lower)
    if formal_score > casual_score:
        return "Formal"
    elif casual_score > formal_score:
        return "Casual"
    return "Neutral"

def extract_location(text):
    text_low = text.lower()
    landmarks = {
        "bus stop": [12.9716, 77.5946],
        "jayanagar": [12.9308, 77.5838],
        "mg road": [12.9756, 77.6066],
        "market": [12.9600, 77.5900],
        "hospital": [12.9601, 77.6411],
        "police station": [12.9600, 77.6000]
    }
    for loc, coords in landmarks.items():
        if loc in text_low:
            return coords, loc.title()
    return [12.9716, 77.5946], "Bangalore City Center"

def risk_level(sentiment_name, text=""):
    emergency_words = ["help","bleeding","accident","dying","weapon","attack",
                       "ರಕ್ತ","ಅಪಘಾತ","मदद","खून","दुर्घटना"]
    if sentiment_name in ["Distress","Fear","Urgency"] or any(w in text.lower() for w in emergency_words):
        return "HIGH","#D32F2F"
    elif sentiment_name == "Anger":
        return "MEDIUM","#FFA000"
    elif sentiment_name == "Confusion":
        return "MEDIUM","#FFA000"
    else:
        return "LOW","#388E3C"

def transcribe_with_whisper(audio_path, lang_code_hint):
    model = load_whisper()
    if model is None:
        return None, 0.0, None
    try:
        lang_arg = None if lang_code_hint == "auto" else lang_code_hint[:2]
        result = model.transcribe(audio_path, language=lang_arg, fp16=False)
        detected_lang = result.get("language", "en")
        return result["text"], 0.9, detected_lang
    except Exception:
        return None, 0.0, None

if "stage" not in st.session_state:
    st.session_state.stage = "idle"
if "issue_text" not in st.session_state:
    st.session_state.issue_text = ""
if "confirmation_count" not in st.session_state:
    st.session_state.confirmation_count = 0
if "greeting_audio_played" not in st.session_state:
    st.session_state.greeting_audio_played = False
if "confirm_audio_played" not in st.session_state:
    st.session_state.confirm_audio_played = False
if "sentiment" not in st.session_state:
    st.session_state.sentiment = ("Calm/Neutral","gray","⚪", 0.5, {})
if "dialect_info" not in st.session_state:
    st.session_state.dialect_info = None
if "formality" not in st.session_state:
    st.session_state.formality = "Neutral"
if "risk" not in st.session_state:
    st.session_state.risk = ("—","gray")
if "ai_action" not in st.session_state:
    st.session_state.ai_action = "—"
if "call_log" not in st.session_state:
    st.session_state.call_log = []
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []
if "asr_confidence" not in st.session_state:
    st.session_state.asr_confidence = 0.0
if "understanding_confidence" not in st.session_state:
    st.session_state.understanding_confidence = 0
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False
if "escalation_reason" not in st.session_state:
    st.session_state.escalation_reason = ""

demo_on = st.sidebar.checkbox("🎭 Demo Mode (automated showcase)", value=False)
view = st.sidebar.radio("View", ["Agent View", "Analytics"], horizontal=True)

current_phrases = phrases.get(language, phrases["English"])

if view == "Agent View":
    with st.sidebar:
        st.header("🖥️ Agent Dashboard")
        if st.session_state.stage not in ["idle","done"]:
            st.subheader("Active Call")
            st.write(f"**Language:** {language}")
            st.write(f"**AI Interpretation:** {st.session_state.issue_text if st.session_state.issue_text else 'Waiting...'}")

            sent_label, sent_color, sent_emoji, sent_conf, sent_scores = st.session_state.sentiment
            st.markdown(f"**Sentiment:** {sent_emoji} <span style='color:{sent_color}; font-weight:bold'>{sent_label}</span>", unsafe_allow_html=True)

            if st.session_state.dialect_info:
                st.info(f"🗣️ {st.session_state.dialect_info}")
            st.caption(f"🗣️ Tone: {st.session_state.formality}")

            risk_text, risk_color = st.session_state.risk
            risk_class = "risk-high" if risk_text=="HIGH" else ("risk-medium" if risk_text=="MEDIUM" else "risk-low")
            st.markdown(f"<div class='{risk_class}'>{'⚠️ '+risk_text+' RISK'}</div>", unsafe_allow_html=True)

            if st.session_state.escalation_reason:
                st.error(f"🔄 Escalation reason: {st.session_state.escalation_reason}")

            coords, location_name = extract_location(st.session_state.issue_text)
            m = folium.Map(location=coords, zoom_start=15)
            folium.Marker(coords, popup=location_name, tooltip="Caller Location").add_to(m)
            st_folium(m, width=280, height=200)

            st.markdown("**Agent Actions:**")
            if st.button("✅ Accept AI Interpretation", key="accept_btn"):
                st.success("Agent accepted the AI interpretation.")

            if st.button("✏️ Edit Transcript" if not st.session_state.edit_mode else "❌ Cancel Edit", key="edit_btn"):
                st.session_state.edit_mode = not st.session_state.edit_mode

            if st.session_state.edit_mode:
                corrected_text = st.text_area("Correct the AI interpretation:", value=st.session_state.issue_text, key="edit_text")
                if st.button("💾 Save Correction", key="save_correction"):
                    st.session_state.feedback_log.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "original": st.session_state.issue_text,
                        "verified": False,
                        "corrected": corrected_text
                    })
                    st.session_state.issue_text = corrected_text
                    st.session_state.edit_mode = False
                    st.success("Transcript corrected and logged for future model improvement.")
                    st.rerun()

            if st.button("🔄 Manual Escalation", key="escal_btn"):
                st.error("Manual escalation triggered. Agent takes over the call.")
                st.session_state.stage = "done"
                st.rerun()

        elif st.session_state.stage == "done":
            st.success("Call ended.")
        else:
            st.info("Press 'Start Call' to begin.")

        st.markdown("---")
        st.subheader("📜 Call Log")
        if st.session_state.call_log:
            df = pd.DataFrame(st.session_state.call_log)
            st.dataframe(df[["Time","Language","Issue","Sentiment","Risk","Status"]], use_container_width=True)
        else:
            st.write("No calls yet.")

elif view == "Analytics":
    with st.sidebar:
        st.header("📊 Supervisor Analytics")
        if st.session_state.call_log:
            df = pd.DataFrame(st.session_state.call_log)
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Calls", len(df))
            col2.metric("Avg Risk HIGH", f"{(df['Risk']=='HIGH').mean():.0%}")
            col3.metric("Escalation Rate", f"{(df['Status']=='Escalated').mean():.0%}")
            if 'SentimentConf' in df.columns and 'UnderstandingConf' in df.columns:
                col4, col5 = st.columns(2)
                avg_sent_conf = df['SentimentConf'].mean()
                avg_under_conf = df['UnderstandingConf'].mean()
                col4.metric("Avg Sentiment Confidence", f"{avg_sent_conf:.0%}")
                col5.metric("Avg Understanding Confidence", f"{avg_under_conf:.0%}")
            st.subheader("Sentiment Distribution")
            sentiment_counts = df['Sentiment'].value_counts()
            st.bar_chart(sentiment_counts)
            st.subheader("Recent Calls")
            st.dataframe(df[["Time","Language","Issue","Sentiment","Risk","Status"]], use_container_width=True)

            st.subheader("📈 Learning Progress")
            if st.session_state.feedback_log:
                total_feedback = len(st.session_state.feedback_log)
                verified_count = sum(1 for entry in st.session_state.feedback_log if entry['verified'])
                st.metric("Feedback Samples", total_feedback)
                st.progress(verified_count / total_feedback if total_feedback > 0 else 0,
                            text=f"Confirmed: {verified_count}/{total_feedback} ({verified_count/total_feedback:.0%})"
                            if total_feedback > 0 else "No data")
                st.caption("Confirmed interpretations become validated training data. Corrections are saved for model fine‑tuning.")
            else:
                st.info("No learning signals yet. Complete calls to generate feedback.")
        else:
            st.info("No call data yet. Start a call to see analytics.")

sim_issues = {
    "English": "Help, a man is following me near the bus stop. I am scared.",
    "Kannada": "ದಯವಿಟ್ಟು ಸಹಾಯ ಮಾಡಿ, ಒಬ್ಬ ಹಿಂಬಾಲಿಸ್ತಾ ಇದ್ದಾನೆ ಬಸ್ ಸ್ಟಾಪ್ ಹತ್ತಿರ.",
    "Hindi": "मदद करो, कोई आदमी बस स्टॉप के पास मेरा पीछा कर रहा है।"
}

if st.session_state.stage == "idle":
    if st.button("📞 Start Call"):
        st.session_state.stage = "issue"
        st.session_state.issue_text = ""
        st.session_state.confirmation_count = 0
        st.session_state.greeting_audio_played = False
        st.session_state.confirm_audio_played = False
        st.session_state.sentiment = ("Calm/Neutral","gray","⚪",0.5,{})
        st.session_state.dialect_info = None
        st.session_state.formality = "Neutral"
        st.session_state.risk = ("—","gray")
        st.session_state.ai_action = "—"
        st.session_state.asr_confidence = 0.0
        st.session_state.understanding_confidence = 0
        st.session_state.edit_mode = False
        st.session_state.escalation_reason = ""
        st.rerun()

elif st.session_state.stage == "issue":
    if not st.session_state.greeting_audio_played:
        autoplay_audio(speak(current_phrases["greeting"], lang_code))
        st.session_state.greeting_audio_played = True
        if demo_on:
            st.markdown("#### 🎙️ Greeting playing... (demo mode will auto-fill)")
            import time
            time.sleep(4)
            issue = sim_issues.get(language, sim_issues["English"])
            sent_label, sent_color, sent_emoji, sent_conf, sent_scores = detect_sentiment(issue, language)
            dialect = detect_dialect(issue, language)
            formality = detect_formality(issue, language)
            risk_text, risk_color = risk_level(sent_label, issue)
            st.session_state.sentiment = (sent_label, sent_color, sent_emoji, sent_conf, sent_scores)
            st.session_state.dialect_info = dialect
            st.session_state.formality = formality
            st.session_state.risk = (risk_text, risk_color)
            st.session_state.issue_text = issue
            st.session_state.asr_confidence = 0.95
            st.session_state.understanding_confidence = round((0.6 * 0.95 + 0.4 * sent_conf) * 100)

            if st.session_state.feedback_log and not st.session_state.feedback_log[-1]['verified'] and not st.session_state.feedback_log[-1]['corrected']:
                st.session_state.feedback_log[-1]['corrected'] = issue

            if risk_text == "HIGH" or st.session_state.understanding_confidence < 40:
                reason = "High risk" if risk_text == "HIGH" else "Low understanding confidence"
                st.session_state.escalation_reason = reason
                st.error(f"🔄 {reason}. Escalating to human agent immediately.")
                autoplay_audio(speak(current_phrases["escalate"], lang_code))
                st.session_state.ai_action = "Escalated"
                st.session_state.call_log.append({
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Language": language,
                    "Issue": st.session_state.issue_text[:50]+"...",
                    "Sentiment": st.session_state.sentiment[0],
                    "SentimentConf": st.session_state.sentiment[3],
                    "UnderstandingConf": st.session_state.understanding_confidence,
                    "Risk": st.session_state.risk[0],
                    "Status": "Escalated"
                })
                st.session_state.stage = "done"
                st.rerun()
            else:
                st.session_state.stage = "confirm"
                st.session_state.confirm_audio_played = False
                st.rerun()

    if not demo_on: 
        st.markdown("#### 🎙️ Greeting is playing… please describe your problem after it finishes.")
        audio_val = st.audio_input("Record your problem", key="issue_rec")
        if audio_val is not None:
            audio_bytes = audio_val.getvalue()
            if audio_bytes and len(audio_bytes) > 0:
                with open("issue.wav","wb") as f:
                    f.write(audio_bytes)
                asr_conf = 0.0
                issue = None
                detected_lang = None

                lang_arg = lang_code if language != "Auto-detect" else "auto"
                if actual_engine == "whisper" or not _google_speech_available or language == "Auto-detect":
                    with st.spinner("Transcribing with Whisper..."):
                        issue, asr_conf, detected_lang = transcribe_with_whisper("issue.wav", lang_arg)

                if not issue and _google_speech_available and language != "Auto-detect":
                    rec = sr.Recognizer()
                    with sr.AudioFile("issue.wav") as src:
                        adata = rec.record(src)
                    try:
                        response = rec.recognize_google(adata, language=f"{lang_code}-IN", show_all=True)
                        if response and 'alternative' in response:
                            issue = response['alternative'][0]['transcript']
                            asr_conf = response['alternative'][0].get('confidence', 0.0)
                    except:
                        pass

                if language == "Auto-detect" and issue and detected_lang:
                    mapping = {"en": "English", "kn": "Kannada", "hi": "Hindi"}
                    new_lang = mapping.get(detected_lang, "English")
                    st.session_state.detected_language = new_lang
                    st.rerun()

                if not issue:
                    issue = "[Could not understand]"
                    asr_conf = 0.0

                if issue == "[Could not understand]":
                    st.session_state.sentiment = ("Unknown", "gray", "❓", 0.0, {})
                    st.session_state.dialect_info = None
                    st.session_state.formality = "Neutral"
                    st.session_state.risk = ("HIGH", "#D32F2F")
                    st.session_state.issue_text = issue
                    st.session_state.asr_confidence = 0.0
                    st.session_state.understanding_confidence = 0
                    st.session_state.escalation_reason = "Speech unrecognized"
                    st.error("🔄 Could not understand speech. Escalating to human agent.")
                    autoplay_audio(speak(current_phrases["escalate"], lang_code))
                    st.session_state.ai_action = "Escalated"
                    st.session_state.call_log.append({
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Language": language,
                        "Issue": issue,
                        "Sentiment": "Unknown",
                        "SentimentConf": 0.0,
                        "UnderstandingConf": 0,
                        "Risk": "HIGH",
                        "Status": "Escalated"
                    })
                    st.session_state.stage = "done"
                    st.rerun()
                else:
                    if asr_conf == 0.0:
                        asr_conf = 0.7
                    sent_label, sent_color, sent_emoji, sent_conf, sent_scores = detect_sentiment(issue, language)
                    dialect = detect_dialect(issue, language)
                    formality = detect_formality(issue, language)
                    risk_text, risk_color = risk_level(sent_label, issue)
                    st.session_state.sentiment = (sent_label, sent_color, sent_emoji, sent_conf, sent_scores)
                    st.session_state.dialect_info = dialect
                    st.session_state.formality = formality
                    st.session_state.risk = (risk_text, risk_color)
                    st.session_state.issue_text = issue
                    st.session_state.asr_confidence = asr_conf
                    st.session_state.understanding_confidence = round((0.6 * asr_conf + 0.4 * sent_conf) * 100)
                    st.markdown(f"**📝 Transcribed:** {issue}")
                    st.markdown(f"**Sentiment:** {sent_emoji} <span style='color:{sent_color}'>{sent_label}</span>", unsafe_allow_html=True)
                    if dialect:
                        st.info(dialect)
                    st.caption(f"🗣️ Tone: {formality}")

                    if st.session_state.feedback_log and not st.session_state.feedback_log[-1]['verified'] and not st.session_state.feedback_log[-1]['corrected']:
                        st.session_state.feedback_log[-1]['corrected'] = issue

                    if risk_text == "HIGH" or st.session_state.understanding_confidence < 40:
                        reason = "High risk" if risk_text == "HIGH" else "Low understanding confidence"
                        st.session_state.escalation_reason = reason
                        st.error(f"🔄 {reason}. Escalating to human agent immediately.")
                        autoplay_audio(speak(current_phrases["escalate"], lang_code))
                        st.session_state.ai_action = "Escalated"
                        st.session_state.call_log.append({
                            "Time": datetime.now().strftime("%H:%M:%S"),
                            "Language": language,
                            "Issue": st.session_state.issue_text[:50]+"...",
                            "Sentiment": st.session_state.sentiment[0],
                            "SentimentConf": st.session_state.sentiment[3],
                            "UnderstandingConf": st.session_state.understanding_confidence,
                            "Risk": st.session_state.risk[0],
                            "Status": "Escalated"
                        })
                        st.session_state.stage = "done"
                        st.rerun()
                    else:
                        st.session_state.stage = "confirm"
                        st.session_state.confirm_audio_played = False
                        st.rerun()
    else:
        if st.session_state.issue_text:
            st.write(f"**📝 Demo Issue:** {st.session_state.issue_text}")
            sent_label, sent_color, sent_emoji, _, _ = st.session_state.sentiment
            st.markdown(f"**Sentiment:** {sent_emoji} <span style='color:{sent_color}'>{sent_label}</span>", unsafe_allow_html=True)
            if st.session_state.dialect_info:
                st.info(st.session_state.dialect_info)
            st.caption(f"🗣️ Tone: {st.session_state.formality}")
            st.rerun()

elif st.session_state.stage == "confirm":
    if not st.session_state.confirm_audio_played:
        confirm_phrase = current_phrases["confirm"](st.session_state.issue_text)
        autoplay_audio(speak(confirm_phrase, lang_code))
        st.session_state.confirm_audio_played = True
        if demo_on:
            import time
            time.sleep(4)
            answer = "yes"
            if answer.lower() in ["yes","ಹೌದು","हाँ"]:
                st.success("✅ Confirmed. Help dispatched.")
                autoplay_audio(speak(current_phrases["help_dispatched"], lang_code))
                st.session_state.call_log.append({
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Language": language,
                    "Issue": st.session_state.issue_text[:50]+"...",
                    "Sentiment": st.session_state.sentiment[0],
                    "SentimentConf": st.session_state.sentiment[3],
                    "UnderstandingConf": st.session_state.understanding_confidence,
                    "Risk": st.session_state.risk[0],
                    "Status": "Dispatched"
                })
                st.session_state.feedback_log.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "original": st.session_state.issue_text,
                    "verified": True,
                    "corrected": None
                })
                st.session_state.understanding_confidence = 100
                st.session_state.stage = "done"
                st.rerun()
            else:
                st.error("Escalating to human agent (demo).")
                autoplay_audio(speak(current_phrases["escalate"], lang_code))
                st.session_state.call_log.append({
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Language": language,
                    "Issue": st.session_state.issue_text[:50]+"...",
                    "Sentiment": st.session_state.sentiment[0],
                    "SentimentConf": st.session_state.sentiment[3],
                    "UnderstandingConf": st.session_state.understanding_confidence,
                    "Risk": st.session_state.risk[0],
                    "Status": "Escalated"
                })
                st.session_state.feedback_log.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "original": st.session_state.issue_text,
                    "verified": False,
                    "corrected": None
                })
                st.session_state.understanding_confidence = max(0, st.session_state.understanding_confidence - 30)
                st.session_state.stage = "done"
                st.rerun()

    if not demo_on:
        st.markdown(f"#### 🤖 AI verifies: *{current_phrases['confirm'](st.session_state.issue_text)}*")
        st.write("🗣️ Answer **yes** or **no** after the question.")
        confirm_audio = st.audio_input("Record your yes/no", key="confirm_rec")
        if confirm_audio is not None:
            confirm_bytes = confirm_audio.getvalue()
            if confirm_bytes and len(confirm_bytes) > 0:
                with open("confirm.wav","wb") as f:
                    f.write(confirm_bytes)

                answer = ""
                if not _google_speech_available:
                    ans_text, _, _ = transcribe_with_whisper("confirm.wav", lang_code)
                    if ans_text:
                        answer = ans_text
                else:
                    try:
                        rec = sr.Recognizer()
                        with sr.AudioFile("confirm.wav") as src:
                            adata = rec.record(src)
                        answer = rec.recognize_google(adata, language=f"{lang_code}-IN")
                    except:
                        answer = ""

                st.write(f"**📝 You answered:** {answer}")

                yes_words = {"English": ["yes","yeah","yep","correct","right"],
                             "Kannada": ["ಹೌದು","ಸರಿ","ಹಂ"],
                             "Hindi": ["हाँ","हां","जी","सही"]}
                no_words = {"English": ["no","nope","wrong","incorrect"],
                            "Kannada": ["ಇಲ್ಲ","ಅಲ್ಲ","ಬೇಡ"],
                            "Hindi": ["ना","नहीं","गलत"]}
                is_yes = any(w in answer.lower() for w in yes_words[language])
                is_no = any(w in answer.lower() for w in no_words[language])

                if is_yes:
                    st.success("✅ Confirmed. Help dispatched.")
                    autoplay_audio(speak(current_phrases["help_dispatched"], lang_code))
                    st.session_state.ai_action = "Dispatched"
                    st.session_state.call_log.append({
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Language": language,
                        "Issue": st.session_state.issue_text[:50]+"...",
                        "Sentiment": st.session_state.sentiment[0],
                        "SentimentConf": st.session_state.sentiment[3],
                        "UnderstandingConf": st.session_state.understanding_confidence,
                        "Risk": st.session_state.risk[0],
                        "Status": "Dispatched"
                    })
                    st.session_state.feedback_log.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "original": st.session_state.issue_text,
                        "verified": True,
                        "corrected": None
                    })
                    st.session_state.understanding_confidence = 100
                    st.session_state.stage = "done"
                    st.rerun()
                elif is_no:
                    st.session_state.confirmation_count += 1
                    st.session_state.feedback_log.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "original": st.session_state.issue_text,
                        "verified": False,
                        "corrected": None
                    })
                    if st.session_state.confirmation_count < 2:
                        st.warning("❌ Not confirmed. Retrying...")
                        st.session_state.issue_text = ""
                        st.session_state.stage = "issue"
                        st.session_state.greeting_audio_played = True
                        st.session_state.understanding_confidence = max(0, st.session_state.understanding_confidence - 30)
                        st.audio(speak(current_phrases["retry"], lang_code), autoplay=True)
                        st.rerun()
                    else:
                        st.session_state.escalation_reason = "Repeated citizen denial"
                        st.error("🔄 Escalating to human agent.")
                        autoplay_audio(speak(current_phrases["escalate"], lang_code))
                        st.session_state.ai_action = "Escalated"
                        st.session_state.call_log.append({
                            "Time": datetime.now().strftime("%H:%M:%S"),
                            "Language": language,
                            "Issue": st.session_state.issue_text[:50]+"...",
                            "Sentiment": st.session_state.sentiment[0],
                            "SentimentConf": st.session_state.sentiment[3],
                            "UnderstandingConf": st.session_state.understanding_confidence,
                            "Risk": st.session_state.risk[0],
                            "Status": "Escalated"
                        })
                        st.session_state.understanding_confidence = max(0, st.session_state.understanding_confidence - 50)
                        st.session_state.stage = "done"
                        st.rerun()
                else:
                    st.warning("Could not understand answer. Escalating.")
                    autoplay_audio(speak(current_phrases["escalate"], lang_code))
                    st.session_state.ai_action = "Escalated"
                    st.session_state.escalation_reason = "Unclear verification response"
                    st.session_state.call_log.append({
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Language": language,
                        "Issue": st.session_state.issue_text[:50]+"...",
                        "Sentiment": st.session_state.sentiment[0],
                        "SentimentConf": st.session_state.sentiment[3],
                        "UnderstandingConf": st.session_state.understanding_confidence,
                        "Risk": st.session_state.risk[0],
                        "Status": "Escalated"
                    })
                    st.session_state.feedback_log.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "original": st.session_state.issue_text,
                        "verified": False,
                        "corrected": None
                    })
                    st.session_state.understanding_confidence = max(0, st.session_state.understanding_confidence - 50)
                    st.session_state.stage = "done"
                    st.rerun()

elif st.session_state.stage == "done":
    st.success("Call completed. Refresh or start a new call.")
    call_summary = {
        "language": language,
        "issue": st.session_state.issue_text,
        "sentiment": st.session_state.sentiment[0],
        "sentiment_confidence": st.session_state.sentiment[3],
        "asr_confidence": st.session_state.asr_confidence,
        "overall_understanding_confidence": st.session_state.understanding_confidence,
        "risk": st.session_state.risk[0],
        "status": "Dispatched" if st.session_state.ai_action == "Dispatched" else "Escalated",
        "escalation_reason": st.session_state.escalation_reason,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.download_button(
        label="📥 Download Call Summary (JSON)",
        data=json.dumps(call_summary, indent=2),
        file_name=f"call_{datetime.now().strftime('%H%M%S')}.json",
        mime="application/json"
    )
    if st.button("🔄 New Call"):
        logs_to_keep = {
            "call_log": st.session_state.get("call_log", []),
            "feedback_log": st.session_state.get("feedback_log", [])
        }
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.call_log = logs_to_keep["call_log"]
        st.session_state.feedback_log = logs_to_keep["feedback_log"]
        st.rerun()
