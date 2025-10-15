import streamlit as st
import time
from datetime import datetime
from transformers import pipeline
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas


@st.cache_resource
def load_models():
    text_gen = pipeline("text2text-generation", model="google/flan-t5-large")
    sentiment = pipeline("sentiment-analysis")
    return text_gen, sentiment

model, sentiment_model = load_models()


st.set_page_config(page_title="Mental Health Companion", page_icon="💬", layout="centered")
st.title("🌿 Mental Health Companion Dashboard")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "mood_log" not in st.session_state:
    st.session_state.mood_log = []
if "journal" not in st.session_state:
    st.session_state.journal = []
if "last_journal_time" not in st.session_state:
    st.session_state.last_journal_time = None


mood_colors = {
    "😊 Happy": "#FFF9C4",
    "😔 Sad": "#E1F5FE",
    "😠 Angry": "#FFCDD2",
    "😨 Anxious": "#D1C4E9",
    "😐 Neutral": "#F5F5F5"
}

def get_ai_response(prompt):
    formatted_input = (
        f"You are a kind, empathetic, and gentle mental health companion. "
        f"Respond supportively to the user's message: {prompt}"
    )
    response = model(formatted_input, max_length=150, do_sample=True)[0]['generated_text']
    return response.strip()

def detect_mood(text):
    result = sentiment_model(text)[0]
    label = result['label']
    score = result['score']
    mood_map = {
        "POSITIVE": "😊 Happy",
        "NEGATIVE": "😔 Sad",
        "NEUTRAL": "😐 Neutral"
    }
    mood_label = mood_map.get(label, "😐 Neutral")
    return mood_label, score

def get_relaxation_tip(mood):
    tips = {
        "😊 Happy": "Keep up the good energy! Maybe take a walk or write down what you're grateful for.",
        "😔 Sad": "Try a breathing exercise or listen to calming music. You're not alone.",
        "😐 Neutral": "How about a short journaling session to reflect on your day?",
        "😠 Angry": "Take a moment to pause. Deep breathing or a walk might help.",
        "😨 Anxious": "Try grounding techniques like naming 5 things you see around you."
    }
    return tips.get(mood, "Take a moment to pause and breathe. You're doing great.")

def plot_mood_trend():
    if st.session_state.mood_log:
        moods = [log["mood"] for log in st.session_state.mood_log]
        times = [log["time"] for log in st.session_state.mood_log]
        mood_map = {"😊 Happy": 5, "😐 Neutral": 3, "😔 Sad": 2, "😠 Angry": 1, "😨 Anxious": 2}
        mood_values = [mood_map.get(m, 3) for m in moods]

        fig, ax = plt.subplots()
        ax.plot(times, mood_values, marker='o', linestyle='-', color='purple')
        ax.set_title("Mood Trend Over Time")
        ax.set_ylabel("Mood Level")
        ax.set_xlabel("Time")
        ax.grid(True)
        fig.autofmt_xdate()
        st.pyplot(fig)
    else:
        st.info("No mood data yet. Log your mood to see trends!")


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Chat Companion",
    "📊 Mood Tracker",
    "📓 Journaling",
    "🌿 Relaxation Tools",
    "📈 Dashboard"
])


with tab1:
    st.subheader("💬 Chat with Your Companion")
    for msg in st.session_state.messages:
        timestamp = msg.get("timestamp", "")
        role = msg["role"]
        st.markdown(f"**{role.capitalize()} ({timestamp}):** {msg['content']}")
    user_input = st.text_input("Type your message here...")
    if user_input:
        now = datetime.now().strftime("%H:%M")
        st.session_state.messages.append({"role": "user", "content": user_input, "timestamp": now})
        ai_text = get_ai_response(user_input)
        mood, confidence = detect_mood(user_input)
        tip = get_relaxation_tip(mood)
        st.session_state.messages.append({"role": "assistant", "content": ai_text, "timestamp": now})

        # Apply mood-based theme
        bg_color = mood_colors.get(mood, "#F5F5F5")
        st.markdown(
            f"""<style>
                .reportview-container {{
                    background-color: {bg_color};
                }}
            </style>""",
            unsafe_allow_html=True
        )

        st.markdown(f"**Assistant ({now}):** {ai_text}")
        st.markdown(f"🧠 **Detected Mood:** {mood} ({confidence:.2f})")
        st.markdown(f"🌿 **Tip:** {tip}")

        confirmed = st.radio("Is this mood accurate?", ["Yes", "No"], horizontal=True)
        if confirmed == "No":
            corrected_mood = st.selectbox("Select your actual mood:", list(mood_colors.keys()))
            tip = get_relaxation_tip(corrected_mood)
            st.markdown(f"🌿 **Updated Tip:** {tip}")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()


with tab2:
    st.subheader("📊 Track Your Mood")
    mood = st.selectbox("How's your mood right now?", list(mood_colors.keys()))
    if st.button("Log Mood"):
        st.session_state.mood_log.append({"mood": mood, "time": datetime.now().strftime("%Y-%m-%d %H:%M")})
        st.success(f"Mood logged: {mood}")
    plot_mood_trend()


with tab3:
    st.subheader("📓 Daily & Gratitude Journal")

    # Reminder if last entry was >24h ago
    now = datetime.now()
    if st.session_state.last_journal_time:
        elapsed = (now - st.session_state.last_journal_time).total_seconds()
        if elapsed > 86400:
            st.warning("⏰ It's been a while since your last journal entry. Want to reflect today?")

    journal_entry = st.text_area("Write your thoughts here...")
    tag = st.selectbox("Tag this entry:", ["Reflection", "Gratitude", "Stress", "Goals", "Other"])
    if st.button("Save Journal Entry"):
        st.session_state.journal.append({
            "entry": journal_entry,
            "tag": tag,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        st.session_state.last_journal_time = datetime.now()
        st.success(f"Journal entry saved under '{tag}' 💾")

    gratitude = st.text_area("What are you grateful for today?")
    if st.button("Save Gratitude"):
        st.session_state.journal.append({
            "entry": f"Gratitude: {gratitude}",
            "tag": "Gratitude",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        st.session_state.last_journal_time = datetime.now()
        st.success("Gratitude saved 💛")

    if st.session_state.journal:
        st.markdown("#### 🗂️ Journal History")
        for entry in st.session_state.journal[-5:][::-1]:
            tag_display = f"**[{entry.get('tag', 'Untagged')}]**"
            st.markdown(f"- *{entry['time']}* {tag_display}: {entry['entry']}")


with tab4:
    st.subheader("🌿 Relaxation Toolkit")
    if st.button("Start Breathing Exercise"):
        st.markdown("🌬️ Breathe in... hold... breathe out...")
        for i in range(3):
            st.markdown(f"Cycle {i+1}: Inhale 🫁")
            time.sleep(4)
            st.markdown("Hold ✋")
            time.sleep(4)
            st.markdown("Exhale 💨")
            time.sleep(4)
        st.success("Feeling calmer? You did great 💖")

    st.markdown("### 🎧 Calming Sounds")
    st.video("https://www.youtube.com/watch?v=2OEL4P1Rz04")

    st.markdown("### 🧘 Guided Meditation")
    st.video("https://www.youtube.com/watch?v=inpok4MKVLM")

    st.markdown("### 🎨 Express Yourself")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Light orange
        stroke_width=2,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=300,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas_result.image_data is not None:
        st.success("Your drawing is saved in memory. Feel free to clear and start again!")


with tab5:
    st.subheader("📈 Wellness Summary")
    st.markdown(f"**Total Journal Entries:** {len(st.session_state.journal)}")
    st.markdown(f"**Total Mood Logs:** {len(st.session_state.mood_log)}")
    plot_mood_trend()
    if st.session_state.last_journal_time:
        last_entry = st.session_state.last_journal_time.strftime("%Y-%m-%d %H:%M")
        st.markdown(f"**Last Journal Entry:** {last_entry}")

