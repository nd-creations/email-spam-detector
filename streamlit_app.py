import streamlit as st
import imaplib
import email
import re
import time
import sqlite3
import pandas as pd
import nltk
from nltk.corpus import stopwords
from datetime import datetime, date
from email.utils import parsedate_to_datetime
from email import policy
from email.parser import BytesParser, Parser
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Spam Detector",
    page_icon="📧",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #0d0f14;
    color: #e8eaf0;
}

h1, h2, h3 { font-family: 'Space Mono', monospace; }

.stButton > button {
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

.metric-card {
    background: #161b26;
    border: 1px solid #2a2f3e;
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    text-align: center;
}
.metric-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.1em; color: #7a8099; }
.metric-value { font-size: 2.2rem; font-weight: 700; font-family: 'Space Mono', monospace; }

.email-row {
    padding: 0.55rem 0.9rem;
    border-radius: 8px;
    margin-bottom: 5px;
    font-size: 0.9rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.safe-row   { background: #0d2418; border-left: 3px solid #22c55e; }
.spam-row   { background: #2a1010; border-left: 3px solid #ef4444; }
.phish-row  { background: #2a1a08; border-left: 3px solid #f97316; }

.score-badge {
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 20px;
    font-weight: 600;
    margin-left: auto;
    white-space: nowrap;
}
.badge-safe   { background: #14532d; color: #86efac; }
.badge-spam   { background: #450a0a; color: #fca5a5; }
.badge-phish  { background: #431407; color: #fdba74; }

.section-header {
    border-bottom: 1px solid #2a2f3e;
    padding-bottom: 0.5rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STOPWORDS
# ─────────────────────────────────────────────
@st.cache_resource
def load_stopwords():
    try:
        return set(stopwords.words("english"))
    except Exception:
        nltk.download("stopwords")
        return set(stopwords.words("english"))

stop_words = load_stopwords()


# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
@st.cache_resource
def get_db():
    conn = sqlite3.connect("spam_results.db", check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS emails (
            uid        TEXT PRIMARY KEY,
            subject    TEXT,
            label      TEXT,
            score      INTEGER,
            email_date TEXT,
            source     TEXT DEFAULT 'imap'
        )
    """)
    conn.commit()
    return conn, c

conn, c = get_db()


# ─────────────────────────────────────────────
# TEXT UTILITIES
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " url ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    return " ".join(w for w in text.split() if w not in stop_words)


# ─────────────────────────────────────────────
# SCORING ENGINE  (returns label, score 0-100)
# ─────────────────────────────────────────────

SPAM_KEYWORDS = {
    # High-weight phishing / scam triggers
    "verify": 15, "password": 15, "urgent": 12, "suspended": 12,
    "account": 8,  "login": 10,   "click": 8,   "bank": 10,
    "credit": 8,   "ssn": 18,     "social security": 18,
    "wire transfer": 15, "bitcoin": 12, "crypto": 10,
    # Medium-weight spam triggers
    "free": 6, "winner": 8, "congratulations": 7, "prize": 8,
    "offer": 5, "deal": 5,  "discount": 5, "sale": 4,
    "limited time": 7, "act now": 8, "guarantee": 6,
    "unsubscribe": 4, "opt out": 4,
    # Filler spam phrases
    "buy now": 7, "order now": 7, "cash": 6, "earn money": 8,
    "work from home": 8, "make money": 8, "million dollars": 12,
    "nigerian": 12, "inheritance": 10, "transfer funds": 12,
    "claim your": 8, "you have been selected": 10,
    "risk free": 6, "100%": 5, "satisfaction guaranteed": 6,
    "no obligation": 5, "no credit check": 7, "pre-approved": 8,
}

PHISHING_PATTERNS = [
    r"verify.{0,30}(account|identity|email)",
    r"(click|tap).{0,20}(link|here|below)",
    r"(your|the).{0,10}account.{0,20}(suspended|locked|disabled)",
    r"(provide|enter|confirm).{0,20}(password|credentials|details)",
    r"unusual.{0,20}(activity|sign.?in|access)",
    r"security.{0,20}(alert|warning|update)",
    r"(update|confirm).{0,20}(payment|billing|card)",
]

def classify_email(subject: str, body: str):
    """Returns (label, score) where score 0-100, label Safe/Spam/Phishing."""
    combined = (subject + " " + body).lower()
    cleaned  = clean_text(subject + " " + body)
    words    = cleaned.split()

    score = 0

    # Keyword scoring
    for kw, weight in SPAM_KEYWORDS.items():
        if kw in combined:
            score += weight

    # Regex phishing patterns
    for pattern in PHISHING_PATTERNS:
        if re.search(pattern, combined):
            score += 20

    # Urgency caps: ALL CAPS words
    caps_words = len(re.findall(r"\b[A-Z]{3,}\b", subject + " " + body))
    score += min(caps_words * 3, 15)

    # Excessive punctuation
    excl = body.count("!") + subject.count("!")
    score += min(excl * 2, 10)

    # URL presence
    urls = len(re.findall(r"http\S+", body))
    score += min(urls * 5, 20)

    score = min(score, 100)

    if score >= 50:
        # Distinguish phishing vs generic spam
        phishing_hit = any(re.search(p, combined) for p in PHISHING_PATTERNS)
        phishing_hit = phishing_hit or any(
            kw in combined for kw in ["verify","password","login","bank","ssn","wire transfer"]
        )
        label = "Phishing" if phishing_hit else "Spam"
    else:
        label = "Safe"

    return label, score


# ─────────────────────────────────────────────
# BODY EXTRACTOR
# ─────────────────────────────────────────────
def extract_body(msg) -> str:
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain":
                raw = part.get_payload(decode=True)
                if raw:
                    body = raw.decode(errors="ignore")
                    break
            elif ct == "text/html" and not body:
                raw = part.get_payload(decode=True)
                if raw:
                    body = re.sub(r"<[^>]+>", " ", raw.decode(errors="ignore"))
    else:
        raw = msg.get_payload(decode=True)
        if raw:
            body = raw.decode(errors="ignore")
    return body[:5000]  # cap to avoid huge emails


# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────
st.markdown("<h1 style='margin-bottom:0'>📧 Smart Spam Detector</h1>", unsafe_allow_html=True)
st.caption("Accurate multi-signal spam & phishing analysis · Date-wise history · File upload support")

tab_scan, tab_upload, tab_history = st.tabs(["🔌 Live Gmail Scan", "📂 Upload Email File", "📅 History & Analytics"])


# ═══════════════════════════════════════════
# TAB 1 — LIVE GMAIL SCAN
# ═══════════════════════════════════════════
with tab_scan:
    st.markdown("### Connect your Gmail")
    st.info("Use a Gmail **App Password** (not your real password). Enable 2FA → Google Account → Security → App Passwords.")

    col1, col2 = st.columns(2)
    email_user = col1.text_input("📩 Gmail address", placeholder="you@gmail.com")
    email_pass = col2.text_input("🔑 App Password", type="password", placeholder="xxxx xxxx xxxx xxxx")
    num_emails = st.slider("Emails to analyse", 10, 200, 50, step=10)

    if st.button("🚀 Analyse Inbox", use_container_width=True):
        if not email_user or not email_pass:
            st.warning("Please enter both email and app password.")
        else:
            mail = None
            try:
                with st.spinner("Connecting to Gmail IMAP…"):
                    mail = imaplib.IMAP4_SSL("imap.gmail.com")
                    mail.login(email_user, email_pass)
                    mail.select("inbox")
                st.success("Connected ✓")

                status, messages = mail.search(None, "ALL")
                email_ids = messages[0].split()[-num_emails:]

                safe_count = spam_count = phish_count = 0
                results = []

                progress = st.progress(0, text="Fetching emails…")
                total_ids = len(email_ids)

                for idx, uid in enumerate(email_ids):
                    try:
                        uid_str = uid.decode()
                        exists = c.execute("SELECT 1 FROM emails WHERE uid=?", (uid_str,)).fetchone()
                        if exists:
                            continue

                        _, msg_data = mail.fetch(uid, "(RFC822)")
                        msg = email.message_from_bytes(msg_data[0][1])

                        subject = msg["subject"] or "(no subject)"
                        body    = extract_body(msg)

                        try:
                            email_date = parsedate_to_datetime(msg["Date"]).date()
                        except Exception:
                            email_date = date.today()

                        label, score = classify_email(subject, body)

                        if label == "Safe":     safe_count  += 1
                        elif label == "Spam":   spam_count  += 1
                        else:                   phish_count += 1

                        results.append((label, score, subject, str(email_date)))

                        c.execute(
                            "INSERT OR IGNORE INTO emails (uid, subject, label, score, email_date, source) VALUES (?,?,?,?,?,'imap')",
                            (uid_str, subject, label, score, str(email_date))
                        )
                        time.sleep(0.05)

                    except Exception:
                        continue

                    progress.progress((idx + 1) / total_ids, text=f"Processed {idx+1}/{total_ids}")

                conn.commit()
                progress.empty()

                # ── Summary metrics ──
                total_new = safe_count + spam_count + phish_count
                st.markdown("---")
                m1, m2, m3, m4 = st.columns(4)
                for col, label, val, color in [
                    (m1, "New Scanned", total_new, "#60a5fa"),
                    (m2, "✅ Safe",      safe_count, "#4ade80"),
                    (m3, "🚫 Spam",      spam_count, "#f87171"),
                    (m4, "⚠️ Phishing",  phish_count,"#fb923c"),
                ]:
                    col.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value" style="color:{color}">{val}</div>
                    </div>""", unsafe_allow_html=True)

                # ── Per-email results ──
                if results:
                    st.markdown("<div class='section-header'><h3>📋 Email Results</h3></div>", unsafe_allow_html=True)
                    with st.container(height=350):
                        for lbl, score, subj, edate in results:
                            row_cls  = "safe-row"  if lbl == "Safe"     else ("spam-row" if lbl == "Spam" else "phish-row")
                            badge_cls= "badge-safe" if lbl == "Safe"    else ("badge-spam" if lbl == "Spam" else "badge-phish")
                            icon     = "✅" if lbl == "Safe" else ("🚫" if lbl == "Spam" else "⚠️")
                            st.markdown(f"""
                            <div class="email-row {row_cls}">
                                <span>{icon}</span>
                                <span style="flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap">{subj}</span>
                                <span style="font-size:0.75rem;color:#6b7280">{edate}</span>
                                <span class="score-badge {badge_cls}">Score {score}</span>
                            </div>""", unsafe_allow_html=True)
                else:
                    st.info("No new emails found (all already in history).")

            except imaplib.IMAP4.error as e:
                st.error(f"❌ Login failed: {e}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
            finally:
                if mail:
                    try: mail.logout()
                    except: pass


# ═══════════════════════════════════════════
# TAB 2 — UPLOAD EMAIL FILE
# ═══════════════════════════════════════════
with tab_upload:
    st.markdown("### Upload & Analyse an Email File")
    st.caption("Supports `.eml` files (standard email format) and plain `.txt` files.")

    uploaded = st.file_uploader("Drop your email file here", type=["eml", "txt"])

    if uploaded:
        raw_bytes = uploaded.read()
        subject_up = body_up = ""

        if uploaded.name.endswith(".eml"):
            try:
                msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)
                subject_up = str(msg.get("subject", "(no subject)"))
                body_up    = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body_up = part.get_content()
                            break
                else:
                    body_up = msg.get_content()
                body_up = body_up or ""
            except Exception as e:
                st.error(f"Could not parse .eml: {e}")
        else:
            body_up    = raw_bytes.decode(errors="ignore")
            subject_up = uploaded.name

        if subject_up or body_up:
            st.markdown("#### Preview")
            st.markdown(f"**Subject:** {subject_up}")
            with st.expander("Show body snippet"):
                st.text(body_up[:1500])

            label_up, score_up = classify_email(subject_up, body_up)

            color_map = {"Safe": "#4ade80", "Spam": "#f87171", "Phishing": "#fb923c"}
            icon_map  = {"Safe": "✅", "Spam": "🚫", "Phishing": "⚠️"}

            st.markdown("---")
            r1, r2, r3 = st.columns(3)
            r1.markdown(f"""<div class="metric-card">
                <div class="metric-label">Result</div>
                <div class="metric-value" style="color:{color_map[label_up]}">{icon_map[label_up]} {label_up}</div>
            </div>""", unsafe_allow_html=True)
            r2.markdown(f"""<div class="metric-card">
                <div class="metric-label">Spam Score</div>
                <div class="metric-value" style="color:{color_map[label_up]}">{score_up} / 100</div>
            </div>""", unsafe_allow_html=True)

            # Accuracy indicator — show which signals fired
            fired = []
            combined_text = (subject_up + " " + body_up).lower()
            for kw, w in SPAM_KEYWORDS.items():
                if kw in combined_text:
                    fired.append(f"`{kw}` (+{w})")
            for pat in PHISHING_PATTERNS:
                if re.search(pat, combined_text):
                    fired.append(f"`regex: {pat[:30]}…` (+20)")

            r3.markdown(f"""<div class="metric-card">
                <div class="metric-label">Signals Found</div>
                <div class="metric-value" style="color:#60a5fa">{len(fired)}</div>
            </div>""", unsafe_allow_html=True)

            if fired:
                st.markdown("#### 🔍 Triggered Signals")
                st.markdown("  ".join(fired[:20]))

            # Save to DB
            uid_up = f"upload_{uploaded.name}_{int(time.time())}"
            c.execute(
                "INSERT OR IGNORE INTO emails (uid, subject, label, score, email_date, source) VALUES (?,?,?,?,?,'upload')",
                (uid_up, subject_up, label_up, score_up, str(date.today()))
            )
            conn.commit()
            st.success("Result saved to history ✓")

            # ── Accuracy Self-test ──
            st.markdown("---")
            st.markdown("#### 🎯 Accuracy Check — Was the result correct?")
            st.caption("Tell us the actual label to measure accuracy of the detector on your uploads.")

            user_label = st.radio("Actual label of this email:", ["Safe", "Spam", "Phishing"], horizontal=True, key="truth_radio")
            if st.button("Submit Feedback"):
                correct = (user_label == label_up)
                # Store feedback in a simple table
                c.execute("""CREATE TABLE IF NOT EXISTS feedback (
                    uid TEXT, predicted TEXT, actual TEXT, correct INTEGER
                )""")
                c.execute("INSERT INTO feedback VALUES (?,?,?,?)", (uid_up, label_up, user_label, int(correct)))
                conn.commit()

                total_fb = c.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
                correct_fb = c.execute("SELECT SUM(correct) FROM feedback").fetchone()[0] or 0
                acc = round(correct_fb / total_fb * 100, 1)

                if correct:
                    st.success(f"✅ Correct prediction! Cumulative accuracy: **{acc}%** over {total_fb} uploads.")
                else:
                    st.warning(f"❌ Mismatch — predicted **{label_up}**, actual **{user_label}**. Cumulative accuracy: **{acc}%** over {total_fb} uploads.")


# ═══════════════════════════════════════════
# TAB 3 — HISTORY & ANALYTICS
# ═══════════════════════════════════════════
with tab_history:
    st.markdown("### 📅 Date-wise Email History")

    # ── Date filter ──
    col_a, col_b, col_c = st.columns([2, 2, 2])
    filter_mode = col_a.selectbox("Filter by", ["All Dates", "Specific Date", "Date Range"])

    all_rows = pd.read_sql("""
        SELECT email_date, label, score, subject, source
        FROM emails
        ORDER BY email_date DESC
    """, conn)

    if all_rows.empty:
        st.info("No emails analysed yet. Use the Gmail Scan or Upload tab first.")
    else:
        all_rows["email_date"] = pd.to_datetime(all_rows["email_date"]).dt.date

        # Filtering
        if filter_mode == "Specific Date":
            avail_dates = sorted(all_rows["email_date"].unique(), reverse=True)
            chosen_date = col_b.selectbox("Pick a date", avail_dates)
            filtered = all_rows[all_rows["email_date"] == chosen_date]
        elif filter_mode == "Date Range":
            d_min = all_rows["email_date"].min()
            d_max = all_rows["email_date"].max()
            d_from = col_b.date_input("From", value=d_min, min_value=d_min, max_value=d_max)
            d_to   = col_c.date_input("To",   value=d_max, min_value=d_min, max_value=d_max)
            filtered = all_rows[(all_rows["email_date"] >= d_from) & (all_rows["email_date"] <= d_to)]
        else:
            filtered = all_rows

        # ── Aggregated summary table ──
        st.markdown("#### Summary by Date")
        summary = (
            filtered.groupby("email_date")
            .apply(lambda df: pd.Series({
                "Total":    len(df),
                "Safe":     (df["label"] == "Safe").sum(),
                "Spam":     (df["label"] == "Spam").sum(),
                "Phishing": (df["label"] == "Phishing").sum(),
                "Spam %":   round((df["label"].isin(["Spam","Phishing"])).mean() * 100, 1),
                "Avg Score":round(df["score"].mean(), 1),
            }))
            .reset_index()
            .rename(columns={"email_date": "Date"})
            .sort_values("Date", ascending=False)
        )
        summary["Date"] = summary["Date"].astype(str)

        st.dataframe(
            summary.style
                .background_gradient(subset=["Spam %"], cmap="RdYlGn_r")
                .format({"Spam %": "{:.1f}%", "Avg Score": "{:.1f}"}),
            use_container_width=True,
            hide_index=True,
        )

        # ── Bar chart ──
        st.markdown("#### Safe vs Spam/Phishing Over Time")
        chart_df = summary.set_index("Date")[["Safe", "Spam", "Phishing"]].sort_index()
        st.bar_chart(chart_df, color=["#4ade80", "#f87171", "#fb923c"])

        # ── Individual email list for selected/all dates ──
        st.markdown("#### Individual Emails")
        show_label = st.multiselect("Show labels", ["Safe", "Spam", "Phishing"],
                                    default=["Safe", "Spam", "Phishing"])
        detail = filtered[filtered["label"].isin(show_label)][
            ["email_date", "label", "score", "subject", "source"]
        ].sort_values("email_date", ascending=False)

        st.dataframe(
            detail.rename(columns={
                "email_date": "Date", "label": "Label",
                "score": "Score", "subject": "Subject", "source": "Source"
            }).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

        # ── Cumulative accuracy from feedback ──
        try:
            fb = pd.read_sql("SELECT predicted, actual, correct FROM feedback", conn)
            if not fb.empty:
                st.markdown("---")
                st.markdown("#### 🎯 Upload Accuracy (from your feedback)")
                total_fb   = len(fb)
                correct_fb = fb["correct"].sum()
                acc        = round(correct_fb / total_fb * 100, 1)

                fa, fb2, fc = st.columns(3)
                fa.metric("Total Feedback", total_fb)
                fb2.metric("Correct",       int(correct_fb))
                fc.metric("Accuracy",       f"{acc}%")

                conf = pd.crosstab(fb["actual"], fb["predicted"],
                                   rownames=["Actual"], colnames=["Predicted"])
                st.markdown("**Confusion Matrix**")
                st.dataframe(conf, use_container_width=True)
        except Exception:
            pass

        # ── Export ──
        st.markdown("---")
        csv_data = filtered.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Filtered History as CSV",
            data=csv_data,
            file_name=f"spam_history_{date.today()}.csv",
            mime="text/csv",
        )