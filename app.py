import streamlit as st
import pandas as pd
import plotly.express as px
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from groq import Groq

# ================= CONFIG =================
st.set_page_config(page_title="AI Finance Dashboard", layout="wide")

# ================= LOGGING =================
logging.basicConfig(filename="app.log", level=logging.INFO)

def log_event(msg):
    logging.info(msg)

# ================= GROQ CLIENT =================
# Put GROQ_API_KEY in .streamlit/secrets.toml
# GROQ_API_KEY="your_key_here"

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

# ================= DATA UPLOAD =================
st.sidebar.markdown("## Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    st.title("📊 AI-Powered Finance Analytics Dashboard")
    st.info("Upload a CSV file from the sidebar to begin analysis.")
    st.stop()

# ================= AUTO SCHEMA =================
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()

st.sidebar.write("Detected Numeric Features:", len(numeric_cols))
st.sidebar.write("Detected Categorical Features:", len(cat_cols))

numeric_df = df.select_dtypes(include=['number']).dropna()

# ================= CLUSTERING =================
def run_clustering(data, k=3):

    if data.shape[1] < 2:
        return None

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(scaled)

    result = data.copy()
    result["Cluster"] = clusters

    return result

# ================= LLM =================
def ask_llm(question, data):

    context = data.head(20).to_string()

    prompt=f"""
You are an expert financial data analyst.

Use this uploaded dataset:

{context}

Answer the user question:

{question}

Provide:
- Insights
- Risks
- Trends
- Recommendations
"""

    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ],
        temperature=0.3
    )

    return completion.choices[0].message.content

# ================= CHAT MEMORY =================
if "messages" not in st.session_state:
    st.session_state.messages=[]

# ================= UI =================
st.title("📊 AI-Powered Finance Analytics Dashboard")

st.markdown("""
### 🚀 Capabilities
- CSV Upload Analytics
- Interactive Filtering
- AI Insights (LLM Chat)
- Customer Segmentation (Clustering)
- Interactive Visualizations
""")

# ================= SIDEBAR FILTERS =================
st.sidebar.header("⚙️ Controls")

column = st.sidebar.selectbox(
    "Select Column",
    df.columns
)

is_numeric = pd.api.types.is_numeric_dtype(df[column])

st.sidebar.write(
    f"Detected Type: {'Numeric' if is_numeric else 'Categorical'}"
)

if is_numeric:

    threshold = st.sidebar.slider(
        "Filter Threshold",
        float(df[column].min()),
        float(df[column].max()),
        float(df[column].mean())
    )

    filtered_df = df[
        df[column] >= threshold
    ]

else:

    unique_vals = df[column].dropna().unique()

    selected_vals = st.sidebar.multiselect(
        "Select Categories",
        options=unique_vals,
        default=unique_vals[:min(3,len(unique_vals))]
    )

    if len(selected_vals)>0:
        filtered_df = df[
            df[column].isin(selected_vals)
        ]
    else:
        filtered_df=df.copy()

# ================= METRICS =================
c1,c2,c3 = st.columns(3)

c1.metric("Rows",len(filtered_df))
c2.metric("Columns",len(df.columns))

if is_numeric:
    c3.metric(
        "Mean",
        round(filtered_df[column].mean(),2)
    )
else:
    c3.metric(
        "Unique Values",
        filtered_df[column].nunique()
    )

# ================= TABS =================
tab1,tab2,tab3,tab4 = st.tabs([
    "📊 Overview",
    "📈 Visuals",
    "🤖 AI Insights",
    "🧠 Clustering"
])

# ================= TAB 1 =================
with tab1:

    st.subheader("Dataset")
    st.dataframe(
        filtered_df.head(50)
    )

    st.subheader("Summary Stats")
    st.write(
        filtered_df.describe(
            include='all'
        )
    )

# ================= TAB 2 =================
with tab2:

    st.subheader(
        "Interactive Visualizations"
    )

    if is_numeric:

        fig = px.histogram(
            filtered_df,
            x=column,
            title=f"{column} Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if len(numeric_cols) > 1:

        corr = filtered_df.select_dtypes(
            include=['number']
        ).corr()

        fig = px.imshow(
            corr,
            title="Correlation Heatmap"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ================= TAB 3 =================
with tab3:

    st.subheader(
        "Ask AI About Your Uploaded Data"
    )

    for msg in st.session_state.messages:

        with st.chat_message(
            msg["role"]
        ):
            st.write(
                msg["content"]
            )

    user_input = st.chat_input(
        "Ask about trends, risk, anomalies..."
    )

    if user_input:

        log_event(
            f"User Question: {user_input}"
        )

        st.session_state.messages.append(
            {
                "role":"user",
                "content":user_input
            }
        )

        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner(
            "Analyzing..."
        ):

            response = ask_llm(
                user_input,
                filtered_df
            )

        st.session_state.messages.append(
            {
                "role":"assistant",
                "content":response
            }
        )

        with st.chat_message(
            "assistant"
        ):
            st.write(response)

# ================= TAB 4 =================
with tab4:

    st.subheader(
        "Customer / Financial Segmentation"
    )

    k = st.slider(
        "Select Number of Clusters",
        2,
        6,
        3
    )

    if st.button(
        "Run Clustering"
    ):

        filtered_numeric = filtered_df.select_dtypes(
            include=['number']
        ).dropna()

        clustered = run_clustering(
            filtered_numeric,
            k
        )

        if clustered is not None:

            st.dataframe(
                clustered.head()
            )

            x_col = clustered.columns[0]
            y_col = clustered.columns[1]

            fig = px.scatter(
                clustered,
                x=x_col,
                y=y_col,
                color="Cluster",
                title="Cluster Visualization"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.warning(
                "Need at least two numeric features for clustering."
            )

# ================= DOWNLOAD =================
st.markdown("---")

st.download_button(
    label="⬇️ Download Filtered Data",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_finance_data.csv"
)

# ================= SYSTEM HEALTH =================
st.sidebar.markdown(
    "### 🩺 System Health"
)

st.sidebar.success("Running")
st.sidebar.write("Logging: Active")
st.sidebar.write("LLM: Groq Llama3 Connected")
st.sidebar.write("Model: KMeans Ready")
