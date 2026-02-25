import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Page Config

st.set_page_config(
    page_title="Tourism Recommendation System",
    page_icon="🌍",
    layout="wide"
)


# THEME SYSTEM

if "theme" not in st.session_state:
    st.session_state.theme = "light"

col_title, col_toggle = st.columns([9, 1])

with col_title:
    st.title("🌍 Tourism Recommendation System")

with col_toggle:
    if st.button("☀" if st.session_state.theme == "light" else "☾"):
        st.session_state.theme = (
            "dark" if st.session_state.theme == "light" else "light"
        )
        st.rerun()


# THEME STYLING

if st.session_state.theme == "dark":
    st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    h1, h2, h3, h4, h5, h6, p, label, div {
        color: white !important;
    }
    div[data-testid="stMetric"] {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 12px;
    }
    
    /* USER ID WIDTH */
    div[data-testid="stNumberInput"] {
        max-width: 300px;
    }

    /* SLIDER WIDTH */
    div[data-testid="stSlider"] {
        max-width: 500px;
    }

    table {
        background-color: #111827 !important;
        color: white !important;
    }
    th {
        background-color: #1f2937 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
        color: #fffefe;
    }
    h1, h2, h3, h4, h5, h6, p, label, div {
        color: #111111 !important;
    }
    div[data-testid="stMetric"] {
        background-color: #e5e7eb;
        padding: 15px;
        border-radius: 12px;
    }
                
    div[data-testid="stButton"] > button {    
    background-color: #ffffff !important;
    color: #111111 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    }

    div[data-testid="stButton"] > button:hover {
        background-color: #e5e7eb !important;
    }
                
    table {
        background-color: #ffffff !important;
        color: #111111 !important;
        border: 1px solid #e5e7eb !important;
    }
                
    th, td {
        border: 0.05px solid black !important;  /* border for headers and cells */
    }
                
    th {
        background-color: #e5e7eb !important;
        color: #111111 !important;
    }
    tr:nth-child(even) {
        background-color: #e5e7eb !important;
    }
                
    /* USER ID WIDTH */
    div[data-testid="stNumberInput"] {
        max-width: 300px;
    }

    /* Number input main background */
    div[data-testid="stNumberInput"] [data-baseweb="input"] {
        background-color: #e5e7eb !important;
    }

    /* The actual input field */
    div[data-testid="stNumberInput"] input {
        background-color: #e5e7eb !important;
        color: black !important;
    }

    /* SLIDER WIDTH */
    div[data-testid="stSlider"] {
        max-width: 500px;
    }


    </style>
    """, unsafe_allow_html=True)

st.markdown("Popular • Collaborative • Content-Based • Hybrid")
st.divider()


# Load Dataset

@st.cache_data
def load_data():
    return pd.read_csv("Tourism_Analysis_data.csv")

df = load_data()

REQUIRED_COLUMNS = [
    "UserId", "Attraction", "Rating",
    "Region", "CityName",
    "AttractionType", "VisitMode", "Country"
]

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

categorical_cols = ["Country", "Region", "CityName", "AttractionType"]
for col in categorical_cols:
    df[col] = df[col].fillna("Unknown").astype(str)

meta = df[['Attraction', 'Region', 'CityName', 'Country']].drop_duplicates()


# KPI Dashboard

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Users", df["UserId"].nunique())
col2.metric("Total Attractions", df["Attraction"].nunique())
col3.metric("Total Ratings", len(df))
col4.metric("Average Rating", round(df["Rating"].mean(), 2))

st.divider()


# Popular Recommendation

def get_popular(df):
    popular = (
        df.groupby(['Attraction', 'Region', 'Country', 'CityName'])
        .agg(no_of_ratings=('Rating', 'count'),
             avg_rating=('Rating', 'mean'))
        .reset_index()
    )
    popular = popular[
        (popular['no_of_ratings'] > 50) &
        (popular['avg_rating'] > 4.0)
    ]
    popular = popular.sort_values(
        ['avg_rating', 'no_of_ratings'],
        ascending=[False, False]
    ).head(20)

    popular.reset_index(drop=True, inplace=True)
    popular.index += 1
    return popular


# Build Models

@st.cache_resource
def build_models(df):

    user_item = df.pivot_table(
        index="UserId",
        columns="Attraction",
        values="Rating",
        aggfunc="mean"
    ).fillna(0)

    svd = TruncatedSVD(n_components=20, random_state=42)
    user_latent = svd.fit_transform(user_item)
    item_latent = svd.components_

    cf_scores = pd.DataFrame(
        np.dot(user_latent, item_latent),
        index=user_item.index,
        columns=user_item.columns
    )

    item_similarity = pd.DataFrame(
        cosine_similarity(item_latent.T),
        index=user_item.columns,
        columns=user_item.columns
    )

    attraction_features = (
        df[['Attraction', 'AttractionType', 'Region']]
        .drop_duplicates()
    )

    attraction_features = pd.get_dummies(
        attraction_features,
        columns=['AttractionType', 'Region']
    ).groupby("Attraction").mean()

    user_profile = user_item.dot(attraction_features)

    content_scores = pd.DataFrame(
        user_profile.dot(attraction_features.T),
        index=user_item.index,
        columns=attraction_features.index
    )

    content_scores = content_scores.reindex(
        columns=user_item.columns,
        fill_value=0
    )

    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    cf_norm = normalize(cf_scores)
    content_norm = normalize(content_scores)

    hybrid_scores = 0.7 * cf_norm + 0.3 * content_norm

    return user_item, cf_norm, content_norm, hybrid_scores, item_similarity

if "models" not in st.session_state:
    with st.spinner("Building models..."):
        st.session_state.models = build_models(df)

user_item, cf_scores, content_scores, hybrid_scores, item_similarity = st.session_state.models


# Recommendation Function

def recommend(user_id, model_type, top_k):

    if user_id not in user_item.index:
        return get_popular(df)

    history = user_item.loc[user_id]

    if model_type == "Collaborative":
        scores = cf_scores.loc[user_id].copy()

    elif model_type == "Content":
        scores = content_scores.loc[user_id].copy()

    else:
        scores = hybrid_scores.loc[user_id].copy()

    scores[history > 0] = 0

    recs = scores.sort_values(ascending=False)

    results = pd.DataFrame({
        "Attraction": recs.index,
        "Score": recs.values
    }).merge(meta, on="Attraction", how="left")

    results["Score"] = results["Score"].round(3)
    results.index = np.arange(1, len(results) + 1)

    return results.head(top_k)


# Tabs

tab1, tab2, tab3, tab4 = st.tabs([
    "🔥 Popular",
    "🤝 Collaborative",
    "🧠 Content-Based",
    "⚡ Hybrid"
])

with tab1:
    st.subheader("🔥 Most Popular Attractions")
    st.table(get_popular(df))

for tab, model_name in zip(
    [tab2, tab3, tab4],
    ["Collaborative", "Content", "Hybrid"]
):
    with tab:

        st.subheader(f"{model_name} Recommendation")

        user_id = st.number_input(
            "Enter User ID",
            min_value=int(df["UserId"].min()),
            max_value=int(df["UserId"].max()),
            step=1,
            key=model_name
        )

        top_k = st.slider(
            "Number of Recommendations",
            5, 30, 15,
            key=f"{model_name}_slider"
        )

        if st.button("Generate Recommendations", key=f"{model_name}_btn"):

            with st.spinner("Generating recommendations..."):

                results = recommend(user_id, model_name, top_k)

                st.table(results)
