# site.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Boundary Analysis",
    layout="wide",
    page_icon="🏏"
)

# -------------------------------
# Header Image (safe load)
# -------------------------------
header_path = "assets/header.png"
if os.path.exists(header_path):
    st.image(header_path, use_container_width=True)
else:
    st.warning("Header image not found! Place a header.jpg in assets/ folder.")

# -------------------------------
# Custom header
# -------------------------------
st.markdown(
    """
    <div style='text-align:center;'>
        <h1 style='color:#FF6F61; font-size:48px;'>🏏 Boundary Analysis Dashboard</h1>
        <p style='color:#FFD700; font-size:18px;'>Analyze what happens after a 4 or 6 and view yearly trends</p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Load & preprocess data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("IPL.csv")

    # Convert date
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df["year"] = df["date"].dt.year

    # Sort
    df = df.sort_values(["match_id", "innings", "over", "ball_no"]).reset_index(drop=True)

    # Last ball of innings
    last_ball = df.groupby(["match_id", "innings"])["ball_no"].max().reset_index()
    last_ball = last_ball.rename(columns={"ball_no": "last_ball_no"})
    df = df.merge(last_ball, on=["match_id", "innings"], how="left")

    # Boundaries
    df["is_boundary"] = (df["valid_ball"] == 1) & (df["runs_batter"].isin([4, 6]))
    df["next_runs_total"] = df.groupby(["match_id", "innings"])["runs_total"].shift(-1)
    df["next_extra_type"] = df.groupby(["match_id", "innings"])["extra_type"].shift(-1)

    boundary_df = df[(df["is_boundary"]) & (df["ball_no"] != df["last_ball_no"])].copy()

    # Next ball classification
    def classify_next_ball(row):
        runs = row["next_runs_total"]
        extra = row["next_extra_type"]
        if pd.isna(runs):
            return None
        if runs in [0, 1, 2, 3, 4, 6] and (pd.isna(extra) or extra == ""):
            return str(int(runs))
        else:
            return "Other"

    boundary_df["next_ball_outcome"] = boundary_df.apply(classify_next_ball, axis=1)

    # Summary for bar chart
    summary = (
        boundary_df
        .groupby(["year", "runs_batter", "next_ball_outcome"])
        .size()
        .reset_index(name="count")
    )
    summary["percentage"] = summary.groupby(["year", "runs_batter"])["count"].transform(
        lambda x: x / x.sum() * 100
    )

    return df, summary, boundary_df

df, summary, boundary_df = load_data()

# -------------------------------
# Sidebar filters
# -------------------------------
st.sidebar.header("Filters")
years = sorted(summary["year"].dropna().unique())

# Option to Compare or Single Year
mode = st.sidebar.radio("Mode", ["Single Year", "Compare Years"])

# -------------------------------
# Function for metric card
# -------------------------------
def metric_card(title, value, color="#FF6F61"):
    st.markdown(
        f"""
        <div style='
            background: {color};
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align:center;
            font-size:20px;
            margin-bottom:10px;
            box-shadow: 2px 4px 8px rgba(0,0,0,0.4);
        '>
            <strong>{title}</strong><br>
            <span style='font-size:28px;'>{value}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# SINGLE YEAR VIEW
# -------------------------------
if mode == "Single Year":
    selected_year = st.sidebar.slider(
        "Select Year",
        min_value=int(min(years)),
        max_value=int(max(years)),
        value=int(max(years))
    )

    boundary_type = st.sidebar.radio(
        "Boundary Type",
        options=[4, 6],
        horizontal=True
    )

    # Filter data for bar chart
    filtered_df = summary[
        (summary["year"] == selected_year) &
        (summary["runs_batter"] == boundary_type)
    ]
    order = ["0", "1", "2", "3", "4", "6", "Other"]
    filtered_df["next_ball_outcome"] = pd.Categorical(
        filtered_df["next_ball_outcome"], categories=order, ordered=True
    )
    filtered_df = filtered_df.sort_values("next_ball_outcome")

    # Layout: 2 columns
    col1, col2 = st.columns([2,1])

    with col1:
        st.subheader(f"Next Ball Outcome after a {boundary_type} in {selected_year}")

        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(8,5))
        bars = ax.bar(filtered_df["next_ball_outcome"], filtered_df["percentage"], color="#FF6F61", alpha=0.85)
        
        # Add % labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0,3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel("Next Ball Outcome")
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0, filtered_df["percentage"].max()+10)
        st.pyplot(fig)

    with col2:
        st.subheader(f"Total 4s & 6s in {selected_year}")

        year_df = df[(df["year"] == selected_year) & (df["valid_ball"]==1)]
        total_4s = year_df[year_df["runs_batter"]==4].shape[0]
        total_6s = year_df[year_df["runs_batter"]==6].shape[0]

        metric_card("Total 4s", total_4s, color="#22c55e")   # Green
        metric_card("Total 6s", total_6s, color="#f97316")   # Orange

# -------------------------------
# COMPARE YEARS VIEW
# -------------------------------
else:
    st.sidebar.subheader("Select Two Years")
    year1 = st.sidebar.selectbox("Year 1", years, index=len(years)-2)
    year2 = st.sidebar.selectbox("Year 2", years, index=len(years)-1)

    boundary_type = st.sidebar.radio(
        "Boundary Type",
        options=[4, 6],
        horizontal=True
    )

    # Filter summary for both years
    df1 = summary[(summary["year"]==year1) & (summary["runs_batter"]==boundary_type)]
    df2 = summary[(summary["year"]==year2) & (summary["runs_batter"]==boundary_type)]

    order = ["0", "1", "2", "3", "4", "6", "Other"]
    df1["next_ball_outcome"] = pd.Categorical(df1["next_ball_outcome"], categories=order, ordered=True)
    df2["next_ball_outcome"] = pd.Categorical(df2["next_ball_outcome"], categories=order, ordered=True)
    df1 = df1.sort_values("next_ball_outcome")
    df2 = df2.sort_values("next_ball_outcome")

    col1, col2 = st.columns(2)

    # Bar charts for each year
    with col1:
        st.subheader(f"{year1} - Next Ball Outcome")
        fig, ax = plt.subplots(figsize=(6,4))
        bars = ax.bar(df1["next_ball_outcome"], df1["percentage"], color="#22c55e", alpha=0.85)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0,3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, max(df1["percentage"].max(), df2["percentage"].max())+10)
        st.pyplot(fig)

    with col2:
        st.subheader(f"{year2} - Next Ball Outcome")
        fig, ax = plt.subplots(figsize=(6,4))
        bars = ax.bar(df2["next_ball_outcome"], df2["percentage"], color="#f97316", alpha=0.85)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0,3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, max(df1["percentage"].max(), df2["percentage"].max())+10)
        st.pyplot(fig)

    # Totals cards for both years
    st.subheader(f"Total 4s & 6s in {year1} vs {year2}")

    col1, col2, col3, col4 = st.columns(4)
    y1_df = df[(df["year"]==year1) & (df["valid_ball"]==1)]
    y2_df = df[(df["year"]==year2) & (df["valid_ball"]==1)]

    metric_card("4s in " + str(year1), y1_df[y1_df["runs_batter"]==4].shape[0], color="#22c55e")
    metric_card("6s in " + str(year1), y1_df[y1_df["runs_batter"]==6].shape[0], color="#f97316")
    metric_card("4s in " + str(year2), y2_df[y2_df["runs_batter"]==4].shape[0], color="#22c55e")
    metric_card("6s in " + str(year2), y2_df[y2_df["runs_batter"]==6].shape[0], color="#f97316")
