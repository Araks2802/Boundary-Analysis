# site.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Boundary Analysis",
    layout="wide",
    page_icon="🏏"
)

# -------------------------------
# Sidebar Header Image
# -------------------------------
header_path = "assets/headers.png"
if os.path.exists(header_path):
    img = Image.open(header_path)
    img = img.resize((img.width, 200))  # adjust height
    st.sidebar.image(img, use_container_width=True)
else:
    st.sidebar.warning("⚠️ Header image not found. Place header.png inside assets/")

# -------------------------------
# Main Title
# -------------------------------
st.markdown(
    """
    <div style='text-align:center; margin-top:10px;'>
        <h1 style='color:#FF6F61;'>🏏 Boundary Analysis Dashboard</h1>
        <p style='color:#666; font-size:18px;'>
        What happens after boundaries in the IPL
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
st.divider()

# -------------------------------
# Load & preprocess data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("IPL_small.csv")

    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df["year"] = df["date"].dt.year

    df = df.sort_values(
        ["match_id", "innings", "over", "ball_no"]
    ).reset_index(drop=True)

    last_ball = df.groupby(["match_id", "innings"])["ball_no"].max().reset_index(name="last_ball_no")
    df = df.merge(last_ball, on=["match_id", "innings"], how="left")

    df["is_boundary"] = (df["valid_ball"] == 1) & (df["runs_batter"].isin([4, 6]))
    df["next_runs_total"] = df.groupby(["match_id", "innings"])["runs_total"].shift(-1)
    df["next_extra_type"] = df.groupby(["match_id", "innings"])["extra_type"].shift(-1)

    boundary_df = df[(df["is_boundary"]) & (df["ball_no"] != df["last_ball_no"])].copy()

    def classify_next_ball(row):
        if pd.isna(row["next_runs_total"]):
            return None
        if (row["next_runs_total"] in [0,1,2,3,4,6]) and (pd.isna(row["next_extra_type"]) or row["next_extra_type"]==""):
            return str(int(row["next_runs_total"]))
        return "Other"

    boundary_df["next_ball_outcome"] = boundary_df.apply(classify_next_ball, axis=1)

    # Summary table for bar charts
    summary = boundary_df.groupby(["year","runs_batter","next_ball_outcome"]).size().reset_index(name="count")
    summary["percentage"] = summary.groupby(["year","runs_batter"])["count"].transform(lambda x: x/x.sum()*100)

    # Extra metrics
    dot_ball_summary = boundary_df.groupby(["year","runs_batter"])\
                        .apply(lambda x: (x["next_ball_outcome"]=="0").sum()/x.shape[0]*100)\
                        .reset_index(name="dot_ball_pct")
    
    avg_next3 = df.groupby(["year","innings"])["runs_total"].rolling(3).sum().shift(-1).reset_index(name="next3_sum")
    
    return df, summary, boundary_df, dot_ball_summary, avg_next3

df, summary, boundary_df, dot_ball_summary, avg_next3 = load_data()

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.title("🎛 Controls")
years = sorted(summary["year"].dropna().unique())
mode = st.sidebar.radio("View Mode", ["Single Year","Compare Years"])
ORDER = ["0","1","2","3","4","6","Other"]

# -------------------------------
# Metric Card
# -------------------------------
def metric_card(title,value,subtitle,gradient):
    st.markdown(
        f"""
        <div style="
            background:{gradient};
            padding:20px;
            border-radius:16px;
            color:white;
            text-align:center;
            box-shadow:0 6px 12px rgba(0,0,0,0.25);
        ">
            <div style="font-size:14px; opacity:0.9;">{subtitle}</div>
            <div style="font-size:32px; font-weight:700; margin:6px 0;">{value}</div>
            <div style="font-size:15px; opacity:0.85;">{title}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ===============================
# SINGLE YEAR VIEW
# ===============================
if mode=="Single Year":
    year = st.sidebar.slider("Select Year", int(min(years)), int(max(years)), int(max(years)))
    boundary = st.sidebar.radio("Boundary", [4,6], horizontal=True)

    data = summary[(summary["year"]==year)&(summary["runs_batter"]==boundary)].copy()
    data["next_ball_outcome"] = pd.Categorical(data["next_ball_outcome"], ORDER, ordered=True)
    data.sort_values("next_ball_outcome", inplace=True)

    # Chart and cards layout
    col1, col2 = st.columns([2.2,1])

    # ---- Bar Chart
    with col1:
        st.subheader(f"📊 Next Ball Outcome after a {boundary} in {year}")
        fig, ax = plt.subplots(figsize=(8,5))
        bars = ax.bar(data["next_ball_outcome"], data["percentage"], color="#FF6F61")
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f"{b.get_height():.1f}%",ha="center",fontsize=10)
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel("Next Ball Outcome")
        ax.set_ylim(0, data["percentage"].max()+10)
        st.pyplot(fig)

    # ---- Cards
    with col2:
        year_df = df[(df["year"]==year)&(df["valid_ball"]==1)]
        c1,c2 = st.columns(2)
        with c1:
            metric_card("Total Fours",year_df[year_df["runs_batter"]==4].shape[0],"🏏 Boundaries","linear-gradient(135deg,#22c55e,#16a34a)")
        with c2:
            metric_card("Total Sixes",year_df[year_df["runs_batter"]==6].shape[0],"🚀 Maximums","linear-gradient(135deg,#f97316,#ea580c)")

        # ---- Extra Metrics
        st.markdown("### ⚡ Additional Insights")
        c3,c4, c5 = st.columns(3)
        with c3:
            total_matches = year_df["match_id"].nunique()
            metric_card(
                "Total Matches",
                total_matches,
                "Season matches",
                "linear-gradient(135deg,#6366f1,#4f46e5)"
            )

        with c4:
            total_sixes = year_df[year_df["runs_batter"]==6].shape[0]
            avg_sixes = total_sixes / total_matches if total_matches else 0
            metric_card(
                "Avg Sixes per Match",
                f"{avg_sixes:.1f}",
                "Six-hitting rate",
                "linear-gradient(135deg,#ec4899,#db2777)"
            )
        with c5:
            next3_avg = avg_next3[avg_next3["year"]==year]["next3_sum"].mean()
            metric_card("Avg Runs Next 3 Balls",f"{next3_avg:.1f}","Rolling 3 Balls","linear-gradient(135deg,#facc15,#eab308)")

    # ---- Outcome Heatmap
    st.subheader(f"📈 Next Ball Outcome Heatmap for {year}")
    heatmap_data = data.pivot(index="runs_batter", columns="next_ball_outcome", values="percentage").fillna(0)
    fig, ax = plt.subplots(figsize=(8,2))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="coolwarm", cbar_kws={"label":"%"} , ax=ax)
    st.pyplot(fig)

    # ---- Trend Analysis
    st.subheader("📊 Trend of Boundaries Over Years")
    trend_data = df[df["valid_ball"]==1].groupby(["year","runs_batter"]).size().reset_index(name="count")
    trend_data = trend_data[trend_data["runs_batter"].isin([4,6])]
    fig, ax = plt.subplots(figsize=(10,4))
    sns.lineplot(data=trend_data,x="year",y="count",hue="runs_batter",palette={4:"#22c55e",6:"#f97316"},marker="o",ax=ax)
    ax.set_ylabel("Total Boundaries")
    ax.set_xlabel("Year")
    # Force years to be integers
    ax.set_xticks(trend_data["year"].unique())  # only show actual years
    ax.set_xticklabels(trend_data["year"].unique().astype(int))  # make them whole numbers
    st.pyplot(fig)

# ===============================
# COMPARE YEARS VIEW
# ===============================
else:
    year1 = st.sidebar.selectbox("Year 1", years, index=len(years)-2)
    year2 = st.sidebar.selectbox("Year 2", years, index=len(years)-1)
    boundary = st.sidebar.radio("Boundary",[4,6],horizontal=True)

    d1 = summary[(summary["year"]==year1)&(summary["runs_batter"]==boundary)].copy()
    d2 = summary[(summary["year"]==year2)&(summary["runs_batter"]==boundary)].copy()
    for d in [d1,d2]:
        d["next_ball_outcome"] = pd.Categorical(d["next_ball_outcome"], ORDER, ordered=True)
        d.sort_values("next_ball_outcome", inplace=True)

    # Charts
    c1,c2 = st.columns(2)
    with c1:
        st.subheader(f"{year1} – After a {boundary}")
        fig, ax = plt.subplots(figsize=(6,4))
        bars = ax.bar(d1["next_ball_outcome"],d1["percentage"],color="#22c55e")
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f"{b.get_height():.1f}%",ha="center",fontsize=9)
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0,max(d1["percentage"].max(),d2["percentage"].max())+10)
        st.pyplot(fig)
    with c2:
        st.subheader(f"{year2} – After a {boundary}")
        fig, ax = plt.subplots(figsize=(6,4))
        bars = ax.bar(d2["next_ball_outcome"],d2["percentage"],color="#f97316")
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f"{b.get_height():.1f}%",ha="center",fontsize=9)
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0,max(d1["percentage"].max(),d2["percentage"].max())+10)
        st.pyplot(fig)

    # ---- Cards
    st.subheader("🏏 Boundary Count Comparison")
    y1 = df[(df["year"]==year1)&(df["valid_ball"]==1)]
    y2 = df[(df["year"]==year2)&(df["valid_ball"]==1)]
    colA,colB = st.columns(2)
    with colA:
        st.markdown(f"### {year1}")
        c1,c2 = st.columns(2)
        with c1: metric_card("Total Fours",y1[y1["runs_batter"]==4].shape[0],"🏏 Boundaries","linear-gradient(135deg,#22c55e,#16a34a)")
        with c2: metric_card("Total Sixes",y1[y1["runs_batter"]==6].shape[0],"🚀 Maximums","linear-gradient(135deg,#22c55e,#16a34a)")
    with colB:
        st.markdown(f"### {year2}")
        c1,c2 = st.columns(2)
        with c1: metric_card("Total Fours",y2[y2["runs_batter"]==4].shape[0],"🏏 Boundaries","linear-gradient(135deg,#f97316,#ea580c)")
        with c2: metric_card("Total Sixes",y2[y2["runs_batter"]==6].shape[0],"🚀 Maximums","linear-gradient(135deg,#f97316,#ea580c)")

