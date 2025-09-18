import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
import os 

st.set_page_config(page_title="Nowcast Dashboard", layout="wide")

# ---- sidebar inputs ----
fc_path = st.sidebar.text_input("Forecast CSV path", "nowcast_forecasts_xgb_all_iso.csv")
mt_path = st.sidebar.text_input("Metrics CSV path",  "nowcast_metrics_xgb_all_iso.csv")
load_path = st.sidebar.text_input("PC Loadings CSV path","all_countries_pcs_loadings_long.csv")
pcs_path = st.sidebar.text_input("PC Scores CSV path","all_countries_pcs_long.csv")

@st.cache_data
def load_pca_tables(load_path: str, pcs_path: str):
    loadings = None
    pcs = None
    try:
        loadings = pd.read_csv(load_path)
    except Exception as e:
        st.sidebar.warning(f"Loadings not loaded: {e}")
    try:
        pcs = pd.read_csv(pcs_path)
        if "date" in pcs.columns:
            # ensure datetime for plotting
            pcs["date"] = pd.to_datetime(pcs["date"], errors="coerce")
    except Exception as e:
        st.sidebar.warning(f"PC scores not loaded: {e}")
    return loadings, pcs


@st.cache_data
def load_data(fc_path, mt_path):
    fc = pd.read_csv(fc_path)
    mt = pd.read_csv(mt_path)
    if "target_month" in fc.columns:
        fc["target_month"] = pd.to_datetime(fc["target_month"], format="%Y-%m", errors="coerce")
    return fc, mt

@st.cache_data
def load_pcs(pcs_path: str):
    if not pcs_path or not os.path.exists(pcs_path):
        return pd.DataFrame()
    df = pd.read_csv(pcs_path)
    # Normalize date column name and type
    date_col = None
    for c in ["date", "month", "time", "period"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        # try to parse index if no date column
        if df.index.name:
            df = df.reset_index().rename(columns={df.index.name: "date"})
            date_col = "date"
        else:
            return pd.DataFrame()  # can't parse
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.rename(columns={date_col: "date"})
    # Detect long vs wide
    is_long = {"country", "date", "PC", "score"}.issubset(set(df.columns))
    if is_long:
        pcs_long = df[["country", "date", "PC", "score"]].copy()
    else:
        # assume wide: country, date, PC1..PCk
        non_pc = {"country", "date"}
        pc_cols = [c for c in df.columns if c not in non_pc]
        if not pc_cols:
            return pd.DataFrame()
        pcs_long = (
            df.melt(["country", "date"], var_name="PC", value_name="score")
        )
    # keep clean
    pcs_long = pcs_long.dropna(subset=["country", "date", "PC", "score"])
    # order PCs by the number in the name if present
    try:
        pcs_long["__pc_order__"] = pcs_long["PC"].str.extract(r"(\d+)").astype(float)
    except Exception:
        pcs_long["__pc_order__"] = np.inf
    return pcs_long

    
try:
    fc, mt = load_data(fc_path, mt_path)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

if fc.empty:
    st.warning("Forecasts file is empty.")
    st.stop()

# infer actual column if any
actual_col = None
for cand in ["actual", "y_true", "truth"]:
    if cand in fc.columns:
        actual_col = cand
        break

st.sidebar.markdown("---")
countries = sorted(fc["country"].astype(str).unique().tolist())
models    = sorted(fc["model"].astype(str).unique().tolist())
cty_sel   = st.sidebar.multiselect("Countries", countries, default=countries[:3])
mdl_sel   = st.sidebar.multiselect("Models", models, default=models[:1])
hz_options = ["All"] + sorted(fc["horizon"].dropna().unique().tolist(), key=lambda x: (isinstance(x, str), x))
hz_sel    = st.sidebar.selectbox("Horizon", options=hz_options, index=0)

# filter
df = fc[fc["country"].astype(str).isin(cty_sel) & fc["model"].astype(str).isin(mdl_sel)].copy()
if hz_sel != "All":
    df = df[df["horizon"]==hz_sel]
df = df.sort_values(["country", "model", "target_month"])

# ===== top: time series =====
st.header("Forecasts over time (YoY %)")

for (cty, mdl), g in df.groupby(["country","model"]):
    g = g.sort_values("target_month")

    fig = plt.figure(figsize=(9,3.6))
    # uncertainty band if available
    if {"p10","p90"}.issubset(g.columns):
        plt.fill_between(g["target_month"], g["p10"], g["p90"], alpha=0.15, label="p10–p90")
    # central forecast
    if "p50" in g.columns:
        plt.plot(g["target_month"], g["p50"], label="Forecast (p50)")
    elif "point" in g.columns:
        plt.plot(g["target_month"], g["point"], label="Forecast (point)")

    # actual if available
    if actual_col and actual_col in g.columns:
        plt.plot(g["target_month"], g[actual_col], label="Actual")

    plt.title(f"{cty} • {mdl} • Horizon: {hz_sel}")
    plt.xlabel("Month")
    plt.ylabel("YoY (%)")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

# ===== metrics =====
# st.header("Model metrics")
# if mt.empty:
#     st.info("No metrics file loaded.")
# else:
#     mview = mt.copy()
#     if mdl_sel:
#         mview = mview[mview["model"].astype(str).isin(mdl_sel)]
#     if cty_sel:
#         mview = mview[mview["country"].astype(str).isin(cty_sel)]
#     st.dataframe(mview.reset_index(drop=True))

#     # bar: RMSE by country for each model selected
#     for mdl in mview["model"].astype(str).unique():
#         mm = mview[mview["model"].astype(str)==mdl].copy().sort_values("rmse")
#         if not mm.empty:
#             fig = plt.figure(figsize=(8,3))
#             plt.bar(mm["country"].astype(str), mm["rmse"])
#             plt.title(f"{mdl} • RMSE by country")
#             plt.xlabel("Country")
#             plt.ylabel("RMSE")
#             plt.xticks(rotation=90)
#             plt.tight_layout()
#             st.pyplot(fig)

# ===== error dist (optional) =====
if actual_col:
    st.header("Error distribution (Actual - Forecast)")
    # pick one pair to visualize
    if not df.empty:
        c0 = df["country"].astype(str).iloc[0]
        m0 = df["model"].astype(str).iloc[0]
        g0 = df[(df["country"].astype(str)==c0) & (df["model"].astype(str)==m0)].copy()
        g0 = g0.dropna(subset=[actual_col])
        if not g0.empty:
            central = "p50" if "p50" in g0.columns else "point"
            err = g0[actual_col] - g0[central]
            fig = plt.figure(figsize=(6,3))
            plt.hist(err, bins=20)
            plt.title(f"{c0} • {m0} • Errors (pp)")
            plt.xlabel("Actual - Forecast")
            plt.ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig)

# ===== Principal Components  ======================================
st.header("Principal Components")
pcs_long = load_pcs(pcs_path)


if pcs_long.empty and pcs_path:
    st.warning("PCs file could not be loaded or had no usable columns (need long: country,date,PC,score; or wide: country,date,PC1..).")
elif not pcs_long.empty:
    # Controls for PCs section
    cty_pcs = st.selectbox("PCs • Country", sorted(pcs_long["country"].astype(str).unique()), index=0)
    # how many PCs to draw (we’ll choose by smallest numeric suffix first)
    max_k_avail = min(12, pcs_long.loc[pcs_long["country"].astype(str)==cty_pcs, "PC"].nunique())
    K = st.slider("Number of PCs to plot", 1, max_k_avail, min(4, max_k_avail))

    pcs_cty = pcs_long[pcs_long["country"].astype(str)==cty_pcs].copy()
    # order by numeric part of PC name if present
    def pc_key(s):
        try:
            n = int("".join(ch for ch in str(s) if ch.isdigit()))
            return n
        except Exception:
            return 10**9
    top_pcs = sorted(pcs_cty["PC"].unique(), key=pc_key)[:K]
    plot_df = pcs_cty[pcs_cty["PC"].isin(top_pcs)].sort_values(["PC", "date"])

    fig = plt.figure(figsize=(9,3.6))
    for pc in top_pcs:
        g = plot_df[plot_df["PC"]==pc]
        plt.plot(g["date"], g["score"], label=str(pc))
    plt.title(f"{cty_pcs} • Top {K} PCs (scores)")
    plt.xlabel("Month")
    plt.ylabel("PC score")
    plt.legend(ncol=min(K,4))
    plt.tight_layout()
    st.pyplot(fig)

    # quick snapshot of last values per PC
    last_vals = (
        pcs_cty.sort_values("date")
               .groupby("PC")["score"].last()
               .sort_index(key=lambda s: [pc_key(x) for x in s])
               .rename("last_value")
               .to_frame()
    )
    st.dataframe(last_vals, use_container_width=True)
else:
    st.info("Provide a PCs CSV path in the sidebar to enable the PCs view.")
    
    # ================= PC LOADINGS PANEL =================
st.header("Principal Components • Loadings")
loadings_long, pcs_long = load_pca_tables(load_path, pcs_path)

if loadings_long is None or loadings_long.empty:
    st.info("No loadings table found/loaded.")
else:
    # Controls
    ctry_opts = sorted(loadings_long["country"].astype(str).unique())
    pc_opts   = sorted(loadings_long["PC"].astype(str).unique(),
                       key=lambda s: (s[:2] != "PC", int(s[2:]) if s[2:].isdigit() else 1e9))

    ctry_sel = st.selectbox("Country (ISO3)", options=ctry_opts, index=0)
    pc_sel   = st.selectbox("Component", options=pc_opts, index=0)
    topN     = st.slider("Top variables by |loading|", 5, 50, 20, step=1)

    g = loadings_long[
        (loadings_long["country"].astype(str) == str(ctry_sel)) &
        (loadings_long["PC"].astype(str) == str(pc_sel))
    ].copy()

    if g.empty:
        st.warning("No loadings for the selected country/PC.")
    else:
        g["abs_loading"] = g["loading"].abs()
        g = g.sort_values("abs_loading", ascending=False).head(topN)

        fig = plt.figure(figsize=(8, max(3, 0.25*len(g))))
        plt.barh(g["variable"], g["loading"])
        plt.gca().invert_yaxis()  # largest at top
        plt.title(f"{ctry_sel} • {pc_sel} • Top {topN} loadings")
        plt.xlabel("Loading")
        plt.ylabel("Variable")
        plt.tight_layout()
        st.pyplot(fig)

        with st.expander("Show table"):
            st.dataframe(g[["variable", "PC", "loading"]].reset_index(drop=True))

# ================= PC SCORES PANEL =================
st.header("Principal Components • Scores over time")

if pcs_long is None or pcs_long.empty:
    st.info("No PC scores table found/loaded.")
else:
    # Controls
    ctry_opts2 = sorted(pcs_long["country"].astype(str).unique())
    ctry_sel2  = st.selectbox("Country (ISO3) for scores", options=ctry_opts2, index=0, key="pcs_cty")
    pcs_in_ctry = sorted(pcs_long.loc[pcs_long["country"] == ctry_sel2, "PC"].astype(str).unique(),
                         key=lambda s: (s[:2] != "PC", int(s[2:]) if s[2:].isdigit() else 1e9))
    pc_multi = st.multiselect("Components to plot", pcs_in_ctry, default=pcs_in_ctry[:3])

    gg = pcs_long[(pcs_long["country"] == ctry_sel2) & (pcs_long["PC"].isin(pc_multi))].copy()
    gg = gg.sort_values("date")

    if gg.empty:
        st.warning("No PC scores for the selected country/PCs.")
    else:
        # One line per PC
        fig = plt.figure(figsize=(10, 4))
        for pc_name, chunk in gg.groupby("PC"):
            plt.plot(chunk["date"], chunk["score"], label=pc_name)
        plt.title(f"{ctry_sel2} • PC scores over time")
        plt.xlabel("Month")
        plt.ylabel("Score (std units)")
        plt.legend(ncol=3)
        plt.tight_layout()
        st.pyplot(fig)

        with st.expander("Show scores table"):
            st.dataframe(
                gg[["country", "date", "PC", "score"]]
                  .sort_values(["PC", "date"])
                  .reset_index(drop=True)
            )