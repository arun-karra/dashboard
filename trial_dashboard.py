import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Clinical Trial Dashboard")

@st.cache_data
def load_report(path):
    # auto-detect header row as before, then return DataFrame
    raw = pd.read_excel(path, header=None)
    hdr = 0
    for i in range(len(raw)):
        vals = raw.iloc[i].astype(str).str.lower()
        if "subject id" in vals.values and "visit" in vals.values:
            hdr = i
            break
    df = pd.read_excel(path, header=hdr)
    df = df.dropna(axis=1, how='all')
    df = df[df.notna().any(axis=1)]
    return df

st.sidebar.title("Upload Reports")
asset_file = st.sidebar.file_uploader("Asset Report", type=["xlsx"])
forms_file = st.sidebar.file_uploader("Forms Report", type=["xlsx"])
sites_file = st.sidebar.file_uploader("Sites Report", type=["xlsx"])

if not asset_file or not forms_file or not sites_file:
    st.sidebar.warning("Please upload all three reports.")
    st.stop()

asset_df = load_report(asset_file)
forms_df = load_report(forms_file)
sites_df = load_report(sites_file)

# --- Identify columns ---
def find_col(df, keys):
    for k in keys:
        for c in df.columns:
            if c.lower().strip() == k.lower().strip():
                return c
    for k in keys:
        for c in df.columns:
            if k.lower() in c.lower():
                return c
    return None

# Sites
assess_id = find_col(sites_df, ["assessment id"])
assess_date = find_col(sites_df, ["assessment date", "study procedure date"])
status_date = find_col(sites_df, ["assessment status date"])
status_col = find_col(sites_df, ["status", "assessment status"])
task_cols = [c for c in sites_df.columns if "task" in c.lower() and "date" in c.lower()]

# Assets
asset_assess_id = find_col(asset_df, ["assessment id"])
asset_date = find_col(asset_df, ["study procedure date", "assessment date"])
upload_date = find_col(asset_df, ["upload date"])

# Forms
form_spid = find_col(forms_df, ["study procedure id"])
form_sub = find_col(forms_df, ["submitted", "submitted date"])
form_created = find_col(forms_df, ["date created", "form created date"])

# Validate
required = {
    "Sites": [assess_id, assess_date, status_col],
    "Assets": [asset_assess_id, asset_date, upload_date],
    "Forms": [form_spid, form_sub],
}
missing = [f"{k}: {reqs}" for k, reqs in required.items() if any(r is None for r in reqs)]
if missing:
    st.error("Missing required columns:\n" + "\n".join(missing))
    st.stop()

# --- Prep data ---
# Ensure datetimes
for df, cols in [(sites_df, [assess_date, status_date] + task_cols),
                 (asset_df, [asset_date, upload_date]),
                 (forms_df, [form_spid, form_sub, form_created])]:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

today = pd.Timestamp(datetime.now().date())

# 1. Total Assessments & Subjects
total_assess = sites_df[assess_id].nunique()
total_subjects = sites_df["Subject ID"].nunique() if "Subject ID" in sites_df.columns else sites_df.iloc[:,1].nunique()

# 2. In Progress
in_progress = sites_df[
    (sites_df[status_col].str.lower() == "in progress")
][assess_id].nunique()

# 3. Avg Time: Assessment → Status
completed = sites_df.dropna(subset=[status_date])
avg_cycle = (completed[status_date] - completed[assess_date]).dt.days.mean()

# 4. Assets >5d
asset_df["upload_delay"] = (asset_df[upload_date] - asset_df[asset_date]).dt.days
late_assets = asset_df[asset_df["upload_delay"] > 5][asset_assess_id].unique()

# 5. Tasks >5d
sites_df["max_task_delay"] = sites_df[task_cols].apply(
    lambda row: np.nanmax((row - sites_df[assess_date]).dt.days), axis=1
)
late_tasks = sites_df[sites_df["max_task_delay"] > 5][assess_id].unique()

# 6. Site Delays assets/tasks
sites_delay_counts = sites_df.groupby("Site Name").apply(
    lambda df: pd.Series({
        "asset_late": df[assess_id].isin(late_assets).sum(),
        "task_late": df[assess_id].isin(late_tasks).sum()
    })
).reset_index()

# 7. Open Actions
raised = find_col(sites_df, ["action required - date raised"])
resolved = find_col(sites_df, ["action resolved date"])
open_actions = sites_df[
    sites_df[raised].notna() & sites_df[resolved].isna()
][assess_id].nunique()

# 8. Forms >5d after upload
# join upload→forms
merge_df = asset_df[[asset_assess_id, upload_date]].drop_duplicates().merge(
    forms_df[[form_spid, form_sub]], left_on=asset_assess_id, right_on=form_spid, how="left"
)
merge_df["form_delay"] = (merge_df[form_sub] - merge_df[upload_date]).dt.days
late_forms = merge_df[merge_df["form_delay"] > 5][asset_assess_id].nunique()

# --- Layout ---
st.title("🏃 Clinical Trial Dashboard")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Assessments", total_assess)
c2.metric("Total Subjects", total_subjects)
c3.metric("In Progress", in_progress)
c4.metric("Avg Assessment→Status (days)", f"{avg_cycle:.1f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Assessments w/o Assets >5d", len(late_assets))
c6.metric("Outstanding Tasks >5d", len(late_tasks))
c7.metric("Open Action Required", open_actions)
c8.metric("Forms >5d post-upload", late_forms)

st.markdown("---")
# Site delay bar chart
st.subheader("🔴 Site Delay Frequency (≥5 days)")
chart = alt.Chart(sites_delay_counts.melt("Site Name", value_name="Count", var_name="Type")).mark_bar().encode(
    x=alt.X("Site Name:N", sort=None),
    y="Count:Q",
    color="Type:N",
    column="Type:N"
).properties(width=150)
st.altair_chart(chart, use_container_width=True)

# Recent activity tables
st.markdown("### Most Recent Activity")
tabA, tabB, tabC = st.tabs(["Assets", "Forms", "Assessments"])
with tabA:
    st.write("#### Recent Asset Uploads")
    dfA = asset_df.sort_values(upload_date, ascending=False).head(5)
    st.table(dfA[[asset_assess_id, asset_date, upload_date, "upload_delay"]])

with tabB:
    st.write("#### Recent Form Submissions")
    dfB = forms_df.sort_values(form_sub, ascending=False).head(5)
    st.table(dfB[[form_spid, assess_date, form_created, form_sub]])

with tabC:
    st.write("#### Recent Assessment Completions")
    dfC = completed.sort_values(status_date, ascending=False).head(5)
    st.table(dfC[[assess_id, assess_date, status_date]])

# Diagnostic: show detected columns on expand
with st.expander("🔧 Detected Columns"):
    st.write("Sites:", list(sites_df.columns))
    st.write("Assets:", list(asset_df.columns))
    st.write("Forms:", list(forms_df.columns))
