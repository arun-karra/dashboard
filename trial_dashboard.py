import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re
from datetime import datetime, timedelta

# Page config
st.set_page_config(layout="wide", page_title="Clinical Trial Dashboard")
st.title("📊 Clinical Trial Snapshot")

# Sidebar uploads
st.sidebar.header("Upload Reports")
asset_buf = st.sidebar.file_uploader("Asset Report (.xlsx)", type="xlsx")
forms_buf = st.sidebar.file_uploader("Forms Report (.xlsx)", type="xlsx")
sites_buf = st.sidebar.file_uploader("Sites Report (.xlsx)", type="xlsx")
if not (asset_buf and forms_buf and sites_buf):
    st.sidebar.info("Upload all three Excel reports to begin.")
    st.stop()

# --------- Helpers ---------
def normalize(col_name):
    s = str(col_name).lower()
    s = re.sub(r'[^a-z0-9]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def col(df, *candidates):
    """Fuzzy-match a column name by normalizing."""
    norm_map = {normalize(c): c for c in df.columns}
    # 1) exact normalized match
    for cand in candidates:
        nc = normalize(cand)
        if nc in norm_map:
            return norm_map[nc]
    # 2) partial substring match
    for cand in candidates:
        nc = normalize(cand)
        for norm_col, orig in norm_map.items():
            if nc in norm_col:
                return orig
    return None

@st.cache_data(show_spinner=False)
def load_report(buffer):
    raw = pd.read_excel(buffer, header=None)
    header_row = 0
    for i in range(len(raw)):
        row = raw.iloc[i].astype(str).str.lower()
        if ("subject number" in row.values and 
            ("study procedure" in row.values or "assessment name" in row.values)):
            header_row = i
            break
    df = pd.read_excel(buffer, header=header_row)
    df = df.dropna(axis=1, how="all")
    df = df[df.notna().any(axis=1)]
    return df

# Load DataFrames
asset_df = load_report(asset_buf)
forms_df = load_report(forms_buf)
sites_df = load_report(sites_buf)

# Show detected columns for debugging
with st.expander("🔧 Detected Columns"):
    st.write("Assets:", list(asset_df.columns))
    st.write("Forms:", list(forms_df.columns))
    st.write("Sites:", list(sites_df.columns))

# --------- Identify columns ---------
# Sites report
site_site_col       = col(sites_df, "Site Name")
site_subject_col    = col(sites_df, "Subject Number")
site_visit_col      = col(sites_df, "Visit Name", "Study Event")
site_assess_name    = col(sites_df, "Assessment Name", "Study Procedure")
site_assess_id      = col(sites_df, "Assessment ID")
site_assess_date    = col(sites_df, "Assessment Date", "Study Procedure Date")
site_status_col     = col(sites_df, "Assessment Status")
site_status_date    = col(sites_df, "Assessment Status Date")
task_cols           = [c for c in sites_df.columns if "task " in c.lower() and "date" in c.lower()]
action_raised_col   = col(sites_df, "Action Required - Date Raised")
action_resolved_col = col(sites_df, "Action Resolved Date")

# Asset report
asset_site_col      = col(asset_df, "Library/Site Name", "Site Name")
asset_subject_col   = col(asset_df, "Subject Number")
asset_visit_col     = col(asset_df, "Study Event", "Visit Name")
asset_assess_name   = col(asset_df, "Study Procedure", "Assessment Name")
asset_date          = col(asset_df, "Study Procedure Date", "Assessment Date")
asset_upload_date   = col(asset_df, "Upload Date")

# Forms report
form_spid_col       = col(forms_df, "Study Procedure ID", "Assessment ID")
form_created_col    = col(forms_df, "Date Created", "Form Created Date")
form_submitted_col  = col(forms_df, "Submitted Date", "Form Submitted Date")

# --------- Validate ---------
required = {
    "Sites": [site_site_col, site_subject_col, site_visit_col, site_assess_name, site_assess_date],
    "Assets": [asset_site_col, asset_subject_col, asset_visit_col, asset_assess_name, asset_date, asset_upload_date],
    "Forms": [form_spid_col, form_submitted_col],
}
errors = []
for rpt, cols in required.items():
    missing = [c for c in cols if c is None]
    if missing:
        errors.append(f"{rpt} missing columns {missing}")
if errors:
    st.error("❌ Required columns not found:\n" + "\n".join(errors))
    st.stop()

# --------- Parse datetimes ---------
for df, cols in [
    (sites_df,     [site_assess_date, site_status_date] + task_cols + [action_raised_col, action_resolved_col]),
    (asset_df,     [asset_date, asset_upload_date]),
    (forms_df,     [form_created_col, form_submitted_col])
]:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

# --------- Compute max asset delay per assessment ---------
asset_df["upload_delay"] = (asset_df[asset_upload_date] - asset_df[asset_date]).dt.days
asset_agg = (
    asset_df
    .groupby([asset_site_col, asset_subject_col, asset_visit_col, asset_assess_name], as_index=False)
    .agg(max_upload_delay=("upload_delay", "max"))
)
sites_df = sites_df.merge(
    asset_agg,
    how="left",
    left_on=[site_site_col, site_subject_col, site_visit_col, site_assess_name],
    right_on=[asset_site_col, asset_subject_col, asset_visit_col, asset_assess_name],
)
sites_df["max_upload_delay"] = sites_df["max_upload_delay"].fillna(0).astype(int)

# --------- KPI Calculations ---------
today = pd.Timestamp(datetime.now().date())

# 1. Total assessments & subjects
total_assessments = sites_df[site_assess_id].nunique()
total_subjects    = sites_df[site_subject_col].nunique()

# 2. Assessments In Progress
in_progress = sites_df[sites_df[site_status_col].str.lower() == "in progress"][site_assess_id].nunique()

# 3. Avg time from assessment → status
completed_mask = sites_df[site_status_date].notna()
avg_cycle = (
    (sites_df.loc[completed_mask, site_status_date] - sites_df.loc[completed_mask, site_assess_date])
    .dt.days
    .mean()
)

# 4. Flag assessments with no assets uploaded within 5 days
late_assets = sites_df[sites_df["max_upload_delay"] > 5][site_assess_id].nunique()

# 5. Flag tasks outstanding > 5 days
sites_df["task_delays"] = sites_df[task_cols].apply(
    lambda row: (row - sites_df.loc[row.name, site_assess_date]).dt.days.max(),
    axis=1
)
late_tasks = sites_df[sites_df["task_delays"] > 5][site_assess_id].nunique()

# 6. Site delay counts (assets vs tasks)
site_delays = (
    sites_df
    .groupby(site_site_col)
    .agg(
        late_assets_count=("max_upload_delay",   lambda s: (s>5).sum()),
        late_tasks_count =("task_delays",        lambda s: (s>5).sum())
    )
    .reset_index()
)

# 7. Open action required
open_actions = sites_df[
    sites_df[action_raised_col].notna() & sites_df[action_resolved_col].isna()
][site_assess_id].nunique()

# 8. Forms not submitted after 5 days of upload
forms_map = forms_df[[form_spid_col, form_submitted_col]]
merge_fu = asset_df[[asset_site_col, asset_subject_col, asset_visit_col, asset_assess_name, asset_upload_date]].drop_duplicates()
merge_fu = merge_fu.merge(
    forms_map,
    how="left",
    left_on=[asset_site_col, asset_subject_col, asset_visit_col, asset_assess_name],
    right_on=[asset_site_col, asset_subject_col, asset_visit_col, asset_assess_name]
)
merge_fu["form_delay"] = (merge_fu[form_submitted_col] - merge_fu[asset_upload_date]).dt.days
late_forms = merge_fu[merge_fu["form_delay"] > 5][asset_assess_name].nunique()

# --------- Display Metrics ---------
st.header("Key Metrics")
r1c1, r1c2, r1c3, r1c4 = st.columns(4)
r1c1.metric("Total Assessments", total_assessments)
r1c2.metric("Total Subjects", total_subjects)
r1c3.metric("In Progress", in_progress)
r1c4.metric("Avg Cycle Time (days)", f"{avg_cycle:.1f}")

r2c1, r2c2, r2c3, r2c4 = st.columns(4)
r2c1.metric("Assessments w/o Assets >5d", late_assets)
r2c2.metric("Tasks Outstanding >5d", late_tasks)
r2c3.metric("Open Action Required", open_actions)
r2c4.metric("Forms >5d post‑upload", late_forms)

st.markdown("---")

# --------- Site Delay Chart ---------
st.subheader("🔴 Site Delay Frequency (≥5 days)")
sd_long = site_delays.melt(
    id_vars=[site_site_col],
    value_vars=["late_assets_count", "late_tasks_count"],
    var_name="Delay Type",
    value_name="Count"
)
chart = (
    alt.Chart(sd_long)
    .mark_bar()
    .encode(
        x=alt.X(f"{site_site_col}:N", title="Site"),
        y="Count:Q",
        color="Delay Type:N",
        column=alt.Column("Delay Type:N", title=None)
    )
    .properties(width=150)
)
st.altair_chart(chart, use_container_width=True)

# --------- Recent Activity ---------
st.subheader("🕒 Most Recent Activity")
tabA, tabB, tabC = st.tabs(["Assets","Forms","Assessments"])

with tabA:
    st.write("#### Recent Asset Uploads")
    dfA = asset_df.sort_values(asset_upload_date, ascending=False).head(5)
    st.table(dfA[[asset_site_col, asset_subject_col, asset_visit_col, asset_assess_name, asset_date, asset_upload_date, "upload_delay"]])

with tabB:
    st.write("#### Recent Form Submissions")
    dfB = forms_df.sort_values(form_submitted_col, ascending=False).head(5)
    st.table(dfB[[form_spid_col, form_created_col, form_submitted_col]])

with tabC:
    st.write("#### Recent Assessment Completions")
    dfC = sites_df.loc[completed_mask].sort_values(site_status_date, ascending=False).head(5)
    st.table(dfC[[site_assess_id, site_assess_date, site_status_date]])

st.success("✅ Dashboard loaded. Adjust sidebar thresholds to explore!")
