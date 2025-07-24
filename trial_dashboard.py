import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Clinical Trial Dashboard")
st.title("📊 Clinical Trial Snapshot")

# ----------------------------------
# 1. File upload
# ----------------------------------
st.sidebar.header("Upload Reports")
asset_file = st.sidebar.file_uploader("Asset Report (.xlsx)", type="xlsx")
forms_file = st.sidebar.file_uploader("Forms Report (.xlsx)", type="xlsx")
sites_file = st.sidebar.file_uploader("Sites Report (.xlsx)", type="xlsx")

if not (asset_file and forms_file and sites_file):
    st.info("⬆️ Upload all three reports to begin.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_report(buf):
    # Auto-detect header row by scanning for key column names
    raw = pd.read_excel(buf, header=None)
    hdr = 0
    for i in range(len(raw)):
        row = raw.iloc[i].astype(str).str.lower()
        if ("subject number" in row.values
            and ("study procedure" in row.values or "assessment name" in row.values)):
            hdr = i
            break
    df = pd.read_excel(buf, header=hdr)
    df = df.dropna(axis=1, how="all")
    df = df[df.notna().any(axis=1)]
    return df

asset_df = load_report(asset_file)
forms_df = load_report(forms_file)
sites_df = load_report(sites_file)

st.sidebar.write("**Detected columns**")
st.sidebar.write("Assets:", list(asset_df.columns))
st.sidebar.write("Forms:", list(forms_df.columns))
st.sidebar.write("Sites:", list(sites_df.columns))

# ----------------------------------
# 2. Fuzzy column match helper
# ----------------------------------
def col(df, *cands):
    for c in cands:
        for name in df.columns:
            if str(name).strip().lower() == c.lower().strip():
                return name
    for c in cands:
        for name in df.columns:
            if c.lower() in str(name).lower():
                return name
    return None

# ----------------------------------
# 3. Identify columns
# ----------------------------------
# Sites
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

# Asset
asset_site_col      = col(asset_df, "Library/Site Name", "Site Name")
asset_subject_col   = col(asset_df, "Subject Number")
asset_visit_col     = col(asset_df, "Study Event", "Visit Name")
asset_assess_name   = col(asset_df, "Study Procedure", "Assessment Name")
asset_date          = col(asset_df, "Study Procedure Date", "Assessment Date")
asset_upload_date   = col(asset_df, "Upload Date")

# Forms
form_spid_col       = col(forms_df, "Study Procedure ID", "Assessment ID")
form_created_col    = col(forms_df, "Date Created", "Form Created Date")
form_submitted_col  = col(forms_df, "Submitted Date", "Form Submitted Date")

# ----------------------------------
# 4. Validate essential columns
# ----------------------------------
required_checks = {
    "Sites":     [site_site_col, site_subject_col, site_visit_col, site_assess_name, site_assess_date],
    "Assets":    [asset_site_col, asset_subject_col, asset_visit_col, asset_assess_name, asset_date, asset_upload_date],
    "Forms":     [form_spid_col, form_submitted_col],
}
missing = []
for report, reqs in required_checks.items():
    miss = [r for r in reqs if r is None]
    if miss:
        missing.append(f"{report}: missing {miss}")
if missing:
    st.error("Cannot proceed, missing columns:\n" + "\n".join(missing))
    st.stop()

# ----------------------------------
# 5. Parse datetimes
# ----------------------------------
for df, cols in [
    (sites_df,     [site_assess_date, site_status_date] + task_cols + [action_raised_col, action_resolved_col]),
    (asset_df,     [asset_date, asset_upload_date]),
    (forms_df,     [form_created_col, form_submitted_col])
]:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

# ----------------------------------
# 6. Compute upload delays and merge into Sites
# ----------------------------------
# Per-asset delay
asset_df["upload_delay"] = (asset_df[asset_upload_date] - asset_df[asset_date]).dt.days

# Aggregate max delay per assessment key
asset_agg = (
    asset_df
    .groupby([asset_site_col, asset_subject_col, asset_visit_col, asset_assess_name], as_index=False)
    .agg(max_upload_delay=("upload_delay", "max"))
)

# Merge into sites_df
sites_df = sites_df.merge(
    asset_agg,
    how="left",
    left_on=[site_site_col, site_subject_col, site_visit_col, site_assess_name],
    right_on=[asset_site_col, asset_subject_col, asset_visit_col, asset_assess_name],
)

# Fill 0 for assessments with no assets yet
sites_df["max_upload_delay"] = sites_df["max_upload_delay"].fillna(0).astype(int)

# ----------------------------------
# 7. KPI calculations
# ----------------------------------
today = pd.Timestamp(datetime.now().date())

# 1. Total assessments & subjects
total_assessments = sites_df[site_assess_id].nunique()
total_subjects    = sites_df[site_subject_col].nunique()

# 2. Assessments In Progress
in_progress = (
    sites_df[sites_df[site_status_col].str.lower() == "in progress"]
    [site_assess_id]
    .nunique()
)

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
# compute per-row max task delay
sites_df["task_delays"] = sites_df[task_cols].apply(
    lambda row: (row - sites_df.loc[row.name, site_assess_date]).dt.days.max(), axis=1
)
late_tasks = sites_df[sites_df["task_delays"] > 5][site_assess_id].nunique()

# 6. Per-site delay counts (assets vs tasks)
site_delays = (
    sites_df
    .groupby(site_site_col)
    .agg(
        late_assets_count = ("max_upload_delay", lambda s: (s>5).sum()),
        late_tasks_count  = ("task_delays",   lambda s: (s>5).sum())
    )
    .reset_index()
)

# 7. Open Action Required
open_actions = (
    sites_df
    .loc[sites_df[action_raised_col].notna() & sites_df[action_resolved_col].isna(), site_assess_id]
    .nunique()
)

# 8. Forms not submitted after 5 days of upload
forms_map = forms_df[[form_spid_col, form_submitted_col]]
merge_fu = asset_df[[asset_site_col, asset_subject_col, asset_visit_col, asset_assess_name, "upload_delay"]]
merge_fu = merge_fu.merge(
    forms_map,
    how="left",
    left_on=[asset_site_col, asset_subject_col, asset_visit_col, asset_assess_name],
    right_on=[asset_site_col, site_subject_col, site_visit_col, site_assess_name],
)
merge_fu["form_delay"] = (merge_fu[form_submitted_col] - merge_fu["upload_delay"].apply(lambda d: pd.Timestamp('2024-01-01') + timedelta(days=d))).dt.days
# (Alternatively, join upload_date directly if preferred)
late_forms = merge_fu[merge_fu["form_delay"] > 5][asset_assess_name].nunique()

# ----------------------------------
# 8. Render metrics
# ----------------------------------
st.header("Key Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Assessments", total_assessments)
c2.metric("Total Subjects", total_subjects)
c3.metric("In Progress", in_progress)
c4.metric("Avg Cycle Time (days)", f"{avg_cycle:.1f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Assessments w/o Assets >5d", late_assets)
c6.metric("Tasks Outstanding >5d", late_tasks)
c7.metric("Open Action Required", open_actions)
c8.metric("Forms >5d post-upload", late_forms)

st.markdown("---")

# ----------------------------------
# 9. Site delay chart
# ----------------------------------
st.subheader("🔴 Site Delays (≥5d)")
sd_long = site_delays.melt(id_vars=[site_site_col], 
                           value_vars=["late_assets_count","late_tasks_count"], 
                           var_name="Delay Type",
                           value_name="Count")
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

# ----------------------------------
# 10. Recent activity
# ----------------------------------
st.subheader("🕒 Most Recent Activity")
tabA, tabB, tabC = st.tabs(["Assets","Forms","Assessments"])

with tabA:
    st.write("#### Recent Asset Uploads")
    recent_assets = asset_df.sort_values(asset_upload_date, ascending=False).head(5)
    st.table(recent_assets[[asset_site_col, asset_subject_col, asset_visit_col, asset_assess_name,
                             asset_date, asset_upload_date, "upload_delay"]])

with tabB:
    st.write("#### Recent Form Submissions")
    recent_forms = forms_df.sort_values(form_submitted_col, ascending=False).head(5)
    st.table(recent_forms[[form_spid_col, form_submitted_col]])

with tabC:
    st.write("#### Recent Assessment Completions")
    recent_completions = sites_df.loc[completed_mask].sort_values(site_status_date, ascending=False).head(5)
    st.table(recent_completions[[site_assess_id, site_assess_date, site_status_date]])

# ----------------------------------
# 11. Debug expander
# ----------------------------------
with st.expander("🔧 Column diagnostics"):
    st.write("Sites:", list(sites_df.columns))
    st.write("Assets:", list(asset_df.columns))
    st.write("Forms:", list(forms_df.columns))
