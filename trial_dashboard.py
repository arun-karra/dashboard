import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Clinical Trial Dashboard")
st.title("📊 Clinical Trial Snapshot")

# --- Sidebar File Uploads ---
st.sidebar.header("Upload Reports")
asset_buf = st.sidebar.file_uploader("Asset Report (.xlsx)", type="xlsx")
forms_buf = st.sidebar.file_uploader("Forms Report (.xlsx)", type="xlsx")
sites_buf = st.sidebar.file_uploader("Sites Report (.xlsx)", type="xlsx")
if not (asset_buf and forms_buf and sites_buf):
    st.sidebar.info("Upload Asset, Forms, and Sites reports to begin.")
    st.stop()

# --- Helpers ---
def normalize(col_name: str) -> str:
    s = str(col_name).lower()
    s = re.sub(r'[^a-z0-9]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def col(df: pd.DataFrame, *cands) -> str | None:
    """Fuzzy-match a column name by normalizing."""
    norm_map = {normalize(c): c for c in df.columns}
    # exact
    for cand in cands:
        nc = normalize(cand)
        if nc in norm_map:
            return norm_map[nc]
    # partial
    for cand in cands:
        nc = normalize(cand)
        for nm, orig in norm_map.items():
            if nc in nm:
                return orig
    return None

@st.cache_data(show_spinner=False)
def load_df(buf) -> pd.DataFrame:
    raw = pd.read_excel(buf, header=None)
    hdr = 0
    for i in range(len(raw)):
        row = raw.iloc[i].astype(str).str.lower()
        if "subject number" in row.values and "study procedure" in row.values:
            hdr = i
            break
    df = pd.read_excel(buf, header=hdr).dropna(axis=1, how="all")
    return df[df.notna().any(axis=1)]

# --- Load DataFrames ---
asset_df = load_df(asset_buf)
forms_df = load_df(forms_buf)
sites_df = load_df(sites_buf)

# --- Debug: detected columns ---
with st.expander("🔧 Detected Columns"):
    st.write("Assets:", asset_df.columns.tolist())
    st.write("Forms:", forms_df.columns.tolist())
    st.write("Sites:", sites_df.columns.tolist())

# --- Identify columns ---
# Sites report
s_site       = col(sites_df, "Site Name")
s_subj       = col(sites_df, "Subject Number")
s_visit      = col(sites_df, "Visit Name", "Study Event")
s_assess     = col(sites_df, "Assessment Name", "Study Procedure")
s_id         = col(sites_df, "Assessment ID")
s_date       = col(sites_df, "Assessment Date", "Study Procedure Date")
s_status     = col(sites_df, "Assessment Status")
s_status_dt  = col(sites_df, "Assessment Status Date")
task_cols    = [c for c in sites_df.columns if "task " in c.lower() and "date" in c.lower()]
act_raised   = col(sites_df, "Action Required - Date Raised")
act_resolved = col(sites_df, "Action Resolved Date")

# Asset report
a_site       = col(asset_df, "Library/Site Name", "Site Name")
a_subj       = col(asset_df, "Subject Number")
a_visit      = col(asset_df, "Study Event", "Visit Name")
a_assess     = col(asset_df, "Study Procedure", "Assessment Name")
a_date       = col(asset_df, "Study Procedure Date", "Assessment Date")
a_upload     = col(asset_df, "Upload Date")

# Forms report
f_spid       = col(forms_df, "Study Procedure ID", "Assessment ID")
f_created    = col(forms_df, "Date Created", "Form Created Date")
f_submitted  = col(forms_df, "Submitted Date", "Form Submitted Date")

# --- Validate presence ---
required = {
    "Sites":  [s_site, s_subj, s_visit, s_assess, s_date],
    "Assets": [a_site, a_subj, a_visit, a_assess, a_date, a_upload],
    "Forms":  [f_spid, f_submitted],
}
errs = []
for name, cols in required.items():
    miss = [c for c in cols if c is None]
    if miss:
        errs.append(f"{name}: missing columns {miss}")
if errs:
    st.error("❌ Required columns missing:\n" + "\n".join(errs))
    st.stop()

# --- Parse datetimes ---
for df, cols in [
    (sites_df, [s_date, s_status_dt, act_raised, act_resolved] + task_cols),
    (asset_df, [a_date, a_upload]),
    (forms_df, [f_created, f_submitted])
]:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

# --- Compute asset delays & merge into sites ---
asset_df["upload_delay"] = (asset_df[a_upload] - asset_df[a_date]).dt.days
asset_agg = (
    asset_df
    .groupby([a_site, a_subj, a_visit, a_assess], as_index=False)
    .agg(
        first_upload=("upload_date", "min"),
        max_upload_delay=("upload_delay", "max")
    )
)
sites_df = sites_df.merge(
    asset_agg,
    how="left",
    left_on=[s_site, s_subj, s_visit, s_assess],
    right_on=[a_site, a_subj, a_visit, a_assess],
)
sites_df["max_upload_delay"] = sites_df["max_upload_delay"].fillna(0).astype(int)

# --- Merge forms into sites on Assessment ID ---
sites_df = sites_df.merge(
    forms_df[[f_spid, f_submitted]].rename(columns={f_spid: s_id, f_submitted: "form_submitted"}),
    how="left",
    on=s_id
)

# --- KPI Calculations ---
today = pd.Timestamp(datetime.now().date())

# 1) Total Assessments & Subjects
total_assess  = sites_df[s_id].nunique()
total_subjects = sites_df[s_subj].nunique()

# 2) Assessments In Progress
in_progress = (
    sites_df[sites_df[s_status].str.lower() == "in progress"][s_id].nunique()
)

# 3) Avg Cycle Time (days)
completed = sites_df[sites_df[s_status_dt].notna()]
avg_cycle = ((completed[s_status_dt] - completed[s_date]).dt.days).mean()

# 4) Assessments w/o assets >5d
late_assets = sites_df[sites_df["max_upload_delay"] > 5][s_id].nunique()

# 5) Tasks outstanding >5d
sites_df["task_delay"] = sites_df[task_cols].apply(
    lambda row: (row - sites_df.loc[row.name, s_date]).dt.days.max(),
    axis=1
)
late_tasks = sites_df[sites_df["task_delay"] > 5][s_id].nunique()

# 6) Site Delay Frequency
site_delays = (
    sites_df
    .groupby(s_site)
    .agg(
        assets_late=("max_upload_delay", lambda s: (s>5).sum()),
        tasks_late =("task_delay",      lambda s: (s>5).sum())
    )
    .reset_index()
)

# 7) Open Action Required
open_actions = sites_df[
    sites_df[act_raised].notna() & sites_df[act_resolved].isna()
][s_id].nunique()

# 8) Forms not submitted >5d post upload
sites_df["form_delay"] = (sites_df["form_submitted"] - sites_df["first_upload"]).dt.days
late_forms = sites_df[sites_df["form_delay"] > 5][s_id].nunique()

# --- Display Metrics ---
st.header("Key Metrics")

# Row 1
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Assessments",   total_assess)
c1.caption("Count of all planned assessments up to today.")
c2.metric("Total Subjects",      total_subjects)
c2.caption("Unique enrolled subjects.")
c3.metric("In Progress",         in_progress)
c3.caption("Assessments not yet marked Complete.")
c4.metric("Avg Cycle Time (days)", f"{avg_cycle:.1f}")
c4.caption("Avg days from Assessment Date → Status Date (Complete).")

# Row 2
c5, c6, c7, c8 = st.columns(4)
c5.metric("Assess w/o Assets >5d", late_assets)
c5.caption("Assessments where assets uploaded >5 days after visit.")
c6.metric("Tasks >5d Outstanding", late_tasks)
c6.caption("Any Task (1–n) >5 days past Assessment Date or missing.")
c7.metric("Open Action Required",  open_actions)
c7.caption("QC flags raised but not yet resolved.")
c8.metric("Forms >5d post‑upload", late_forms)
c8.caption("Form submissions delayed >5 days after first asset upload.")

st.markdown("---")

# --- Delayed Assessments Section ---
st.subheader("⏰ Assessments Delayed >5 days")
delay_mask = (
    (sites_df[s_status].str.lower() == "in progress") &
    ((today - sites_df[s_date]).dt.days >= 5)
)
delayed_assess = sites_df[delay_mask]
st.dataframe(
    delayed_assess[[s_site, s_subj, s_visit, s_assess, s_date]],
    height=250
)

# --- Assets Not Yet Uploaded Section ---
st.subheader("📁 Assets Not Uploaded >5 days")
# Criteria: first_upload is null or max_upload_delay >= 5
missing_mask = (sites_df["max_upload_delay"] >= 5)
st.dataframe(
    sites_df[missing_mask][[s_site, s_subj, s_visit, s_assess, s_date, "max_upload_delay"]],
    height=250
)

# --- Site Delay Frequency Chart ---
st.subheader("🔴 Site Delay Frequency (Late >5 days)")
st.caption("Bars show number of assessments with asset or task delays ≥5 days, grouped by site.")
sd_long = site_delays.melt(
    id_vars=[s_site],
    value_vars=["assets_late","tasks_late"],
    var_name="Delay Type",
    value_name="Count"
)
chart = (
    alt.Chart(sd_long)
    .mark_bar()
    .encode(
        x=alt.X(f"{s_site}:N", title="Site"),
        y="Count:Q",
        color="Delay Type:N",
        column=alt.Column("Delay Type:N", title=None)
    )
    .properties(width=150)
)
st.altair_chart(chart, use_container_width=True)

# --- Recent Activity Panels ---
st.subheader("🕒 Most Recent Activity")
tabA, tabB, tabC = st.tabs(["Assets","Forms","Assessments"])
with tabA:
    st.write("Recent Asset Uploads")
    dfA = asset_df.sort_values(a_upload, ascending=False).head(5)
    st.table(dfA[[a_site, a_subj, a_visit, a_assess, a_date, a_upload, "upload_delay"]])
with tabB:
    st.write("Recent Form Submissions")
    dfB = forms_df.sort_values(f_submitted, ascending=False).head(5)
    st.table(dfB[[f_spid, f_created, f_submitted]])
with tabC:
    st.write("Recent Assessment Completions")
    dfC = sites_df.dropna(subset=[s_status_dt]).sort_values(s_status_dt, ascending=False).head(5)
    st.table(dfC[[s_id, s_date, s_status_dt]])

st.success("✅ Dashboard generated! Adjust criteria in code as needed.")
