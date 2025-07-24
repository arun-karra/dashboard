import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re
from datetime import datetime, timedelta

# --- Page Setup ---
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
def normalize(name: str) -> str:
    s = str(name).lower()
    s = re.sub(r'[^a-z0-9]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def col(df: pd.DataFrame, *cands) -> str | None:
    norm_map = {normalize(c): c for c in df.columns}
    # exact
    for cand in cands:
        nc = normalize(cand)
        if nc in norm_map:
            return norm_map[nc]
    # fuzzy
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

# --- Debug: show detected columns ---
with st.expander("🔧 Detected Columns"):
    st.write("Assets:", list(asset_df.columns))
    st.write("Forms:", list(forms_df.columns))
    st.write("Sites:", list(sites_df.columns))

# --- Identify Columns ---
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

# --- Validate Required Columns ---
required = {
    "Sites":  [s_site, s_subj, s_visit, s_assess, s_date],
    "Assets": [a_site, a_subj, a_visit, a_assess, a_date, a_upload],
    "Forms":  [f_spid, f_submitted],
}
errors = []
for name, cols in required.items():
    missing = [c for c in cols if c is None]
    if missing:
        errors.append(f"{name} missing {missing}")
if errors:
    st.error("❌ Missing required columns:\n" + "\n".join(errors))
    st.stop()

# --- Parse Date Columns ---
for df, cols in [
    (sites_df, [s_date, s_status_dt, act_raised, act_resolved] + task_cols),
    (asset_df, [a_date, a_upload]),
    (forms_df, [f_created, f_submitted])
]:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

# --- Compute Asset Delays & Merge into Sites ---
asset_df["upload_delay"] = (asset_df[a_upload] - asset_df[a_date]).dt.days
first_upload_col = "first_upload_date"
asset_agg = (
    asset_df
    .groupby([a_site, a_subj, a_visit, a_assess], as_index=False)
    .agg(
        **{first_upload_col: (a_upload, "min")},
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

# --- Merge Forms by Assessment ID ---
sites_df = sites_df.merge(
    forms_df[[f_spid, f_submitted]].rename(columns={f_spid: s_id, f_submitted: "form_submitted"}),
    how="left",
    on=s_id
)

# --- KPI Computations ---
today = pd.Timestamp(datetime.now().date())

# 1) Total Assessments & Subjects
total_assess = sites_df[s_id].nunique()
total_subj   = sites_df[s_subj].nunique()

# 2) Assessments In Progress
in_prog = sites_df[sites_df[s_status].str.lower()=="in progress"][s_id].nunique()

# 3) Avg Cycle Time (days)
completed = sites_df[sites_df[s_status_dt].notna()]
avg_cycle = ((completed[s_status_dt] - completed[s_date]).dt.days).mean()

# 4) Assessments w/o assets > 5 days
late_assets = sites_df[sites_df["max_upload_delay"] > 5][s_id].nunique()

# 5) Tasks outstanding > 5 days
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
        assets_late = ("max_upload_delay", lambda s: (s>5).sum()),
        tasks_late  = ("task_delay",        lambda s: (s>5).sum())
    )
    .reset_index()
)

# 7) Open Action Required
open_actions = sites_df[
    sites_df[act_raised].notna() & sites_df[act_resolved].isna()
][s_id].nunique()

# 8) Forms not submitted > 5 days post-upload
sites_df["form_delay"] = (sites_df["form_submitted"] - sites_df[first_upload_col]).dt.days
late_forms = sites_df[sites_df["form_delay"] > 5][s_id].nunique()

# --- Display Metrics with Descriptions ---
st.header("Key Metrics")

# Row 1
r1c1, r1c2, r1c3, r1c4 = st.columns(4)
r1c1.metric("Total Assessments",    total_assess)
r1c1.caption("All planned assessments to date.")
r1c2.metric("Total Subjects",       total_subj)
r1c2.caption("Unique subjects enrolled.")
r1c3.metric("In Progress",          in_prog)
r1c3.caption("Assessments not marked complete.")
r1c4.metric("Avg Cycle Time (days)", f"{avg_cycle:.1f}")
r1c4.caption("Mean days from assessment date to status date for completed.")

# Row 2
r2c1, r2c2, r2c3, r2c4 = st.columns(4)
r2c1.metric("Assets Late (>5d)",    late_assets)
r2c1.caption("Assessments with assets uploaded >5 days after date.")
r2c2.metric("Tasks Outstanding (>5d)", late_tasks)
r2c2.caption("Assessments with any task outstanding >5 days post date.")
r2c3.metric("Open Action Required",  open_actions)
r2c3.caption("QC flags raised but not resolved.")
r2c4.metric("Forms Late (>5d)",      late_forms)
r2c4.caption("Forms submitted >5 days after first asset upload.")

st.markdown("---")

# --- Delayed Assessments Section ---
st.subheader("⏰ Assessments Delayed ≥5 days (In Progress)")
delay_mask = (
    (sites_df[s_status].str.lower()=="in progress") &
    ((today - sites_df[s_date]).dt.days >= 5)
)
st.dataframe(
    sites_df[delay_mask][[s_site, s_subj, s_visit, s_assess, s_date]],
    height=250
)

# --- Assets Not Yet Uploaded Section ---
st.subheader("📁 Assets Not Uploaded ≥5 days")
asset_mask = sites_df["max_upload_delay"] >= 5
st.dataframe(
    sites_df[asset_mask][[s_site, s_subj, s_visit, s_assess, first_upload_col, "max_upload_delay"]],
    height=250
)

# --- Site Delay Frequency Chart ---
st.subheader("🔴 Site Delay Frequency (≥5 days)")
st.caption("Bars show count of assessments with asset or task delays ≥5 days.")
sd_long = site_delays.melt(
    id_vars=[s_site],
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
    st.write("### Recent Asset Uploads")
    dfA = asset_df.sort_values(a_upload, ascending=False).head(5)
    st.table(dfA[[a_site, a_subj, a_visit, a_assess, a_date, a_upload, "upload_delay"]])
with tabB:
    st.write("### Recent Form Submissions")
    dfB = forms_df.sort_values(f_submitted, ascending=False).head(5)
    st.table(dfB[[f_spid, f_created, f_submitted]])
with tabC:
    st.write("### Recent Assessment Completions")
    dfC = sites_df[sites_df[s_status_dt].notna()].sort_values(s_status_dt, ascending=False).head(5)
    st.table(dfC[[s_id, s_date, s_status_dt]])

st.success("✅ Dashboard loaded successfully!")
