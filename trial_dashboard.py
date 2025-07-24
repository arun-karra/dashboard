import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re
from datetime import datetime

st.set_page_config(layout="wide", page_title="Clinical Trial Dashboard")
st.title("📊 Clinical Trial Snapshot")

# Sidebar uploads
st.sidebar.header("Upload Reports")
asset_buf = st.sidebar.file_uploader("Asset Report (.xlsx)", type="xlsx")
forms_buf = st.sidebar.file_uploader("Forms Report (.xlsx)", type="xlsx")
sites_buf = st.sidebar.file_uploader("Sites Report (.xlsx)", type="xlsx")
if not (asset_buf and forms_buf and sites_buf):
    st.sidebar.info("Please upload Asset, Forms, and Sites reports.")
    st.stop()

# Helpers
def normalize(name):
    s = str(name).lower()
    s = re.sub(r'[^a-z0-9]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def col(df, *cands):
    norm_map = {normalize(c): c for c in df.columns}
    for cand in cands:
        nc = normalize(cand)
        if nc in norm_map:
            return norm_map[nc]
    for cand in cands:
        nc = normalize(cand)
        for nm, orig in norm_map.items():
            if nc in nm:
                return orig
    return None

@st.cache_data(show_spinner=False)
def load_df(buf):
    raw = pd.read_excel(buf, header=None)
    hdr = 0
    for i in range(len(raw)):
        row = raw.iloc[i].astype(str).str.lower()
        if "subject number" in row.values and "study procedure" in row.values:
            hdr = i
            break
    df = pd.read_excel(buf, header=hdr).dropna(axis=1, how="all")
    return df[df.notna().any(axis=1)]

# Load
asset_df = load_df(asset_buf)
forms_df = load_df(forms_buf)
sites_df = load_df(sites_buf)

# Debug columns
with st.expander("🔧 Detected Columns"):
    st.write("Assets:", list(asset_df.columns))
    st.write("Forms:", list(forms_df.columns))
    st.write("Sites:", list(sites_df.columns))

# Identify columns
# Sites
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

# Asset
a_site       = col(asset_df, "Library/Site Name", "Site Name")
a_subj       = col(asset_df, "Subject Number")
a_visit      = col(asset_df, "Study Event", "Visit Name")
a_assess     = col(asset_df, "Study Procedure", "Assessment Name")
a_date       = col(asset_df, "Study Procedure Date", "Assessment Date")
a_upload     = col(asset_df, "Upload Date")

# Forms
f_spid       = col(forms_df, "Study Procedure ID", "Assessment ID")
f_created    = col(forms_df, "Date Created", "Form Created Date")
f_submitted  = col(forms_df, "Submitted Date", "Form Submitted Date")

# Validate
reqs = {
    "Sites": [s_site, s_subj, s_visit, s_assess, s_date],
    "Assets": [a_site, a_subj, a_visit, a_assess, a_date, a_upload],
    "Forms": [f_spid, f_submitted]
}
errs = []
for name, cols in reqs.items():
    m = [c for c in cols if c is None]
    if m:
        errs.append(f"{name} missing {m}")
if errs:
    st.error("Missing required columns:\n" + "\n".join(errs))
    st.stop()

# Parse dates
for df, cols in [
    (sites_df, [s_date, s_status_dt, act_raised, act_resolved] + task_cols),
    (asset_df, [a_date, a_upload]),
    (forms_df, [f_created, f_submitted])
]:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

# Asset delays per assessment by 4‑key join
asset_df["upload_delay"] = (asset_df[a_upload] - asset_df[a_date]).dt.days
asset_agg = (
    asset_df
    .groupby([a_site, a_subj, a_visit, a_assess], as_index=False)
    .agg(
        first_upload_date=(a_upload, "min"),
        max_upload_delay =("upload_delay", "max")
    )
)
sites_df = sites_df.merge(
    asset_agg,
    how="left",
    left_on=[s_site, s_subj, s_visit, s_assess],
    right_on=[a_site, a_subj, a_visit, a_assess],
)
sites_df["max_upload_delay"] = sites_df["max_upload_delay"].fillna(0).astype(int)

# Forms merge by Assessment ID
sites_df = sites_df.merge(
    forms_df[[f_spid, f_submitted]].rename(columns={f_spid: s_id, f_submitted: "form_submitted"}),
    how="left",
    on=s_id
)

# KPI computations
total_assess = sites_df[s_id].nunique()
total_subj   = sites_df[s_subj].nunique()
in_prog      = sites_df[sites_df[s_status].str.lower()=="in progress"][s_id].nunique()
completed    = sites_df[sites_df["form_submitted"].notna()]  # use forms or status?
avg_cycle    = ((sites_df.loc[sites_df[s_status_dt].notna(), s_status_dt]
               - sites_df.loc[sites_df[s_status_dt].notna(), s_date]).dt.days).mean()

late_assets  = sites_df[sites_df["max_upload_delay"] > 5][s_id].nunique()
sites_df["task_delay"] = sites_df[task_cols].apply(
    lambda row: (row - sites_df.loc[row.name, s_date]).dt.days.max(), axis=1
)
late_tasks   = sites_df[sites_df["task_delay"] > 5][s_id].nunique()
open_actions = sites_df[sites_df[act_raised].notna() & sites_df[act_resolved].isna()][s_id].nunique()

# Forms delay calculated vs first upload
sites_df["form_delay"] = (sites_df["form_submitted"] - sites_df["first_upload_date"]).dt.days
late_forms   = sites_df[sites_df["form_delay"] > 5][s_id].nunique()

# Site delay counts
site_delays = sites_df.groupby(s_site).agg(
    assets_late = ("max_upload_delay", lambda x: (x>5).sum()),
    tasks_late  = ("task_delay",      lambda x: (x>5).sum())
).reset_index()

# Display metrics
st.header("Key Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Assessments", total_assess)
c2.metric("Total Subjects", total_subj)
c3.metric("In Progress", in_prog)
c4.metric("Avg Cycle Time (days)", f"{avg_cycle:.1f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Assess w/o Assets >5d", late_assets)
c6.metric("Tasks >5d", late_tasks)
c7.metric("Open Action Required", open_actions)
c8.metric("Forms >5d post-upload", late_forms)

st.markdown("---")

# Site delays chart
st.subheader("🔴 Site Delay Frequency")
sd_long = site_delays.melt(id_vars=[s_site], var_name="Type", value_name="Count")
chart = (
    alt.Chart(sd_long)
    .mark_bar()
    .encode(
        x=alt.X(f"{s_site}:N", sort=None),
        y="Count:Q",
        color="Type:N",
        column=alt.Column("Type:N", title=None)
    )
    .properties(width=150)
)
st.altair_chart(chart, use_container_width=True)

# Recent activity
st.subheader("🕒 Recent Activity")
tA, tB, tC = st.tabs(["Assets","Forms","Assessments"])
with tA:
    st.write("#### Asset Uploads")
    dfA = asset_df.sort_values(a_upload, ascending=False).head(5)
    st.table(dfA[[a_site, a_subj, a_visit, a_assess, a_date, a_upload, "upload_delay"]])
with tB:
    st.write("#### Form Submissions")
    dfB = forms_df.sort_values(f_submitted, ascending=False).head(5)
    st.table(dfB[[f_spid, f_created, f_submitted]])
with tC:
    st.write("#### Assessment Completions")
    dfC = sites_df[sites_df[s_status_dt].notna()].sort_values(s_status_dt, ascending=False).head(5)
    st.table(dfC[[s_id, s_date, s_status_dt]])

st.success("✅ Dashboard ready! Adjust sidebar thresholds as needed.")
