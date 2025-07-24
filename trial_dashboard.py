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

# --- Helpers for Fuzzy Column Matching ---
def normalize(name: str) -> str:
    s = str(name).lower()
    s = re.sub(r'[^a-z0-9]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def col(df: pd.DataFrame, *cands) -> str | None:
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

# --- Debug: Show Detected Columns ---
with st.expander("🔧 Detected Columns"):
    st.write("Assets:", asset_df.columns.tolist())
    st.write("Forms:",  forms_df.columns.tolist())
    st.write("Sites:",  sites_df.columns.tolist())

# --- Identify Key Columns ---
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
rev_comment  = col(forms_df, "Review Comment", "ReviewComment")

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
        if c and c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

# --- Compute Asset Delays & Merge into Sites ---
asset_df["upload_delay"] = (asset_df[a_upload] - asset_df[a_date]).dt.days
first_up = "first_upload"
asset_agg = (
    asset_df
    .groupby([a_site, a_subj, a_visit, a_assess], as_index=False)
    .agg(
        **{first_up: (a_upload, "min")},
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

# --- Merge Forms by Assessment ID (conditional review_comment) ---
forms_merge_cols = [f_spid, f_submitted]
rename_map = {f_spid: s_id, f_submitted: "form_submitted"}
if rev_comment and rev_comment in forms_df.columns:
    forms_merge_cols.append(rev_comment)
    rename_map[rev_comment] = "review_comment"
forms_subset = forms_df[forms_merge_cols].rename(columns=rename_map)
sites_df = sites_df.merge(forms_subset, how="left", on=s_id)

# --- KPI Computations ---
today = pd.Timestamp(datetime.now().date())

# 1) Total Assessments & Subjects
total_assess = sites_df[s_id].nunique()
total_subj   = sites_df[s_subj].nunique()

# 2) In Progress
in_prog = sites_df[sites_df[s_status].str.lower()=="in progress"][s_id].nunique()

# 3) Avg Cycle Time (days)
comp_mask = sites_df[s_status_dt].notna()
avg_cycle = ((sites_df.loc[comp_mask, s_status_dt] - sites_df.loc[comp_mask, s_date]).dt.days).mean()

# 4) Assets Late >5 days
late_assets = sites_df[sites_df["max_upload_delay"] > 5][s_id].nunique()

# 5) Tasks Outstanding >5 days
sites_df["task_delay"] = sites_df[task_cols].apply(
    lambda r: (r - sites_df.loc[r.name, s_date]).dt.days.max(), axis=1
)
late_tasks = sites_df[sites_df["task_delay"] > 5][s_id].nunique()

# 6) Open Action Required
open_actions = sites_df[
    sites_df[act_raised].notna() & sites_df[act_resolved].isna()
][s_id].nunique()

# 7) Forms Late >5 days post-upload
sites_df["form_delay"] = (sites_df["form_submitted"] - sites_df[first_up]).dt.days
late_forms = sites_df[sites_df["form_delay"] > 5][s_id].nunique()

# --- Additional Metrics ---
# Composite Risk Score per site
flags = sites_df.groupby(s_site).agg(
    upload_late_rate=("max_upload_delay", lambda x: (x>5).sum()/len(x)),
    task_late_rate  =("task_delay",       lambda x: (x>5).sum()/len(x)),
    form_late_rate  =("form_delay",       lambda x: (x>5).sum()/len(x)),
    action_rate     =(act_raised,         lambda x: (x.notna() & sites_df.loc[x.index, act_resolved].isna()).sum()/len(x))
).reset_index()
flags["risk_score"] = flags[["upload_late_rate","task_late_rate","form_late_rate","action_rate"]].mean(axis=1)*100

# Visit-Window Adherence
offsets = {
    "Baseline":0, "Week 4":28, "Week 12":84,
    "Month 6":182, "Month 12":364
}
tolerances = {k:(3 if "Week" in k else 14) for k in offsets}
baseline = sites_df[sites_df[s_assess]=="Baseline"][[s_site,s_subj,s_date]].rename(columns={s_date:"baseline"})
vw = sites_df.merge(baseline, on=[s_site,s_subj], how="left")
vw["days_since_base"] = (vw[s_date] - vw["baseline"]).dt.days
vw["within_window"] = vw.apply(
    lambda r: abs(r["days_since_base"]-offsets.get(r[s_assess],0)) <= tolerances.get(r[s_assess],0),
    axis=1
)
adh = vw.groupby(s_assess).agg(
    total=("within_window","size"),
    on_time=("within_window","sum")
).assign(pct=lambda df: df["on_time"]/df["total"]*100).reset_index()

# Site Engagement Trend
trend = sites_df[comp_mask].copy()
trend["week"] = trend[s_status_dt].dt.to_period("W").apply(lambda p:p.start_time)
tr = trend.groupby([s_site,"week"]).size().reset_index(name="completions")

# Missing Data Summary
mv = sites_df.groupby([s_site,s_subj,s_visit]).size().reset_index(name="count")
missing = mv[mv["count"]<6]

# Action-Type Breakdown
if rev_comment and "review_comment" in sites_df.columns:
    top_reasons = sites_df["review_comment"].value_counts().head(3).reset_index()
    top_reasons.columns = ["reason","count"]

# Data-Quality Index
dq = flags.set_index(s_site)
miss_visit = missing.groupby(s_site).size()/sites_df.groupby(s_site)[s_id].nunique()
dq["miss_visit_rate"] = miss_visit.reindex(dq.index).fillna(0)
dq["dqi"] = (1 - dq[["upload_late_rate","task_late_rate","form_late_rate","action_rate","miss_visit_rate"]].mean(axis=1))*100
dq_df = dq.reset_index()

# --- Display Metrics ---
st.header("Key Metrics")

# Row 1
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Assessments", total_assess);       c1.caption("All planned assessments to date.")
c2.metric("Total Subjects",    total_subj);         c2.caption("Unique subjects enrolled.")
c3.metric("In Progress",       in_prog);            c3.caption("Assessments not yet Complete.")
c4.metric("Avg Cycle Time (d)",f"{avg_cycle:.1f}"); c4.caption("Mean days from assessment to status.")

# Row 2
c5,c6,c7,c8 = st.columns(4)
c5.metric("Assets Late (>5d)",    late_assets); c5.caption("Assets uploaded >5d post-assessment.")
c6.metric("Tasks Outstanding (>5d)", late_tasks); c6.caption("Any task >5d past assessment.")
c7.metric("Open Action Required",   open_actions); c7.caption("QC flags unresolved.")
c8.metric("Forms Late (>5d)",       late_forms); c8.caption("Forms submitted >5d after upload.")

st.markdown("---")

# Composite Risk Score Chart
st.subheader("🚨 Composite Risk Score per Site")
st.caption("Avg of delay rates (assets, tasks, forms, actions) ×100.")
risk_chart = alt.Chart(flags).mark_bar().encode(
    x=alt.X(f"{s_site}:N", title="Site"),
    y=alt.Y("risk_score:Q", title="Risk Score (%)"),
    color=alt.Color("risk_score:Q", scale=alt.Scale(scheme="redblue"), title="Risk")
)
st.altair_chart(risk_chart, use_container_width=True)

# Visit-Window Adherence Chart
st.subheader("✅ Visit-Window Adherence")
st.caption("Percentage of visits within protocol window.")
adh_chart = alt.Chart(adh).mark_bar().encode(
    x=alt.X(f"{s_assess}:N", title="Visit Type"),
    y=alt.Y("pct:Q", title="% Within Window"),
    tooltip=["on_time","total","pct"]
)
st.altair_chart(adh_chart, use_container_width=True)

# Site Engagement Trend Chart
st.subheader("📈 Site Engagement Trend (4-week)")
trend_chart = alt.Chart(tr).mark_line(point=True).encode(
    x="week:T",
    y="completions:Q",
    color=alt.Color(f"{s_site}:N", title="Site")
)
st.altair_chart(trend_chart, use_container_width=True)

# Missing Data Summary Table
st.subheader("⚠️ Missing Data Summary")
st.caption("Visits with fewer than 6 assessments recorded.")
st.dataframe(missing[[s_site,s_subj,s_visit,"count"]], height=250)

# Action-Type Breakdown
if rev_comment and "reason" in locals():
    st.subheader("🔍 Top 3 QC Issues")
    st.bar_chart(top_reasons.set_index("reason")["count"])

# Data-Quality Index Table
st.subheader("🏅 Data-Quality Index per Site")
st.caption("100×(1 − avg delay/error rates & missing visit rate).")
st.dataframe(dq_df[[s_site,"dqi"]].sort_values("dqi", ascending=False), height=250)

# Recent Activity Tabs
st.subheader("🕒 Most Recent Activity")
tabA,tabB,tabC = st.tabs(["Assets","Forms","Assessments"])
with tabA:
    st.write("### Recent Asset Uploads")
    dfA = asset_df.sort_values(a_upload, ascending=False).head(5)
    st.table(dfA[[a_site,a_subj,a_visit,a_assess,a_date,a_upload,"upload_delay"]])
with tabB:
    st.write("### Recent Form Submissions")
    dfB = forms_df.sort_values(f_submitted, ascending=False).head(5)
    st.table(dfB[[f_spid,f_created,f_submitted]])
with tabC:
    st.write("### Recent Assessment Completions")
    dfC = sites_df[comp_mask].sort_values(s_status_dt, ascending=False).head(5)
    st.table(dfC[[s_id,s_date,s_status_dt]])

st.success("✅ Enhanced dashboard loaded successfully!")\

