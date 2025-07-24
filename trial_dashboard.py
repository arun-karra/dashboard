import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re
from datetime import datetime, timedelta

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="Clinical Trial Dashboard")
st.title("📊 Clinical Trial Snapshot")

# --- Sidebar: Upload Reports ---
st.sidebar.header("Upload Reports")
asset_buf = st.sidebar.file_uploader("Asset Report (.xlsx)", type="xlsx")
forms_buf = st.sidebar.file_uploader("Forms Report (.xlsx)", type="xlsx")
sites_buf = st.sidebar.file_uploader("Sites Report (.xlsx)", type="xlsx")
if not (asset_buf and forms_buf and sites_buf):
    st.sidebar.info("Upload all three reports to proceed.")
    st.stop()

# --- Helpers for Column Matching ---
def normalize(name: str) -> str:
    s = str(name).lower()
    s = re.sub(r'[^a-z0-9]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def col(df: pd.DataFrame, *cands) -> str | None:
    norm_map = {normalize(c): c for c in df.columns}
    # exact match
    for cand in cands:
        nc = normalize(cand)
        if nc in norm_map:
            return norm_map[nc]
    # fuzzy match
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

# --- Debug: Detected Columns ---
with st.expander("🔧 Detected Columns"):
    st.write("Assets:", asset_df.columns.tolist())
    st.write("Forms:",  forms_df.columns.tolist())
    st.write("Sites:",  sites_df.columns.tolist())

# --- Identify Key Columns ---
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
rev_comment  = col(forms_df, "Review Comment", "ReviewComment")

# --- Validate Required Columns ---
required = {
    "Sites":  [s_site, s_subj, s_visit, s_assess, s_date],
    "Assets": [a_site, a_subj, a_visit, a_assess, a_date, a_upload],
    "Forms":  [f_spid, f_submitted],
}
errors = []
for name, cols in required.items():
    miss = [c for c in cols if c is None]
    if miss:
        errors.append(f"{name}: missing {miss}")
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
first_up_col = "first_upload"
asset_agg = (
    asset_df
    .groupby([a_site, a_subj, a_visit, a_assess], as_index=False)
    .agg(
        **{first_up_col: (a_upload, "min")},
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

# --- Merge Forms (include ReviewComment if available) ---
forms_merge = [f_spid, f_submitted]
rename_map = {f_spid: s_id, f_submitted: "form_submitted"}
if rev_comment and rev_comment in forms_df.columns:
    forms_merge.append(rev_comment)
    rename_map[rev_comment] = "review_comment"
forms_subset = forms_df[forms_merge].rename(columns=rename_map)
sites_df = sites_df.merge(forms_subset, how="left", on=s_id)

# --- Core KPI Calculations ---
today = pd.Timestamp(datetime.now().date())
total_assess = sites_df[s_id].nunique()
total_subj   = sites_df[s_subj].nunique()
in_prog      = sites_df[sites_df[s_status].str.lower()=="in progress"][s_id].nunique()
comp_mask    = sites_df[s_status_dt].notna()
avg_cycle    = ((sites_df.loc[comp_mask, s_status_dt] - sites_df.loc[comp_mask, s_date]).dt.days).mean()
late_assets  = sites_df[sites_df["max_upload_delay"]>5][s_id].nunique()
sites_df["task_delay"] = sites_df[task_cols].apply(
    lambda r: (r - sites_df.loc[r.name, s_date]).dt.days.max(), axis=1
)
late_tasks   = sites_df[sites_df["task_delay"]>5][s_id].nunique()
open_actions = sites_df[sites_df[act_raised].notna() & sites_df[act_resolved].isna()][s_id].nunique()
sites_df["form_delay"] = (sites_df["form_submitted"] - sites_df[first_up_col]).dt.days
late_forms   = sites_df[sites_df["form_delay"]>5][s_id].nunique()

# --- Display Restored KPIs ---
st.header("Key Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Assessments",    total_assess);   c1.caption("All planned assessments to date.")
c2.metric("Total Subjects",       total_subj);     c2.caption("Unique subjects enrolled.")
c3.metric("In Progress",          in_prog);        c3.caption("Assessments not yet Complete.")
c4.metric("Avg Cycle Time (days)", f"{avg_cycle:.1f}"); c4.caption("Mean days from assessment to status.")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Assets Late (>5d)",        late_assets); c5.caption("Assets uploaded >5 days late.")
c6.metric("Tasks Outstanding (>5d)",  late_tasks);  c6.caption("Any task >5 days past assessment.")
c7.metric("Open Action Required",     open_actions);c7.caption("QC flags raised but unresolved.")
c8.metric("Forms Late (>5d)",         late_forms);  c8.caption("Forms submitted >5 days after upload.")

st.markdown("---")

# --- 🚩 Action‑Type Breakdown ---
if rev_comment and "review_comment" in sites_df.columns:
    st.subheader("🔍 Top 3 QC Issue Reasons")
    top3 = sites_df["review_comment"].value_counts().head(3).reset_index()
    top3.columns = ["Reason","Count"]
    chart = alt.Chart(top3).mark_bar().encode(
        x=alt.X("Reason:N", title="Reason"),
        y=alt.Y("Count:Q", title="Frequency"),
        tooltip=["Reason","Count"]
    )
    st.altair_chart(chart, use_container_width=True)

# --- ⚠️ Missing Data Summary ---
st.subheader("⚠️ Missing Data Summary")
st.caption("Visits missing one or more of the 6 assessments.")
md = sites_df.groupby([s_site,s_subj,s_visit]).size().reset_index(name="Count")
missing = md[md["Count"]<6]
st.metric("Visits Missing Assessments", missing.shape[0])
st.dataframe(missing[[s_site,s_subj,s_visit,"Count"]], height=250)

# --- ⏱️ Visit‑Window Adherence (outside window) ---
st.subheader("⏱️ Visits Outside Allowed Window")
st.caption("Week 4 ±10 days; Month 6 ±14 days.")
baseline = sites_df[sites_df[s_assess]=="Baseline"][[s_site,s_subj,s_date]].rename(columns={s_date:"baseline"})
vw = sites_df.merge(baseline, on=[s_site,s_subj], how="left")
vw["days_from_base"] = (vw[s_date] - vw["baseline"]).dt.days
def is_out(row):
    name = row[s_assess]
    tol = 10 if "week" in name.lower() else 14 if "month" in name.lower() else 0
    offset = {"Baseline":0,"Week 4":28,"Month 6":182}.get(name,0)
    return abs(row["days_from_base"] - offset) > tol
vw["out_of_window"] = vw.apply(is_out, axis=1)
pct_out = vw["out_of_window"].mean() * 100
st.metric("% Visits Outside Window", f"{pct_out:.1f}%")

# --- 🏴‍☠️ Late Sites Table ---
st.subheader("🏴‍☠️ Assessments with Asset Delay >5 days")
st.caption("Grouped by site.")
late_df = sites_df[sites_df["max_upload_delay"]>5][
    [s_site, s_subj, s_visit, s_assess, s_date, first_up_col, "max_upload_delay"]
].rename(columns={
    first_up_col: "First Upload Date",
    "max_upload_delay": "Upload Delay (days)"
})
st.dataframe(late_df.sort_values(s_site), height=300)

# --- Recent Activity Tabs ---
st.subheader("🕒 Most Recent Activity")
tA, tB, tC = st.tabs(["Assets","Forms","Assessments"])
with tA:
    st.write("### Recent Asset Uploads")
    ra = asset_df.sort_values(a_upload, ascending=False).head(5)
    st.table(ra[[a_site,a_subj,a_visit,a_assess,a_date,a_upload,"upload_delay"]])
with tB:
    st.write("### Recent Form Submissions")
    rf = forms_df.sort_values(f_submitted, ascending=False).head(5)
    st.table(rf[[f_spid,f_created,f_submitted]])
with tC:
    st.write("### Recent Assessment Completions")
    rc = sites_df[comp_mask].sort_values(s_status_dt, ascending=False).head(5)
    st.table(rc[[s_id,s_date,s_status_dt]])

st.success("✅ Dashboard updated with requested metrics!")
