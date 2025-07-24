import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re
from datetime import datetime, timedelta

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="Clinical Trial Dashboard")
st.title("📊 Clinical Trial Snapshot")

# --- Sidebar: Upload the three reports ---
st.sidebar.header("Upload Reports")
asset_buf = st.sidebar.file_uploader("Asset Report (.xlsx)", type="xlsx")
forms_buf = st.sidebar.file_uploader("Forms Report (.xlsx)", type="xlsx")
sites_buf = st.sidebar.file_uploader("Sites Report (.xlsx)", type="xlsx")
if not (asset_buf and forms_buf and sites_buf):
    st.sidebar.info("Upload all three reports to begin.")
    st.stop()

# --- Helper: Fuzzy column detection ---
def normalize(name: str) -> str:
    s = str(name).lower()
    s = re.sub(r'[^a-z0-9]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def col(df: pd.DataFrame, *candidates) -> str | None:
    norm_map = {normalize(c): c for c in df.columns}
    # exact first
    for cand in candidates:
        nc = normalize(cand)
        if nc in norm_map:
            return norm_map[nc]
    # then partial
    for cand in candidates:
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

# --- Load the three workbooks ---
asset_df = load_df(asset_buf)
forms_df = load_df(forms_buf)
sites_df = load_df(sites_buf)

# --- Debug: show detected columns ---
with st.expander("🔧 Detected Columns"):
    st.write("Assets:", asset_df.columns.tolist())
    st.write("Forms:",  forms_df.columns.tolist())
    st.write("Sites:",  sites_df.columns.tolist())

# --- Identify key columns ---
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

# --- Validate presence of required columns ---
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

# --- Parse dates ---
for df, cols in [
    (sites_df, [s_date, s_status_dt, act_raised, act_resolved] + task_cols),
    (asset_df, [a_date, a_upload]),
    (forms_df, [f_created, f_submitted])
]:
    for c in cols:
        if c and c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

# --- Compute upload delays & merge into sites ---
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

# --- Merge forms (conditionally include review_comment) ---
forms_merge = [f_spid, f_submitted]
rename_map = {f_spid: s_id, f_submitted: "form_submitted"}
if rev_comment and rev_comment in forms_df.columns:
    forms_merge.append(rev_comment)
    rename_map[rev_comment] = "review_comment"
forms_subset = forms_df[forms_merge].rename(columns=rename_map)
sites_df = sites_df.merge(forms_subset, how="left", on=s_id)

# --- Basic KPI calculations ---
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
open_actions = sites_df[
    sites_df[act_raised].notna() & sites_df[act_resolved].isna()
][s_id].nunique()
sites_df["form_delay"] = (sites_df["form_submitted"] - sites_df[first_up_col]).dt.days
late_forms   = sites_df[sites_df["form_delay"]>5][s_id].nunique()

# --- Display restored KPIs ---
st.header("Key Metrics")
r1c1, r1c2, r1c3, r1c4 = st.columns(4)
r1c1.metric("Total Assessments",    total_assess); r1c1.caption("All planned assessments to date.")
r1c2.metric("Total Subjects",       total_subj);   r1c2.caption("Unique subjects enrolled.")
r1c3.metric("In Progress",          in_prog);      r1c3.caption("Assessments not yet Complete.")
r1c4.metric("Avg Cycle Time (days)", f"{avg_cycle:.1f}"); r1c4.caption("Mean days from assessment → status.")

r2c1, r2c2, r2c3, r2c4 = st.columns(4)
r2c1.metric("Assets Late (>5d)",        late_assets);   r2c1.caption("Assets uploaded >5 days late.")
r2c2.metric("Tasks Outstanding (>5d)",  late_tasks);    r2c2.caption("Any task >5 days past visit.")
r2c3.metric("Open Action Required",     open_actions);  r2c3.caption("QC flags raised but unresolved.")
r2c4.metric("Forms Late (>5d)",         late_forms);    r2c4.caption("Forms submitted >5 days after upload.")

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
st.caption("Visits where fewer than 6 assessments are recorded.")
md = sites_df.groupby([s_site,s_subj,s_visit]).size().reset_index(name="Count")
missing = md[md["Count"]<6]
st.metric("Visits Missing Assessments", missing.shape[0])
st.dataframe(missing[[s_site,s_subj,s_visit,"Count"]], height=250)

# --- ⏱️ Visit‑Window Adherence (outside window) ---
st.subheader("⏱️ Visits Outside Allowed Window")
st.caption("Week 4 ±10 days; Month 6 ±14 days; others no tolerance.")
# get baseline per subject
baseline = sites_df[sites_df[s_assess]=="Baseline"][[s_site,s_subj,s_date]].rename(columns={s_date:"baseline"})
vw = sites_df.merge(baseline, on=[s_site,s_subj], how="left")
vw["days_from_base"] = (vw[s_date] - vw["baseline"]).dt.days
# determine tolerance
def out_of_window(row):
    name = row[s_assess]
    tol = 10 if "week" in name.lower() else 14 if "month" in name.lower() else 0
    offset = {"Baseline":0, "Week 4":28, "Month 6":182}.get(name, 0)
    return abs(row["days_from_base"] - offset) > tol

vw["out_of_window"] = vw.apply(out_of_window, axis=1)
outside_pct = vw["out_of_window"].mean() * 100
st.metric("% Visits Outside Window", f"{outside_pct:.1f}%")

# --- 🏴‍☠️ Late Sites (assessments w/o assets >5d) ---
st.subheader("🏴‍☠️ Assessments with Asset Delay >5 days")
st.caption("Grouped by site.")
late_df = sites_df[sites_df["max_upload_delay"]>5][[s_site,s_subj,s_visit,s_assess,s_date, a_upload, "max_upload_delay"]]
# rename a_upload column for display
late_df = late_df.rename(columns={a_upload:"Upload Date"})
st.dataframe(late_df.sort_values(s_site), height=300)

# --- Recent Activity Tabs (unchanged) ---
st.subheader("🕒 Most Recent Activity")
tabA, tabB, tabC = st.tabs(["Assets","Forms","Assessments"])
with tabA:
    st.write("### Recent Asset Uploads")
    ra = asset_df.sort_values(a_upload, ascending=False).head(5)
    st.table(ra[[a_site,a_subj,a_visit,a_assess,a_date,a_upload,"upload_delay"]])
with tabB:
    st.write("### Recent Form Submissions")
    rf = forms_df.sort_values(f_submitted, ascending=False).head(5)
    st.table(rf[[f_spid,f_created,f_submitted]])
with tabC:
    st.write("### Recent Assessment Completions")
    rc = sites_df[comp_mask].sort_values(s_status_dt, ascending=False).head(5)
    st.table(rc[[s_id,s_date,s_status_dt]])

st.success("✅ Dashboard updated with additional metrics!")
