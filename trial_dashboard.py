"""
Streamlit Dashboard for Clinical Trial Snapshot

- Upload Asset, Forms, and Sites Excel reports
- Auto-detect headers and validate columns
- Calculate key KPIs and display late uploads, late/missing forms
- Safely handle missing columns with user-friendly errors

Usage:
    streamlit run trial_dashboard.py

Dependencies:
    streamlit
    pandas
    numpy
    openpyxl
"""
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Trial Snapshot Dashboard", layout="wide")
st.title("📊 Trial Snapshot Dashboard")

# ----------------------
# 1. File Upload
# ----------------------
asset_file = st.file_uploader("Upload Asset Report", type=["xlsx"], key="asset")
forms_file = st.file_uploader("Upload Forms Report", type=["xlsx"], key="forms")
sites_file = st.file_uploader("Upload Sites Report", type=["xlsx"], key="sites")

if not all([asset_file, forms_file, sites_file]):
    st.info("⬆️ Please upload all three Excel reports to continue.")
    st.stop()

# ----------------------
# 2. Helper Functions
# ----------------------
def load_flex(excel_buffer):
    """Read Excel, detect header row by keywords"""
    raw = pd.read_excel(excel_buffer, header=None)
    hdr = 0
    for i in range(len(raw)):
        row = raw.iloc[i].astype(str).str.lower()
        if any(k in row.values for k in ["site", "subject", "assessment", "study procedure"]):
            hdr = i
            break
    df = pd.read_excel(excel_buffer, header=hdr).dropna(axis=1, how="all")
    df = df[df.notna().any(axis=1)]
    return df


def col(df, *cands):
    """Fuzzy match column name from candidates"""
    # exact match first
    for c in cands:
        for col in df.columns:
            if str(col).strip().lower() == c.lower():
                return col
    # partial match
    for c in cands:
        for col in df.columns:
            if c.lower() in str(col).strip().lower():
                return col
    return None

# Load dataframes
asset_df = load_flex(asset_file)
forms_df = load_flex(forms_file)
sites_df = load_flex(sites_file)

st.write("**Detected columns**:", {
    'Asset': list(asset_df.columns),
    'Forms': list(forms_df.columns),
    'Sites': list(sites_df.columns)
})

# ----------------------
# 3. Column Detection
# ----------------------
# Sites
assess_id_col_sites = col(sites_df, "Assessment ID", "Study Procedure ID")
assess_date_col_sites = col(sites_df, "Assessment Date", "Study Procedure Date")
status_col_sites = col(sites_df, "Assessment Status")
# Asset
upload_col = col(asset_df, "Upload Date", "Asset Upload Date")
asset_assess_date_col = col(asset_df, "Study Procedure Date", "Assessment Date", "Visit Date")
asset_assess_id_col = col(asset_df, "Assessment ID", "Study Procedure ID")
# Forms
forms_spid_col = col(forms_df, "Study Procedure ID", "Assessment ID")
forms_created_col = col(forms_df, "Date Created", "Form Created Date")
forms_sub_col = col(forms_df, "Submitted Date", "Form Submitted Date")

# ----------------------
# 4. Validate Columns
# ----------------------
missing = []
for name, reqs in [
    ("Asset report", [asset_assess_date_col, upload_col]),
    ("Sites report", [assess_id_col_sites, assess_date_col_sites]),
    ("Forms report", [forms_spid_col, forms_sub_col])
]:
    miss = [r for r in reqs if r is None]
    if miss:
        missing.append(f"{name}: missing {miss}")

if missing:
    st.error("Cannot compute metrics due to missing columns:\n" + "\n".join(missing))
    st.stop()

# ----------------------
# 5. Parse datetime
# ----------------------
for df, dcols in [
    (sites_df, [assess_date_col_sites]),
    (asset_df, [asset_assess_date_col, upload_col]),
    (forms_df, [forms_created_col, forms_sub_col])
]:
    for dc in dcols:
        df[dc] = pd.to_datetime(df[dc], errors='coerce')

# ----------------------
# 6. KPI Calculations
# ----------------------
# Sidebar thresholds
t1 = st.sidebar.number_input("Late upload threshold (days)", 1, 14, 3)
t2 = st.sidebar.number_input("Late form threshold (days)", 1, 14, 7)

# Asset delays
asset_df['upload_delay'] = (asset_df[upload_col] - asset_df[asset_assess_date_col]).dt.days
late_uploads = asset_df[asset_df['upload_delay'] > t1]

# Forms delays and missing
forms_map = forms_df[[forms_spid_col, forms_sub_col]].rename(columns={forms_spid_col: assess_id_col_sites})
site_forms = sites_df.merge(forms_map, on=assess_id_col_sites, how='left')
site_forms['form_delay'] = (site_forms[forms_sub_col] - site_forms[assess_date_col_sites]).dt.days
late_forms = site_forms[site_forms['form_delay'] > t2]
missing_forms = site_forms[site_forms[forms_sub_col].isna()]

# High-level metrics
total_assess = len(sites_df)
completed_assess = sites_df[sites_df[status_col_sites].str.contains("complete", case=False, na=False)].shape[0] if status_col_sites else None
on_time = 100 - (len(late_uploads) / len(asset_df) * 100) if len(asset_df) else 0

# Display metrics
c1, c2, c3 = st.columns(3)
c1.metric("Total Assessments", f"{total_assess}")
c2.metric("% Completed", f"{completed_assess/total_assess:.1%}" if completed_assess is not None else "–")
c3.metric("On-time Uploads", f"{on_time:.1f}%")

# ----------------------
# 7. Tabs View
# ----------------------
tab1, tab2, tab3 = st.tabs(["Late Uploads", "Late/Missing Forms", "Raw Data"])

with tab1:
    st.subheader("📂 Late Asset Uploads")
    cols = [c for c in [asset_assess_id_col, asset_assess_date_col, upload_col, 'upload_delay'] if c in asset_df]
    st.dataframe(late_uploads[cols])

with tab2:
    st.subheader("📝 Late or Missing Forms")
    st.write("**Late Forms**")
    lcols = [c for c in [assess_id_col_sites, assess_date_col_sites, forms_sub_col, 'form_delay'] if c in site_forms]
    st.dataframe(late_forms[lcols])
    st.write("**Missing Forms**")
    mcols = [c for c in [assess_id_col_sites, assess_date_col_sites] if c in missing_forms]
    st.dataframe(missing_forms[mcols])

with tab3:
    st.write("### Sites Report")
    st.dataframe(sites_df)
    st.write("### Asset Report")
    st.dataframe(asset_df)
    st.write("### Forms Report")
    st.dataframe(forms_df)

st.success("Dashboard loaded. Adjust thresholds in the sidebar.")
