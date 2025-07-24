"""
Simple Streamlit dashboard for CRO/Sponsor trial snapshot.
Drop the three exported Excel reports (Asset, Forms, Sites) and the app
parses them, aligns dates, calculates key KPIs, and renders visuals.
Run locally:
    streamlit run trial_dashboard.py
Dependencies: streamlit, pandas, numpy, openpyxl
"""

import io
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Trial Snapshot Dashboard", layout="wide")
st.title("📊 Trial Snapshot Dashboard")

# ------------------------------------------------------
# 1. File uploaders
# ------------------------------------------------------
asset_file = st.file_uploader("Upload Asset Report", type=["xlsx"], key="asset")
forms_file = st.file_uploader("Upload Forms Report", type=["xlsx"], key="forms")
sites_file = st.file_uploader("Upload Sites Report", type=["xlsx"], key="sites")

if not all([asset_file, forms_file, sites_file]):
    st.info("⬆️ Upload the three Excel reports to begin.")
    st.stop()

# Helper to read Excel with flexible header detection
def load_flex(path_or_buf):
    raw = pd.read_excel(path_or_buf, header=None)
    hdr_idx = 0
    for i in range(len(raw)):
        row = raw.iloc[i].astype(str).str.lower()
        if any(k in row.values for k in ["site", "subject", "assessment", "study procedure"]):
            hdr_idx = i
            break
    df = pd.read_excel(path_or_buf, header=hdr_idx).dropna(axis=1, how="all")
    df = df[df.notna().any(axis=1)]
    return df

asset_df = load_flex(asset_file)
forms_df = load_flex(forms_file)
sites_df = load_flex(sites_file)

# ------------------------------------------------------
# 2. Fuzzy column matching helpers
# ------------------------------------------------------

def col(df, *cands):
    for c in cands:
        for colname in df.columns:
            if c.lower() == str(colname).strip().lower():
                return colname
    for c in cands:
        for colname in df.columns:
            if c.lower() in str(colname).strip().lower():
                return colname
    return None

assess_id_col_sites = col(sites_df, "Assessment ID")
assess_date_col_sites = col(sites_df, "Assessment Date", "Study Procedure Date")

upload_col = col(asset_df, "Upload Date")
asset_assess_date_col = col(asset_df, "Study Procedure Date", "Assessment Date")
asset_assess_id_col = col(asset_df, "Assessment ID")

forms_spid_col = col(forms_df, "Study Procedure ID")
forms_created_col = col(forms_df, "Date Created")
forms_sub_col = col(forms_df, "Submitted Date")

# Ensure datetime
for df, dcols in [
    (sites_df, [assess_date_col_sites]),
    (asset_df, [upload_col, asset_assess_date_col]),
    (forms_df, [forms_created_col, forms_sub_col]),
]:
    for dc in dcols:
        if dc and dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors="coerce")

# ------------------------------------------------------
# 3. KPI calculations
# ------------------------------------------------------
late_upload_threshold = st.sidebar.number_input("Late upload threshold (days)", 1, 14, 3)
form_delay_threshold = st.sidebar.number_input("Late form threshold (days)", 1, 14, 7)

# Late uploads
if upload_col and asset_assess_date_col:
    asset_df["upload_delay"] = (asset_df[upload_col] - asset_df[asset_assess_date_col]).dt.days
late_uploads = asset_df[asset_df["upload_delay"] > late_upload_threshold]

# Join forms to sites for missing / delayed forms
forms_date_map = forms_df[[forms_spid_col, forms_sub_col]].rename(columns={forms_spid_col: assess_id_col_sites})
site_forms = sites_df.merge(forms_date_map, on=assess_id_col_sites, how="left")
site_forms["form_delay"] = (site_forms[forms_sub_col] - site_forms[assess_date_col_sites]).dt.days

late_forms = site_forms[site_forms["form_delay"] > form_delay_threshold]
missing_forms = site_forms[site_forms[forms_sub_col].isna()]

# High‑level metrics
total_assessments = len(sites_df)
completed_assessments = sites_df[sites_df["Assessment Status"].str.contains("complete", case=False, na=False)].shape[0] if col(sites_df, "Assessment Status") else np.nan
on_time_upload_rate = 100 - (len(late_uploads) / len(asset_df) * 100)

# ------------------------------------------------------
# 4. Layout – Metrics
# ------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Assessments", f"{total_assessments}")
col2.metric("% Completed", f"{completed_assessments/total_assessments:.1%}" if pd.notna(completed_assessments) else "–")
col3.metric("On‑time Uploads", f"{on_time_upload_rate:.1f}%")

# ------------------------------------------------------
# 5. Tabs for deep dives
# ------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Late Uploads", "Late/Missing Forms", "Raw Tables"])

with tab1:
    st.subheader("📂 Late Asset Uploads (> threshold)")
    st.dataframe(late_uploads[[asset_assess_id_col, asset_assess_date_col, upload_col, "upload_delay"]])

with tab2:
    st.subheader("📝 Late or Missing Forms")
    st.write("### Late Forms (> threshold)")
    st.dataframe(late_forms[[assess_id_col_sites, assess_date_col_sites, forms_sub_col, "form_delay"]])
    st.write("### Missing Forms")
    st.dataframe(missing_forms[[assess_id_col_sites, assess_date_col_sites]])

with tab3:
    st.write("### Sites Report")
    st.dataframe(sites_df)
    st.write("### Asset Report")
    st.dataframe(asset_df)
    st.write("### Forms Report")
    st.dataframe(forms_df)

st.success("Dashboard generated – adjust thresholds in sidebar to explore.")
