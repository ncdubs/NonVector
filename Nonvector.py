import streamlit as st
import pandas as pd
import re
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Bulk SKU Similarity Finder (TF-IDF Cosine)")

# STEP 1: Paste or upload list of SKUs
st.header("Step 1: Enter Competitor SKUs")
uploaded_file = st.file_uploader("Upload Excel file with SKUs", type=["xlsx", "xls"])
pasted_data = st.text_area("Paste competitor SKU data here:")

def extract_skus_from_excel(df):
    all_text = df.astype(str).values.flatten()
    sku_pattern = re.compile(r"\b[A-Z]{2,}[0-9]{2,}[A-Z0-9]*\b")
    skus = []
    seen = set()
    for text in all_text:
        matches = sku_pattern.findall(text)
        for match in matches:
            if len(match) >= 6 and match not in seen:
                skus.append(match)
                seen.add(match)
    return skus

def extract_skus_from_text(text):
    sku_pattern = re.compile(r"\b[A-Z]{2,}[0-9]{2,}[A-Z0-9]*\b")
    skus = []
    seen = set()
    for line in text.upper().splitlines():
        matches = sku_pattern.findall(line)
        for sku in matches:
            if len(sku) >= 6 and sku not in seen:
                skus.append(sku)
                seen.add(sku)
    return skus

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

skus = []
if uploaded_file:
    df_upload = pd.read_excel(uploaded_file, header=None)
    skus = extract_skus_from_excel(df_upload)
elif pasted_data.strip():
    skus = extract_skus_from_text(pasted_data)

if skus:
    st.success(f"✅ Found {len(skus)} unique SKUs:")
    st.dataframe(pd.DataFrame({'SKU': skus}))
    excel_data = to_excel(pd.DataFrame({'SKU': skus}))
    st.download_button("Download SKUs to Excel", data=excel_data, file_name="sku_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# STEP 2: Upload catalog (memory-efficient, multi-sheet)
st.header("Step 2: Upload Appliance Catalog (Tall Format, Multiple Sheets OK)")
appliance_file = st.file_uploader(
    "Upload catalog Excel (tall, features as columns, multiple sheets allowed)", 
    type=["xlsx", "xls"], 
    key="appliance_upload"
)
if not appliance_file:
    st.stop()

# Read all sheets but do not concatenate
all_sheets = pd.read_excel(appliance_file, sheet_name=None)
sheet_lookup = {}
required = ['SKU', 'Brand', 'Model Status', 'Configuration']
for name, df_sheet in all_sheets.items():
    df_sheet.columns = [str(c).strip() for c in df_sheet.columns]
    if all(col in df_sheet.columns for col in required):
        for sku in df_sheet['SKU'].astype(str):
            sheet_lookup[sku] = name

results = []
vectorizers = {}
ge_tfidfs = {}
ge_dfs = {}
for sku in skus:
    sheet_name = sheet_lookup.get(sku)
    if not sheet_name:
        results.append({
            'Entered SKU': sku,
            'Closest GE SKU': 'Not found (SKU not in catalog)',
            'Matched GE Model Status': '',
            'Similarity Score': 0
        })
        continue
    sheet_df = all_sheets[sheet_name]
    sheet_df.columns = [str(c).strip() for c in sheet_df.columns]
    discard_cols = ['SKU', 'Brand', 'Model Status', 'combined_specs']
    all_features = [col for col in sheet_df.columns if col not in discard_cols]
    if 'combined_specs' not in sheet_df.columns:
        sheet_df['combined_specs'] = sheet_df[all_features].astype(str).agg(' '.join, axis=1)
    # Only build vectorizer per sheet
    if sheet_name not in vectorizers:
        vec = TfidfVectorizer()
        tfidf_mat = vec.fit_transform(sheet_df['combined_specs'])
        vectorizers[sheet_name] = vec
        # Only "GE" brand AND "active model"
        ge_mask = (sheet_df['Brand'].str.lower() == 'ge') & (sheet_df['Model Status'].str.lower() == 'active model')
        ge_df = sheet_df[ge_mask].reset_index(drop=True)
        nge_tfidf = vec.transform(ge_df['combined_specs'])
        ge_dfs[sheet_name] = ge_df
        ge_tfidfs[sheet_name] = nge_tfidf
    else:
        vec = vectorizers[sheet_name]
        ge_df = ge_dfs[sheet_name]
        nge_tfidf = ge_tfidfs[sheet_name]
    comp_row = sheet_df[sheet_df['SKU'] == sku]
    if comp_row.empty:
        results.append({
            'Entered SKU': sku,
            'Closest GE SKU': 'Not found (SKU not in sheet)',
            'Matched GE Model Status': '',
            'Similarity Score': 0
        })
        continue
    comp_config = comp_row.iloc[0]['Configuration']
    config_mask = ge_df['Configuration'].str.lower() == str(comp_config).lower()
    filtered_ge = ge_df[config_mask]
    filtered_ge_tfidf = nge_tfidf[config_mask.values]
    if filtered_ge.empty:
        results.append({
            'Entered SKU': sku,
            'Closest GE SKU': 'Not found (no GE match for config)',
            'Matched GE Model Status': '',
            'Similarity Score': 0
        })
        continue
    comp_tfidf = vec.transform([comp_row['combined_specs'].values[0]])
    sims = cosine_similarity(comp_tfidf, filtered_ge_tfidf)[0]
    if sims.max() == 0:
        results.append({
            'Entered SKU': sku,
            'Closest GE SKU': 'Not found (no similar GE model)',
            'Matched GE Model Status': '',
            'Similarity Score': 0
        })
    else:
        best_idx = sims.argmax()
        best_sku = filtered_ge.iloc[best_idx]['SKU']
        best_status = filtered_ge.iloc[best_idx]['Model Status']
        best_score = round(sims[best_idx], 3)
        results.append({
            'Entered SKU': sku,
            'Closest GE SKU': best_sku,
            'Matched GE Model Status': best_status,
            'Similarity Score': best_score
        })

results_df = pd.DataFrame(results)

st.subheader("Matching Results")
st.dataframe(results_df)

# Option to download the results table as Excel
if not results_df.empty:
    results_excel = to_excel(results_df)
    st.download_button("Download Matching Results to Excel", data=results_excel, file_name="matching_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
