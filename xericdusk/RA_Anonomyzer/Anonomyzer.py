import streamlit as st
import pandas as pd
import spacy
import tempfile
import os

# Load spaCy model for NER
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    st.error('The spaCy model en_core_web_sm is not installed. Please add it to requirements.txt as a direct URL. See README for details.')
    st.stop()

st.title('CSV Anonomyzer')
st.write('Upload a CSV file. All person names and locations will be replaced with "redacted".')

# Add a text input at the TOP of the app to allow adding a name to first_names.txt (permanently)
new_name = st.text_input('Add a name to first_names.txt (permanently):')
if new_name:
    with open('first_names.txt', 'a') as f:
        f.write(new_name + '\n')
    st.experimental_rerun()

uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

import pandas as pd
import numpy as np

import re

def load_first_names(txt_path='first_names.txt', csv_path='common_first_names.csv'):
    names = set()
    # Load from txt
    try:
        with open(txt_path, 'r') as f:
            names.update(name.strip().lower() for name in f if name.strip())
    except Exception:
        pass
    # Load from csv
    try:
        import csv
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row and row[0].strip():
                    names.add(row[0].strip().lower())
    except Exception:
        pass
    return names

FIRST_NAMES = load_first_names()

def redact_text(text, nlp):
    # Handle NaN or missing values
    if pd.isna(text):
        return text
    text = str(text)
    # SpaCy NER redaction
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'GPE', 'LOC']:
            text = text.replace(ent.text, 'REDACTED')
    # Regex for street addresses (simple version)
    street_regex = r"\b\d{1,5}\s+([A-Za-z0-9.,'\-]+\s){1,4}(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Terrace|Ter|Place|Pl|Circle|Cir)\b"
    text = re.sub(street_regex, 'REDACTED', text, flags=re.IGNORECASE)
    # Redact standalone first names (case-insensitive, robust boundaries)
    def redact_first_name(match):
        return 'REDACTED'
    if FIRST_NAMES:
        pattern = r'(?<!\w)(' + '|'.join(re.escape(name) for name in FIRST_NAMES) + r')(?!\w)'
        text = re.sub(pattern, redact_first_name, text, flags=re.IGNORECASE)
    return text

# Debug: Show first 20 first names and regex pattern in Streamlit
with st.expander('Debug: First Names and Regex Pattern'):
    st.write('Sample first names:', list(FIRST_NAMES)[:20])
    if FIRST_NAMES:
        pattern = r'(?<!\w)(' + '|'.join(re.escape(name) for name in FIRST_NAMES) + r')(?!\w)'
        st.write('Regex pattern:', pattern)
    else:
        pattern = ''

# Add a test field for live redaction
with st.expander('Test Redaction'):
    test_input = st.text_input('Type a sample string to see redaction:')
    if test_input:
        st.write('Redacted:', redact_text(test_input, nlp))


def highlight_redacted(val):
    if isinstance(val, str) and 'REDACTED' in val:
        # Highlight all occurrences of REDACTED in red
        return val.replace('REDACTED', '<span style="color: red; font-weight: bold;">REDACTED</span>')
    return val


if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    # Redact all string columns
    redacted_df = df.copy()
    for col in redacted_df.select_dtypes(include=['object', 'string']).columns:
        redacted_df[col] = redacted_df[col].apply(lambda x: redact_text(x, nlp))
    st.subheader('Redacted CSV Preview')
    # Transpose the DataFrame for vertical display
    transposed_df = redacted_df.transpose()
    # Use Styler to highlight 'REDACTED' in red
    st.write(transposed_df.style.format(highlight_redacted).to_html(escape=False), unsafe_allow_html=True)

    # Optionally, allow download (original orientation)
    csv = redacted_df.to_csv(index=False)
    st.download_button('Download Redacted CSV', csv, file_name='redacted.csv', mime='text/csv')
