import streamlit as st
import pandas as pd
import spacy
import tempfile
import os

# Load spaCy model for NER (download if necessary)
def load_spacy_model():
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        import subprocess
        subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
        nlp = spacy.load('en_core_web_sm')
    return nlp

nlp = load_spacy_model()

st.title('CSV Anonomyzer')
st.write('Upload a CSV file. All person names and locations will be replaced with "redacted".')

uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

import pandas as pd
import numpy as np

import re

def redact_text(text, nlp):
    # Handle NaN or missing values
    if pd.isna(text):
        return text
    text = str(text)
    # Regex for street addresses (simple version)
    street_regex = r"\b\d{1,5}\s+([A-Za-z0-9.,'\-]+\s){1,4}(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Terrace|Ter|Place|Pl|Circle|Cir)\b"
    redacted = re.sub(street_regex, 'REDACTED', text, flags=re.IGNORECASE)
    doc = nlp(redacted)
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'GPE', 'LOC']:
            redacted = redacted.replace(ent.text, 'REDACTED')
    return redacted

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
