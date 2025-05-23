import streamlit as st
import pandas as pd
import spacy
import tempfile
import os
import re

# Create sidebar section for model info
st.sidebar.title("Model Information")

# Try to load the model
try:
    # Load the spaCy model - using only the small model for cloud compatibility
    nlp = spacy.load('en_core_web_sm')
    model_name = 'en_core_web_sm'
    st.sidebar.success(f"Using model: {model_name}")
    st.sidebar.write("Basic recognition of entities including PERSON, LOCATION, etc.")
except Exception as e:
    st.error(f"Error loading spaCy model: {str(e)}")
    st.error('Please ensure spaCy and en_core_web_sm are installed correctly.')
    st.stop()

st.title('CSV Anonomyzer')

# Create a model indicator at the top of the main UI
st.markdown(f"<div style='padding:10px; background-color:#FFF3CD; border-radius:5px; margin-bottom:15px;'>"
           f"<span style='font-weight:bold; color:orange;'>Active Model:</span> "
           f"<span style='font-weight:bold;'>Standard (Basic)</span>"
           f"</div>", unsafe_allow_html=True)

st.write('Upload a CSV file. All person names and locations will be replaced with "redacted".')

# Add a text input at the TOP of the app to allow adding a name to first_names.txt (permanently)
with st.form(key='add_name_form'):
    new_name = st.text_input('Add a name to first_names.txt (permanently):')
    add_clicked = st.form_submit_button('Add Name')
    if add_clicked:
        name = new_name.strip().lower()
        if name:
            with open('first_names.txt', 'a+') as f:
                f.seek(0)
                existing = set(line.strip().lower() for line in f)
                if name not in existing:
                    f.write(f"{name}\n")
                    st.success(f'Added "{name}" to first_names.txt. Please refresh the page to see the update.')
                else:
                    st.info(f'"{name}" is already in first_names.txt.')
        else:
            st.warning('Please enter a valid name.')

uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

import pandas as pd
import numpy as np

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
    
    # Skip empty strings
    if not text.strip():
        return text
    
    try:
        # Use a max length for processing to avoid memory issues
        MAX_LENGTH = 25000
        
        if len(text) > MAX_LENGTH:
            chunks = [text[i:i+MAX_LENGTH] for i in range(0, len(text), MAX_LENGTH)]
            processed_text = text
            for chunk in chunks:
                try:
                    doc = nlp(chunk)
                    # Apply entity redaction on this chunk
                    for ent in doc.ents:
                        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'FAC']:
                            processed_text = processed_text.replace(ent.text, 'REDACTED')
                except Exception as chunk_err:
                    st.sidebar.warning(f"Error processing chunk: {str(chunk_err)[:100]}...")
            text = processed_text
        else:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'FAC']:
                    text = text.replace(ent.text, 'REDACTED')
    except Exception as e:
        st.sidebar.error(f"NER processing error: {str(e)[:100]}...")
        # Continue with other redaction methods if NER fails
    
    # Regex for street addresses
    street_regex = r"\b\d{1,5}\s+([A-Za-z0-9.,'\-]+\s){1,4}(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Terrace|Ter|Place|Pl|Circle|Cir)\b"
    text = re.sub(street_regex, 'REDACTED', text, flags=re.IGNORECASE)
    
    # Redact standalone first names (case-insensitive, robust boundaries)
    if FIRST_NAMES:
        pattern = r'(?<!\w)(' + '|'.join(re.escape(name) for name in FIRST_NAMES) + r')(?!\w)'
        text = re.sub(pattern, 'REDACTED', text, flags=re.IGNORECASE)
    
    return text

# Debug: Show technical information and first names
with st.expander('Debug: Information'):
    # Model info
    st.subheader("Model Information")
    st.write(f"Active Model: {model_name}")
    
    # First names list info
    st.subheader("First Names Data")
    st.write(f"Total first names loaded: {len(FIRST_NAMES)}")
    st.write('Sample first names:', list(FIRST_NAMES)[:20])

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
    
    # Show progress
    progress_bar = st.progress(0)
    total_cols = len(redacted_df.select_dtypes(include=['object', 'string']).columns)
    
    for i, col in enumerate(redacted_df.select_dtypes(include=['object', 'string']).columns):
        redacted_df[col] = redacted_df[col].apply(lambda x: redact_text(x, nlp))
        # Update progress
        progress_bar.progress((i + 1) / total_cols)
    
    st.subheader('Redacted CSV Preview')
    # Transpose the DataFrame for vertical display
    transposed_df = redacted_df.transpose()
    # Use Styler to highlight 'REDACTED' in red
    st.write(transposed_df.style.format(highlight_redacted).to_html(escape=False), unsafe_allow_html=True)

    # Optionally, allow download (original orientation)
    csv = redacted_df.to_csv(index=False)
    st.download_button('Download Redacted CSV', csv, file_name='redacted.csv', mime='text/csv')
