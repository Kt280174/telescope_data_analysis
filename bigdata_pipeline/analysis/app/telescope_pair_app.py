"""Optional standalone pair explorer; can also be integrated in the main TeleScope app."""
import argparse
from pathlib import Path
import streamlit as st
from telescope_pair_explorer import render_pair_explorer

st.set_page_config(page_title='TeleScope — pair explorer',layout='wide')
ap=argparse.ArgumentParser()
ap.add_argument('--analysis',default='bigdata_pipeline/analysis')
args,_=ap.parse_known_args()
st.title('TeleScope — Pair Explorer')
lang=st.sidebar.selectbox('Language / Язык / Langue',['ru','en','fr'])
root=st.sidebar.text_input('Analysis folder',args.analysis)
render_pair_explorer(Path(root).expanduser(),lang)
