"""Add the pair explorer to the existing main app with a backup and idempotent patch.
Run from repo root: python install_pair_explorer.py
"""
import argparse
import ast
from datetime import datetime
from pathlib import Path

IMPORT='from telescope_pair_explorer import render_pair_explorer'
ANCHOR='    st.write(tr(f"{page_id}_intro"))'
BLOCK='\n    # TELEScope_PAIR_EXPLORER\n    if page_id == "theme5":\n        render_pair_explorer(analysis, LANG)'


def patch(path):
    source=path.read_text(encoding='utf-8-sig')
    if '# TELEScope_PAIR_EXPLORER' in source:
        print('Pair explorer is already installed.');return None
    if source.count(ANCHOR)!=1:
        raise ValueError('Main app differs from the expected version; no file changed. Use telescope_pair_app.py or add render_pair_explorer(analysis, LANG) inside theme5 manually.')
    if '"theme5"' not in source and "'theme5'" not in source:raise ValueError('Main app has no theme5; use the standalone pair app.')
    updated=source.replace(ANCHOR,ANCHOR+BLOCK,1)
    if IMPORT not in updated:
        import_anchor='from telescope_captions import describe, title_of'
        if import_anchor not in updated:raise ValueError('Import anchor not found; use standalone pair app. No file changed.')
        updated=updated.replace(import_anchor,import_anchor+'\n'+IMPORT,1)
    ast.parse(updated)
    backup=path.with_name(path.name+'.'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.bak')
    backup.write_bytes(path.read_bytes())
    path.write_text(updated,encoding='utf-8')
    print('Installed pair explorer. Backup:',backup)
    return backup

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--app',type=Path,default=Path('bigdata_pipeline/analysis/app/telescope_app.py'))
    args=ap.parse_args()
    patch(args.app)
