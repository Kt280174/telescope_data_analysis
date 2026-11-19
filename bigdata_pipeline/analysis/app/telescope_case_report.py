"""Case dossiers from saved results only; no chains, permutations, or raw-data access."""
from pathlib import Path
import argparse
import json
import pandas as pd

KEYS = ['channel_a', 'channel_b']
REVIEW_FIELDS = ['reviewer', 'review_date', 'sync_evidence_label', 'explanation_label',
                 'suspicion_label', 'evidence_reference', 'notes']
LABELS = {'supported', 'not_supported', 'insufficient_data'}

def table_path(folder, name):
    folder = Path(folder)
    for path in (folder/name/(name+'.csv'), folder/(name+'.csv')):
        if path.is_file():
            return path
    return None

def folders(analysis, metadata=None):
    root = Path(analysis)
    deep = root/'suspicious_deep' if (root/'suspicious_deep').is_dir() else root
    meta = Path(metadata) if metadata else deep.parent/'suspicious_metadata'
    if not metadata and any(table_path(deep, name) for name in ['metadata_channel_profiles', 'metadata_pair_profiles']):
        meta = deep
    return deep, meta

def saved_paths(analysis, metadata=None):
    deep, meta = folders(analysis, metadata)
    return {name: table_path(folder, name) for name, folder in [
        ('pair_profiles', deep), ('suspicious_examples', deep),
        ('metadata_pair_profiles', meta), ('metadata_channel_profiles', meta)]}

def pair_frame(frame, name):
    if not set(KEYS) <= set(frame):
        raise ValueError(name+': missing pair IDs')
    frame = frame.copy()
    for key in KEYS:
        values = pd.to_numeric(frame[key], errors='raise')
        if values.isna().any() or not (values % 1 == 0).all():
            raise ValueError(name+': invalid pair ID')
        frame[key] = values.astype('int64')
    if frame.duplicated(KEYS).any():
        raise ValueError(name+': duplicate pair IDs')
    return frame

def assemble(paths):
    if paths['pair_profiles'] is None:
        raise FileNotFoundError('Missing saved pair_profiles.csv; run this on the existing deep output folder.')
    p = pair_frame(pd.read_csv(paths['pair_profiles']), 'pair_profiles')
    required = ['n_sync_posts', 'n_shared_posts', 'n_sync_days', 'sync_share', 'overlap_share', 'median_gap_s', 'p90_gap_s']
    if not set(required) <= set(p):
        raise ValueError('pair_profiles: missing deep-profile metrics')
    for col in required:
        p[col] = pd.to_numeric(p[col], errors='raise')
        if (p[col].dropna() < 0).any(): raise ValueError('Negative metric: '+col)
    for col in ['sync_share', 'overlap_share']:
        if (p[col].dropna() > 1).any(): raise ValueError('Invalid proportion: '+col)
    if (p.n_sync_posts > p.n_shared_posts).any(): raise ValueError('Sync count exceeds shared count')
    diagnostics = {'missing_optional_tables': [k for k,v in paths.items() if v is None and k != 'pair_profiles']}
    if paths['metadata_pair_profiles']:
        m = pair_frame(pd.read_csv(paths['metadata_pair_profiles']), 'metadata_pair_profiles')
        check = p[KEYS+required[:3]].merge(m, on=KEYS, how='outer', validate='one_to_one', indicator=True, suffixes=('_deep',''))
        if not check['_merge'].eq('both').all():
            raise ValueError('Deep and metadata pair lists differ: use results from the same run.')
        for col in required[:3]:
            if col not in m or not check[col].eq(check[col+'_deep']).all():
                raise ValueError('Deep/metadata count mismatch: '+col)
        p = p.merge(m[KEYS+[col for col in m if col not in p]], on=KEYS, validate='one_to_one')
    if paths['metadata_channel_profiles']:
        c = pd.read_csv(paths['metadata_channel_profiles'])
        if 'channel_id' not in c or c.channel_id.isna().any() or c.channel_id.duplicated().any():
            raise ValueError('metadata_channel_profiles: invalid or duplicate channel IDs')
        c.channel_id = pd.to_numeric(c.channel_id,errors='raise').astype('int64')
        if c.channel_id.duplicated().any(): raise ValueError('Duplicate numeric channel IDs')
        lookup = c.set_index('channel_id')
        for side in ['a', 'b']:
            p['has_channel_metadata_'+side] = p['channel_'+side].isin(lookup.index)
            for col in ['title','username','scam','fake','restricted','verified']:
                if col in lookup: p[col+'_'+side] = p['channel_'+side].map(lookup[col])
    else:
        for side in ['a', 'b']: p['has_channel_metadata_'+side] = False
    for side in ['a', 'b']:
        for col in ['title','username']:
            if col+'_'+side not in p: p[col+'_'+side] = ''
    titles = p.title_a.fillna('')+' '+p.title_b.fillna('')
    p['possible_backup_title'] = titles.str.contains(r'резерв|backup|reserve|dự phòng',case=False,regex=True)
    p['possible_discussion_title'] = titles.str.contains(r'обсуждени|\bчат\b|\bchat\b|discussion',case=False,regex=True)
    # Descriptive flags, not calibrated thresholds or abuse labels.
    p['persistent_high_sync'] = (p.sync_share >= .95) & (p.n_sync_days >= 30)
    p['low_overlap'] = p.overlap_share < .1
    e = pd.DataFrame()
    if paths['suspicious_examples']:
        e = pd.read_csv(paths['suspicious_examples'])
        if not set(KEYS+['date_a','date_b','gap_s','origin_channel','origin_post']) <= set(e):
            raise ValueError('Invalid saved example schema')
        gap = (pd.to_datetime(e.date_a,utc=True,errors='raise')-pd.to_datetime(e.date_b,utc=True,errors='raise')).dt.total_seconds().abs()
        if gap.isna().any() or not gap.eq(pd.to_numeric(e.gap_s,errors='raise')).all():
            raise ValueError('Example timestamps disagree with gap_s')
        counts = e.groupby(KEYS).size().rename('available_example_rows').reset_index()
        p = p.merge(counts,on=KEYS,how='left',validate='one_to_one')
    if 'available_example_rows' not in p: p['available_example_rows'] = 0
    p['available_example_rows'] = p.available_example_rows.fillna(0).astype(int)
    return p, e, diagnostics

def review_summary(review, dossiers):
    review = pair_frame(review, 'review')
    if not set(KEYS+REVIEW_FIELDS) <= set(review): raise ValueError('Review file is missing template columns')
    if not review.merge(dossiers[KEYS],on=KEYS,how='left',indicator=True)['_merge'].eq('both').all():
        raise ValueError('Review contains unknown candidate pairs')
    s = review.suspicion_label.fillna('').str.strip()
    if not s.isin(LABELS|{''}).all(): raise ValueError('Invalid suspicion_label')
    # Decisions require an identifiable reviewer, date, and independent evidence reference.
    decided = s.isin({'supported','not_supported'})
    for col in ['reviewer','review_date','evidence_reference']:
        if (decided & review[col].fillna('').str.strip().eq('')).any():
            raise ValueError('Decided reviews require '+col)
    tp=int(s.eq('supported').sum());fp=int(s.eq('not_supported').sum())
    return {'reviewed_file_pairs':len(review),'supported':tp,'not_supported':fp,
            'unresolved_or_blank':len(review)-tp-fp,
            'false_alarm_fraction_among_decided':fp/(tp+fp) if tp+fp else None,
            'classical_fpr':None,'scope':'Only pairs in the supplied review file; sampling and adjudication must be documented.'}

def export_saved(analysis, metadata=None, out=None, review=None, sample_size=100, seed=42):
    deep,_ = folders(analysis,metadata)
    p,e,diagnostics = assemble(saved_paths(analysis,metadata))
    output = Path(out) if out else deep/'case_report'
    if output.resolve() in {deep.resolve(), Path(analysis).resolve()}:
        raise ValueError('Use a separate case_report output directory')
    if sample_size < 1: raise ValueError('sample-size must be positive')
    review_info = review_summary(pd.read_csv(review),p) if review else None
    output.mkdir(parents=True,exist_ok=True)
    p.to_csv(output/'case_dossiers.csv',index=False,encoding='utf-8-sig')
    template=output/'manual_review_template.csv'
    # Never overwrite labels entered into a previous template.
    created = not template.exists()
    if created:
        sample=p.sort_values(KEYS).sample(n=min(sample_size,len(p)),random_state=seed).sort_values(KEYS).copy()
        for col in REVIEW_FIELDS: sample[col]=''
        sample.to_csv(template,index=False,encoding='utf-8-sig')
    summary={'candidate_pairs':len(p),'unique_endpoints':len(set(p.channel_a)|set(p.channel_b)),
        'median_sync_share':None if p.empty else float(p.sync_share.median()),
        'pairs_with_saved_examples':int(p.available_example_rows.gt(0).sum()),
        'title_hint_backup':int(p.possible_backup_title.sum()),
        'title_hint_discussion':int(p.possible_discussion_title.sum()),
        'diagnostics':diagnostics,'review':review_info,
        'template_created_this_run':created,'new_template_seed':seed if created else None,
        'new_template_requested_size':sample_size if created else None,
        'note':'Descriptive title/metric flags are unverified leads, not abuse labels. Existing review templates are preserved.'}
    (output/'case_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    top=p.sort_values(['n_sync_posts','channel_a','channel_b'],ascending=[False,True,True]).head(10)
    lines=['# 5.5. Подозрительные тенденции — сохранённые результаты','',
        f'Пар-кандидатов: {len(p):,}. Это предварительный отбор по синхронности, а не подтверждённые боты или спам.',
        '','sync_share = синхронные / общие исходные сообщения. overlap_share = общие / min(posts_a, posts_b).',
        '','## Примеры: первые десять пар по числу синхронных сообщений','']
    for r in top.itertuples():
        lines.append(f'- {r.channel_a} ({r.title_a}) / {r.channel_b} ({r.title_b}): n={r.n_sync_posts}, дней={r.n_sync_days}, sync_share={r.sync_share:.2%}, overlap_share={r.overlap_share:.2%}, медиана задержки={r.median_gap_s:g} с; сохранённых примеров={r.available_example_rows}.')
    lines += ['','## Проверка интерпретации','',
        'Названия с указанием резерва или обсуждения — только подсказки. Проверяйте отношения каналов, например linked_chat_id для группы обсуждения. Названия из текущего снимка могут не отражать историческое состояние.',
        'Порог persistent_high_sync (sync_share ≥ 0.95 и ≥ 30 дней) описательный; он не валидирован для классификации злоупотреблений.',
        'Отсутствие примеров не означает отсутствия синхронности. Агрегированные временные профили и платформенные флаги не заменяют независимую проверку.',
        '','## Ложные тревоги','']
    if review_info and review_info['false_alarm_fraction_among_decided'] is not None:
        lines.append('Доля not_supported среди определённых меток в предоставленном файле: '+f"{review_info['false_alarm_fraction_among_decided']:.2%}. Неопределённых/пустых: {review_info['unresolved_or_blank']}. Это не классическая FPR и не автоматически оценка для всей сети.")
    else: lines.append('Доля ложных тревог не оценена: независимые определённые метки не предоставлены.')
    lines += ['','Шаблон: manual_review_template.csv. suspicion_label: supported / not_supported / insufficient_data. Требуются reviewer, review_date, evidence_reference. Перед разметкой согласуйте критерии злоупотребления; сама синхронность недостаточна. Пары могут иметь общие каналы, поэтому наблюдения зависимы.']
    (output/'section_5_5_saved.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return output, summary

def words(lang,ru,en,fr): return {'ru':ru,'en':en,'fr':fr}.get(lang,ru)

def render_case(p,e,a,b,lang='ru',key='case'):
    import streamlit as st
    selected=p.loc[p.channel_a.eq(a)&p.channel_b.eq(b)]
    if selected.empty: return
    r=selected.iloc[0]
    w=lambda ru,en,fr:words(lang,ru,en,fr)
    st.subheader(w('5.5 — Разбор выбранной пары','5.5 — Selected-pair evidence','5.5 — Éléments de la paire'))
    for side in ['a','b']:
        title=r.get('title_'+side,'');user=r.get('username_'+side,'')
        text=f"{int(r['channel_'+side])}: {title if pd.notna(title) else '—'}"
        if pd.notna(user) and str(user).strip():text+=' (@'+str(user)+')'
        st.text(text)
    cols=st.columns(4)
    for col,label,value in zip(cols,[w('Доля синхронных','Sync share','Part synchrone'),w('Перекрытие','Overlap','Recouvrement'),w('Дни синхронности','Sync days','Jours synchrones'),w('Медиана / p90','Median / p90','Médiane / p90')],
        [f'{r.sync_share:.2%}',f'{r.overlap_share:.2%}',str(int(r.n_sync_days)),f'{r.median_gap_s:g} / {r.p90_gap_s:g} s']): col.metric(label,value)
    st.caption(w('Доли: синхронные / общие; общие / min(posts_a, posts_b). Задержки рассчитаны только для синхронных событий.',
        'Shares: synchronized/shared; shared/min(posts_a, posts_b). Gap statistics cover synchronized events only.',
        'Parts : synchrones/communs ; communs/min(posts_a, posts_b). Délais des événements synchrones uniquement.'))
    if r.possible_backup_title:st.info(w('Название указывает на возможный резервный канал. Связь требует проверки.','Title suggests a possible backup channel; verify the relationship.','Le titre suggère un canal de secours ; relation à vérifier.'))
    if r.possible_discussion_title:st.info(w('Название указывает на возможное обсуждение. Проверьте linked_chat_id: автоматические пересылки могут объяснять синхронность.','Title suggests a discussion group. Check linked_chat_id: automatic forwarding may explain synchrony.','Le titre suggère une discussion. Vérifiez linked_chat_id : les transferts automatiques peuvent expliquer la synchronisation.'))
    if r.persistent_high_sync:st.info(w('Длительная высокая синхронность (≥95%, ≥30 дней). Описательный признак, не доказательство злоупотребления.','Persistent high synchrony (≥95%, ≥30 days). Descriptive flag, not proof of abuse.','Synchronisation persistante élevée (≥95 %, ≥30 jours). Indice descriptif, pas preuve d’abus.'))
    if r.low_overlap:st.info(w('Перекрытие ниже 10%: учитывайте объём активности, а не только число совпадений.','Overlap below 10%: consider activity volume alongside raw counts.','Recouvrement inférieur à 10 % : tenir compte du volume d’activité.'))
    st.warning(w('Алгоритмические результаты сами по себе не подтверждают злоупотребление. Ручные метки показываются отдельно в разделе проверки.','Algorithmic results alone do not confirm abuse. Manual labels are shown separately in the review section.','Les résultats algorithmiques seuls ne confirment pas un abus. Les annotations manuelles figurent dans la section de validation.'))
    if not bool(r.has_channel_metadata_a) or not bool(r.has_channel_metadata_b):
        st.caption(w('Метаданные одного или обоих участников отсутствуют.','Metadata missing for one or both endpoints.','Métadonnées absentes pour au moins un membre.'))
    if set(KEYS)<=set(e):
        examples=e.loc[e.channel_a.eq(a)&e.channel_b.eq(b)]
    else: examples=pd.DataFrame()
    if examples.empty:st.caption(w('Сохранённых примеров нет; это не отрицательный результат.','No saved examples; this is not a negative finding.','Aucun exemple sauvegardé ; ce n’est pas un résultat négatif.'))
    else:
        st.caption(w('Выборка сохранённых событий, не полная история и не случайная валидационная выборка.','Saved event sample, not full history or a random validation sample.','Exemples sauvegardés, pas un historique complet ni un échantillon de validation aléatoire.'))
        st.dataframe(examples,width='stretch',hide_index=True)
    st.download_button(w('Скачать досье CSV','Download dossier CSV','Télécharger le dossier CSV'),selected.to_csv(index=False).encode('utf-8-sig'),file_name=f'case_{a}_{b}.csv',mime='text/csv',key=key+'_dossier')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analysis',type=Path,default=Path('bigdata_pipeline/analysis'))
    ap.add_argument('--metadata',type=Path)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--review',type=Path)
    ap.add_argument('--sample-size',type=int,default=100)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()
    out,summary=export_saved(a.analysis,a.metadata,a.out,a.review,a.sample_size,a.seed)
    print(json.dumps(summary,ensure_ascii=False,indent=2));print('Saved-results report:',out)

if __name__=='__main__':main()
