"""Session-only manual review editor; exports explicit human labels, never infers them."""
from datetime import datetime, date
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st
from telescope_case_report import REVIEW_FIELDS, review_summary, words

def text_value(row, key):
    value=row.get(key,'')
    return '' if pd.isna(value) else str(value)

def save_decision(reviews, cases, a, b, fields):
    """Validate the complete proposed result before changing the session."""
    selected=cases.loc[cases.channel_a.eq(a)&cases.channel_b.eq(b)]
    if len(selected)!=1: raise ValueError('Select one known candidate pair')
    row=selected.copy()
    for col in REVIEW_FIELDS:row[col]=fields.get(col,'')
    old=reviews.loc[reviews.channel_a.eq(a)&reviews.channel_b.eq(b)]
    row['review_selection']=text_value(old.iloc[0],'review_selection') if len(old) else 'selected_pair'
    remaining=reviews.loc[~(reviews.channel_a.eq(a)&reviews.channel_b.eq(b))]
    result=pd.concat([remaining,row],ignore_index=True)
    review_summary(result,cases)
    return result

def render_manual_review(cases,deep,lang,prefix,a,b):
    from telescope_pair_explorer import read_csv
    w=lambda ru,en,fr:words(lang,ru,en,fr)
    with st.expander(w('Ручная проверка и ложные тревоги','Manual review and false alarms','Validation manuelle et fausses alertes'),expanded=True):
        st.caption(w('Сначала согласуйте критерий: подозрение на злоупотребление требует дополнительных доказательств, а не только синхронности. Подтверждённые резервные каналы или обсуждения могут иметь обычное объяснение.',
            'Agree on the criterion first: suspected abuse requires evidence beyond synchrony. Verified backup/discussion relationships may provide an ordinary explanation.',
            'Définir le critère : un soupçon d’abus nécessite des preuves au-delà de la synchronisation. Une relation de secours/discussion vérifiée peut fournir une explication ordinaire.'))
        state=prefix+'_human_reviews'
        template=deep/'case_report/manual_review_template.csv'
        if state not in st.session_state:
            if template.is_file():
                sample=read_csv(template).copy()
                if 'review_selection' not in sample:sample['review_selection']='existing_template'
            else:
                sample=cases.sort_values(['channel_a','channel_b']).sample(n=min(100,len(cases)),random_state=42).sort_values(['channel_a','channel_b']).copy()
                for col in REVIEW_FIELDS:sample[col]=''
                sample['review_selection']='random_seed42'
            try:review_summary(sample,cases)
            except (ValueError,KeyError) as error:
                st.error(str(error));return
            st.session_state[state]=sample
        uploaded=st.file_uploader(w('Продолжить проверку из CSV','Continue review from CSV','Reprendre depuis un CSV'),type=['csv'],key=prefix+'_human_upload')
        if uploaded is not None and st.button(w('Заменить рабочие метки загруженным файлом','Replace session labels with uploaded file','Remplacer les annotations par le fichier importé'),key=prefix+'_human_import'):
            try:
                uploaded.seek(0);incoming=pd.read_csv(uploaded)
                review_summary(incoming,cases)
                if 'review_selection' not in incoming:incoming['review_selection']='uploaded_unknown_sampling'
                st.session_state[state]=incoming
                # Clear cached input values so imported labels appear in the current form.
                for k in list(st.session_state):
                    if k.startswith(prefix+'_decision_'):del st.session_state[k]
                st.success(w('Метки импортированы.','Labels imported.','Annotations importées.'))
            except (ValueError,KeyError,pd.errors.ParserError) as error:st.error(str(error))
        reviews=st.session_state[state]
        current=reviews.loc[reviews.channel_a.eq(a)&reviews.channel_b.eq(b)]
        row=current.iloc[0] if len(current) else pd.Series(dtype=object)
        st.markdown(w(f'**Проверка пары {a} / {b}**',f'**Review pair {a} / {b}**',f'**Validation de la paire {a} / {b}**'))
        key=f'{prefix}_decision_{a}_{b}_'
        statuses=['','supported','not_supported','insufficient_data']
        names={'':w('Не проверено','Not reviewed','Non évalué'),
            'supported':w('Есть основания подозревать злоупотребление','Evidence supports suspected abuse','Éléments en faveur d’un abus suspecté'),
            'not_supported':w('Обычное объяснение подтверждено; подозрение не поддержано','Ordinary explanation verified; suspicion not supported','Explication ordinaire vérifiée ; soupçon non étayé'),
            'insufficient_data':w('Недостаточно данных','Insufficient evidence','Éléments insuffisants')}
        sync_options=['','supported','contradicted','insufficient_data']
        explanations=['','confirmed_primary_backup','confirmed_discussion_link','other_documented_relation','unexplained','insufficient_data']
        today=datetime.now(ZoneInfo('Europe/Moscow')).date()
        try:review_date=date.fromisoformat(text_value(row,'review_date'))
        except ValueError:review_date=today
        with st.form(key+'form'):
            reviewer=st.text_input(w('Проверяющий','Reviewer','Évaluateur'),value=text_value(row,'reviewer'),key=key+'reviewer')
            day=st.date_input(w('Дата проверки','Review date','Date de validation'),value=review_date,key=key+'date')
            saved_sync=text_value(row,'sync_evidence_label')
            sync=st.selectbox(w('Подтверждаются ли синхронные события?','Are synchronized events supported?','Les événements synchrones sont-ils étayés ?'),sync_options,index=sync_options.index(saved_sync) if saved_sync in sync_options else 0,key=key+'sync')
            saved_explanation=text_value(row,'explanation_label')
            explanation=st.selectbox(w('Проверенное объяснение','Verified explanation','Explication vérifiée'),explanations,index=explanations.index(saved_explanation) if saved_explanation in explanations else 0,key=key+'explanation')
            saved_status=text_value(row,'suspicion_label')
            status=st.selectbox(w('Оценка подозрения на злоупотребление','Assessment of suspected abuse','Évaluation du soupçon d’abus'),statuses,index=statuses.index(saved_status) if saved_status in statuses else 0,format_func=lambda v:names[v],key=key+'status')
            evidence=st.text_area(w('Источник доказательств (ссылка или файл/строки)','Evidence reference (URL or file/rows)','Source de preuve (URL ou fichier/lignes)'),value=text_value(row,'evidence_reference'),key=key+'evidence')
            notes=st.text_area(w('Обоснование и ограничения','Reasoning and limitations','Justification et limites'),value=text_value(row,'notes'),key=key+'notes')
            submit=st.form_submit_button(w('Сохранить оценку в сессии','Save assessment in session','Enregistrer dans la session'))
        if submit:
            fields=dict(reviewer=reviewer.strip(),review_date=day.isoformat(),sync_evidence_label=sync,
                explanation_label=explanation,suspicion_label=status,evidence_reference=evidence.strip(),notes=notes.strip())
            try:
                if status in {'supported','not_supported'} and not notes.strip():raise ValueError(w('Добавьте обоснование.','Add a reason for the decision.','Ajoutez une justification.'))
                st.session_state[state]=save_decision(reviews,cases,a,b,fields)
                st.success(w('Оценка сохранена в сессии. Скачайте CSV для постоянного сохранения.','Assessment saved in session. Download CSV to retain it.','Évaluation enregistrée dans la session. Téléchargez le CSV pour la conserver.'))
            except (ValueError,KeyError) as error:st.error(str(error))
        reviews=st.session_state[state]
        result=review_summary(reviews,cases)
        cols=st.columns(4)
        rate=result['false_alarm_fraction_among_decided']
        for col,label,value in zip(cols,[w('Подозрение поддержано','Suspicion supported','Soupçon étayé'),w('Подозрение не поддержано','Suspicion not supported','Soupçon non étayé'),w('Не определено / не проверено','Unresolved / not reviewed','Indéterminé / non évalué'),w('Доля ложных тревог*','False-alarm fraction*','Part de fausses alertes*')],
            [result['supported'],result['not_supported'],result['unresolved_or_blank'],f'{rate:.1%}' if rate is not None else '—']):col.metric(label,value)
        if rate is None:st.info(w('Нет определённых ручных меток: доля ложных тревог не оценена.','No decided human labels: false-alarm fraction not assessed.','Aucune annotation décidée : part de fausses alertes non évaluée.'))
        st.caption(w('* not_supported / (supported + not_supported), только для определённых меток этой рабочей таблицы. Это не классическая FPR и не оценка всей сети. Добавление выбранных вручную пар меняет состав выборки; неопределённые случаи учитываются отдельно.',
            '* not_supported / (supported + not_supported), for decided labels in this working table only. Not classical FPR or a network-wide estimate. Adding hand-picked pairs changes the sample; unresolved cases are separate.',
            '* not_supported / (supported + not_supported), pour les annotations décidées de cette table uniquement. Ni FPR classique ni estimation du réseau. Ajouter des paires choisies modifie l’échantillon ; cas indéterminés séparés.'))
        shown=reviews.loc[reviews.suspicion_label.fillna('').ne(''),['channel_a','channel_b']+REVIEW_FIELDS]
        if not shown.empty:st.dataframe(shown,width='stretch',hide_index=True)
        st.download_button(w('Скачать все рабочие метки CSV','Download all working labels CSV','Télécharger toutes les annotations CSV'),reviews.to_csv(index=False).encode('utf-8-sig'),file_name='reviewed_cases.csv',mime='text/csv',key=prefix+'_human_download')
        st.caption(w('Метки хранятся только в текущей сессии. Перед закрытием страницы скачайте CSV; затем импортируйте его, чтобы продолжить. Deep не пересчитывается.',
            'Labels live in this session only. Download CSV before closing; import it to resume. Deep is not recomputed.',
            'Annotations limitées à la session. Téléchargez le CSV avant de fermer, puis importez-le pour reprendre. Aucun recalcul deep.'))
