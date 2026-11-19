"""Interactive pair selector over saved CSVs. No chains loading or permutation reruns."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import streamlit as st

TEXT={
 'title':('Просмотр выбранной пары','Selected pair explorer','Explorer une paire'),
 'only_null':('Только пары с перестановками','Only pairs with saved permutations','Paires avec permutations uniquement'),
 'missing_permutation':('Для этой пары перестановки не сохранены. Можно посмотреть рассчитанные характеристики и подробности в других вкладках.','No saved permutations for this pair. Available profiles and details can still be viewed in other tabs.','Pas de permutations sauvegardées ; consultez les profils et détails disponibles dans les autres onglets.'),
 'exceedances':('Перестановок не ниже наблюдения','Permutations at least as high as observed','Permutations au moins égales à l’observation'),
 'select':('Выберите пару каналов (можно искать по ID)','Choose a channel pair (search by ID)','Choisissez une paire (recherche par ID)'),
 'previous':('← Предыдущая','← Previous','← Précédente'),
 'next':('Следующая →','Next →','Suivante →'),
 'observed':('Наблюдение','Observed','Observé'),
 'mean':('Среднее перестановок','Permutation mean','Moyenne des permutations'),
 'ratio':('Наблюдение / среднее','Observed / mean','Observé / moyenne'),
 'trials':('Перестановок','Permutations','Permutations'),
 'distribution':('Распределение перестановок — приближение','Permutation distribution — zoomed','Distribution des permutations — zoom'),
 'comparison':('Сравнение на общей шкале','Comparison on a shared scale','Comparaison sur une échelle commune'),
 'posts':('Синхронные исходные публикации','Synchronized original posts','Publications originales synchronisées'),
 'count':('Число перестановок','Permutation count','Nombre de permutations'),
 'p95':('95-й процентиль','95th percentile','95e percentile'),
 'permutation':('Перестановки','Permutations','Permutations'),
 'time':('Активность и примеры','Activity and examples','Activité et exemples'),
 'sources':('Общие источники','Shared sources','Sources communes'),
 'data':('Данные выбранной пары','Selected-pair data','Données de la paire'),
 'weekly':('Синхронные публикации по неделям','Synchronized posts per week','Publications synchronisées par semaine'),
 'week':('Неделя UTC','UTC week','Semaine UTC'),
 'relative':('Секунды относительно пересылки канала A','Seconds relative to relay at channel A','Secondes par rapport au relais du canal A'),
 'raster':('Сохранённые примеры: относительная временная шкала','Saved examples: relative-time raster','Exemples sauvegardés : temps relatif'),
 'source_share':('Доля синхронных публикаций','Share of synchronized posts','Part des publications synchronisées'),
 'source_id':('ID канала-источника','Source channel ID','ID du canal source'),
 'download':('Скачать график HTML','Download chart HTML','Télécharger le graphe HTML'),
 'download_csv':('Скачать перестановки CSV','Download permutations CSV','Télécharger les permutations CSV'),
 'no_data':('Для этой пары подробная таблица не рассчитана или отсутствует. Это не означает отсутствие активности.','No saved detail table for this pair. This does not mean no activity.','Pas de détail sauvegardé pour cette paire ; cela ne signifie pas aucune activité.'),
 'scope':('Снимите фильтр, чтобы выбрать другие пары из сохранённого профиля. Показываются только уже рассчитанные результаты, без перерасчёта.','Uncheck the filter to select other saved-profile pairs. Only existing results are shown; no recomputation.','Décochez le filtre pour voir les autres paires du profil sauvegardé. Aucun recalcul.'),
 'warning':('Исследовательская проверка ранее отобранной пары. Синхронность не доказывает ботов, спам или злонамеренную координацию.','Exploratory check of a previously selected pair. Synchrony does not prove bots, spam or abusive coordination.','Vérification exploratoire après sélection ; la synchronisation ne prouve ni bots ni abus.'),
 'examples_note':('Это сохранённая выборка примеров, не полный временной ряд пары.','These are saved sampled examples, not the full pair history.','Exemples échantillonnés sauvegardés, pas un historique complet.'),
 'missing':('Нет двух таблиц перестановок. Ожидается suspicious_deep/permutation_summary и permutation_distribution внутри папки analysis.','Missing permutation tables. Expected suspicious_deep/permutation_summary and permutation_distribution under analysis.','Tables de permutations absentes sous suspicious_deep dans analysis.'),
}

def tr(key,lang='ru'):
    return TEXT[key][{'ru':0,'en':1,'fr':2}.get(lang,0)]

def integer(v):return f'{int(v):,}'.replace(',',' ')

def number(v):return f'{float(v):,.1f}'.replace(',',' ')

@st.cache_data(show_spinner=False)
def _csv(path,mtime_ns,size):
    return pd.read_csv(path)

def read_csv(path):
    stat=path.stat();return _csv(str(path),stat.st_mtime_ns,stat.st_size)

def locate_deep(root):
    root=Path(root)
    if (root/'suspicious_deep').is_dir():return root/'suspicious_deep'
    return root

def optional_table(deep,name):
    from telescope_case_report import table_path
    path=table_path(deep,name)
    return read_csv(path) if path else pd.DataFrame()

def subset(frame,a,b):
    if not {'channel_a','channel_b'}<=set(frame):return pd.DataFrame()
    return frame.loc[frame.channel_a.eq(a)&frame.channel_b.eq(b)].copy()

def pair_stats(summary,distribution,a,b):
    if not {'channel_a','channel_b','observed_sync','n_permutations'}<=set(summary):
        raise ValueError('Summary schema is missing pair/observed/permutation columns')
    if not {'channel_a','channel_b','trial','null_sync_posts'}<=set(distribution):
        raise ValueError('Distribution schema is missing pair/trial/count columns')
    selected=subset(summary,a,b)
    if len(selected)!=1:raise ValueError('Selected pair must have exactly one summary row')
    row=selected.iloc[0];d=subset(distribution,a,b)
    if len(d)!=int(row.n_permutations) or d.trial.duplicated().any() or d.empty:
        raise ValueError('Permutation count or trial IDs do not match the selected summary row')
    values=pd.to_numeric(d.null_sync_posts,errors='raise').to_numpy(dtype=float)
    observed=float(row.observed_sync)
    if not np.isfinite(values).all() or (values<0).any() or not np.isfinite(observed) or observed<0:
        raise ValueError('Counts must be finite and non-negative')
    mean=float(values.mean());q95=float(np.quantile(values,.95))
    for col,value in [('null_mean',mean),('null_p95',q95)]:
        if col in row and not np.isclose(float(row[col]),value):raise ValueError(f'{col} differs between saved tables')
    return {'a':int(a),'b':int(b),'observed':observed,'mean':mean,'p95':q95,
            'ratio':observed/mean if mean else np.nan,'n':len(values),
            'exceedances':int((values>=observed).sum())},values,d


def permutation_figure(stats,values,lang='ru'):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig=make_subplots(rows=1,cols=2,horizontal_spacing=.18,subplot_titles=[tr('distribution',lang),tr('comparison',lang)])
    bins=np.array([values.min()-.5,values.max()+.5]) if values.min()==values.max() else np.histogram_bin_edges(values,bins='fd')
    counts,bins=np.histogram(values,bins=bins)
    centres=(bins[1:]+bins[:-1])/2
    fig.add_trace(go.Bar(x=centres,y=counts,width=np.diff(bins),marker_color='#386CB0',
                        name=tr('permutation',lang),hovertemplate='%{y} — bin centre %{x:.1f}<extra></extra>'),row=1,col=1)
    fig.add_vline(x=stats['mean'],line_color='#183E68',row=1,col=1)
    fig.add_vline(x=stats['p95'],line_color='#697785',line_dash='dash',row=1,col=1)
    fig.add_trace(go.Bar(x=[stats['mean'],stats['observed']],y=[tr('mean',lang),tr('observed',lang)],
                        orientation='h',marker_color=['#386CB0','#D65642'],
                        text=[number(stats['mean']),integer(stats['observed'])],textposition='outside',cliponaxis=False,
                        hovertemplate='%{y}: %{x:,.1f}<extra></extra>'),row=1,col=2)
    fig.update_xaxes(title_text=tr('posts',lang),row=1,col=1)
    fig.update_yaxes(title_text=tr('count',lang),row=1,col=1)
    fig.update_xaxes(title_text=tr('posts',lang),range=[0,max(stats['observed'],stats['mean'],1)*1.28],row=1,col=2)
    fig.update_layout(height=480,template='plotly_white',showlegend=False,margin=dict(l=30,r=55,t=65,b=50))
    return fig


def weekly_figure(frame,lang):
    import plotly.graph_objects as go
    dates=pd.to_datetime(frame.week,errors='coerce')
    if dates.isna().any():raise ValueError('Invalid saved weekly dates')
    s=pd.Series(pd.to_numeric(frame.n_sync_posts).to_numpy(),index=dates).groupby(level=0).sum().sort_index()
    s=s.reindex(pd.date_range(s.index.min(),s.index.max(),freq='7D'),fill_value=0)
    fig=go.Figure(go.Scatter(x=s.index,y=s.values,mode='lines+markers',line=dict(color='#386CB0')))
    fig.update_layout(title=tr('weekly',lang),xaxis_title=tr('week',lang),yaxis_title=tr('posts',lang),template='plotly_white',height=350)
    return fig


def example_figure(frame,a,b,lang):
    import plotly.graph_objects as go
    e=frame.copy();e['date_a']=pd.to_datetime(e.date_a,utc=True,errors='coerce');e['date_b']=pd.to_datetime(e.date_b,utc=True,errors='coerce')
    e=e.dropna(subset=['date_a','date_b']).sort_values('date_a')
    gap=(e.date_b-e.date_a).dt.total_seconds().to_numpy();ys=np.arange(len(e))
    xx=[];yy=[]
    for i,g in enumerate(gap):xx.extend([0,g,None]);yy.extend([i,i,None])
    fig=go.Figure(go.Scatter(x=xx,y=yy,mode='lines',line=dict(color='#BDC7D1',width=1),hoverinfo='skip',showlegend=False))
    detail=np.array([[str(r.date_a),str(r.date_b),f'{int(r.origin_channel)}/{int(r.origin_post)}'] for r in e.itertuples(index=False)])
    for name,x in [(str(a),np.zeros(len(e))),(str(b),gap)]:
        fig.add_trace(go.Scatter(x=x,y=ys,mode='markers',name=name,customdata=detail,
                                hovertemplate='%{customdata[2]}<br>A: %{customdata[0]}<br>B: %{customdata[1]}<extra>%{fullData.name}</extra>'))
    fig.update_layout(title=tr('raster',lang),xaxis_title=tr('relative',lang),template='plotly_white',height=max(350,min(900,20*len(e))))
    fig.update_yaxes(tickmode='array',tickvals=ys,ticktext=[f'{r.date_a:%Y-%m-%d} | {int(r.origin_channel)}/{int(r.origin_post)}' for r in e.itertuples(index=False)])
    return fig


def move_pair(key,options,offset):
    current=st.session_state.get(key,options[0]);idx=options.index(current)
    st.session_state[key]=options[(idx+offset)%len(options)]


def render_pair_explorer(analysis,lang='ru'):
    deep=locate_deep(analysis)
    st.subheader(tr('title',lang))
    all_profiles=optional_table(deep,'pair_profiles')
    if all_profiles.empty:
        fallback=deep.parent/'suspicious/suspicious_pairs.csv'
        if fallback.is_file():all_profiles=read_csv(fallback)
    try:
        import plotly.graph_objects as go
        summary=optional_table(deep,'permutation_summary')
        distribution=optional_table(deep,'permutation_distribution')
        null_options=[]
        if not summary.empty and not distribution.empty:
            if not {'channel_a','channel_b','observed_sync'}<=set(summary):raise ValueError('Invalid summary schema')
            if summary.duplicated(['channel_a','channel_b']).any():raise ValueError('Duplicate summary pairs')
            summary=summary.sort_values('observed_sync',ascending=False)
            null_options=[(int(r.channel_a),int(r.channel_b)) for r in summary.itertuples(index=False)]
        if not null_options and all_profiles.empty:st.info(tr('missing',lang));return
    except ImportError:
        st.error('Install plotly in this Python environment: python -m pip install plotly');return
    except (ValueError,KeyError,pd.errors.ParserError) as error:st.error(str(error));return
    prefix='pair_explorer_'+hashlib.sha1(str(deep.resolve()).encode()).hexdigest()[:10]
    key=prefix+'_selected'
    only_null=st.checkbox(tr('only_null',lang),value=bool(null_options),disabled=not null_options,key=prefix+'_only_null')
    if not null_options:only_null=False
    if only_null or not {'channel_a','channel_b'}<=set(all_profiles):options=null_options
    else:
        candidate_options=[(int(r.channel_a),int(r.channel_b)) for r in all_profiles.itertuples(index=False)]
        options=list(dict.fromkeys(null_options+candidate_options))
    if not options:st.info(tr('no_data',lang));return
    if st.session_state.get(key) not in options:st.session_state[key]=options[0]
    st.caption(tr('scope',lang)+f' ({len(options)})')
    cases=case_examples=None
    try:
        from telescope_case_report import saved_paths, render_case, review_summary, words, REVIEW_FIELDS
        paths=saved_paths(analysis)
        signature=tuple((name,str(path),path.stat().st_mtime_ns,path.stat().st_size) for name,path in paths.items() if path)
        cases,case_examples,diagnostics=cached_cases(signature)
        if diagnostics['missing_optional_tables']:
            st.caption(words(lang,'Часть дополнительных таблиц отсутствует: ','Some optional tables are missing: ','Certaines tables facultatives sont absentes : ')+', '.join(diagnostics['missing_optional_tables']))
    except (ValueError,KeyError,FileNotFoundError,pd.errors.ParserError) as error:
        st.warning(str(error))
    names={}
    if cases is not None:
        for side in ['a','b']:
            names.update(dict(zip(cases['channel_'+side],cases['title_'+side].fillna(''))))
    selected=st.selectbox(tr('select',lang),options,format_func=lambda p:f'{p[0]} {names.get(p[0], "")} ↔ {p[1]} {names.get(p[1], "")}',key=key)
    left,right=st.columns(2)
    left.button(tr('previous',lang),key=prefix+'_previous',on_click=move_pair,args=(key,options,-1),disabled=len(options)<2)
    right.button(tr('next',lang),key=prefix+'_next',on_click=move_pair,args=(key,options,1),disabled=len(options)<2)
    a,b=selected
    has_null=(a,b) in null_options
    if has_null:
        try:stats,values,d=pair_stats(summary,distribution,a,b)
        except (ValueError,KeyError) as error:st.error(str(error));return
    else:
        profile=subset(all_profiles,a,b)
        observed=float(profile.n_sync_posts.iloc[0]) if len(profile) and 'n_sync_posts' in profile else np.nan
        stats={'observed':observed,'mean':np.nan,'ratio':np.nan,'n':0}
        values=np.array([]);d=pd.DataFrame()
    cols=st.columns(4)
    for c,label,value in zip(cols,[tr('observed',lang),tr('mean',lang),tr('ratio',lang),tr('trials',lang)],
                             [integer(stats['observed']) if np.isfinite(stats['observed']) else '—',number(stats['mean']) if np.isfinite(stats['mean']) else '—',f"×{stats['ratio']:.2f}" if np.isfinite(stats['ratio']) else '—',str(stats['n'])]):
        c.metric(label,value)
    if cases is not None:
        render_case(cases,case_examples,a,b,lang,prefix)
        render_review_panel(cases,deep,lang,prefix,a,b)
    tabs=st.tabs([tr('permutation',lang),tr('time',lang),tr('sources',lang),tr('data',lang)])
    config={'displaylogo':False,'toImageButtonOptions':{'format':'png','filename':f'pair_{a}_{b}','scale':2}}
    with tabs[0]:
        if has_null:
            fig=permutation_figure(stats,values,lang)
            st.plotly_chart(fig,width="stretch",key=prefix+'_permutation',config=config)
            st.caption(f"{tr('mean',lang)}: {number(stats['mean'])}; {tr('p95',lang)}: {number(stats['p95'])}. "
                       f"{tr('exceedances',lang)}: {stats['exceedances']}/{stats['n']}.")
            st.caption(tr('warning',lang))
            st.download_button(tr('download',lang),fig.to_html(include_plotlyjs=True).encode('utf-8'),file_name=f'pair_{a}_{b}.html',mime='text/html',key=prefix+'_html')
        else:st.info(tr('missing_permutation',lang))
    with tabs[1]:
        weekly=subset(optional_table(deep,'pair_weekly_activity'),a,b)
        examples=subset(optional_table(deep,'suspicious_examples'),a,b)
        if weekly.empty and examples.empty:st.info(tr('no_data',lang))
        if not weekly.empty:
            st.plotly_chart(weekly_figure(weekly,lang),width="stretch",key=prefix+'_weekly',config=config)
        if not examples.empty:
            st.caption(tr('examples_note',lang))
            st.plotly_chart(example_figure(examples,a,b,lang),width="stretch",key=prefix+'_timeline',config=config)
    with tabs[2]:
        sources=subset(optional_table(deep,'pair_source_profiles'),a,b)
        if sources.empty:st.info(tr('no_data',lang))
        else:
            top=sources.sort_values('n_sync_posts',ascending=False).head(15)
            fig=go.Figure(go.Bar(x=top.source_share,y=top.origin_channel.astype('int64').astype(str),orientation='h',marker_color='#386CB0',customdata=top.n_sync_posts,hovertemplate='%{y}<br>%{x:.1%}<br>n=%{customdata}<extra></extra>'))
            fig.update_yaxes(type='category',autorange='reversed')
            fig.update_xaxes(title=tr('source_share',lang),tickformat='.0%')
            fig.update_layout(title=tr('sources',lang),yaxis_title=tr('source_id',lang),template='plotly_white',height=max(350,len(top)*28))
            st.plotly_chart(fig,width="stretch",key=prefix+'_sources',config=config)
    with tabs[3]:
        profile=subset(all_profiles,a,b)
        if not profile.empty:st.dataframe(profile,width="stretch",hide_index=True)
        if has_null:
            st.dataframe(subset(summary,a,b),width="stretch",hide_index=True)
            st.dataframe(d,width="stretch",hide_index=True)
            st.download_button(tr('download_csv',lang),d.to_csv(index=False).encode('utf-8-sig'),file_name=f'permutations_{a}_{b}.csv',mime='text/csv',key=prefix+'_csv')


@st.cache_data(show_spinner=False)
def cached_cases(signature):
    from telescope_case_report import assemble
    paths={name:None for name in ['pair_profiles','suspicious_examples','metadata_pair_profiles','metadata_channel_profiles']}
    paths.update({name:Path(path) for name,path,mtime,size in signature})
    return assemble(paths)


def render_review_panel(cases,deep,lang,prefix,a,b):
    from telescope_review_ui import render_manual_review
    render_manual_review(cases,deep,lang,prefix,a,b)
