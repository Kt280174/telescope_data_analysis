#!/usr/bin/env python3
"""TeleScope theme 5: in-depth, exploratory analysis of existing synchronization candidates.
Reads all chains for candidate endpoints. Does not discover pairs outside the input list.
Synchrony is not a bot/spam label. Permutation results are exploratory after candidate selection.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

DAY=86400


def epoch(series):
    if pd.api.types.is_numeric_dtype(series.dtype):
        return pd.to_datetime(series,unit='s',utc=True,errors='coerce')
    return pd.to_datetime(series,utc=True,errors='coerce',format='ISO8601')


def load_chains(path, endpoints):
    import pyarrow.dataset as ds
    import pyarrow.compute as pc
    dataset=ds.dataset(str(path),format='parquet')
    needed=['origin_channel','origin_post','dest_channel','date']
    if set(needed)-set(dataset.schema.names):
        raise ValueError('chains must contain '+', '.join(needed))
    coverage={'all_chain_rows':dataset.count_rows(),'candidate_endpoint_channels':len(endpoints)}
    scanner=dataset.scanner(columns=needed,filter=ds.field('dest_channel').isin(sorted(endpoints)),batch_size=500_000)
    parts=[]; missing=invalid_time=selfrows=raw=0
    for n,batch in enumerate(scanner.to_batches(),1):
        d=batch.to_pandas(types_mapper=pd.ArrowDtype)
        raw+=len(d)
        bad=d[['origin_channel','origin_post','dest_channel']].isna().any(axis=1)
        missing+=int(bad.sum()); d=d.loc[~bad].copy()
        for col in ['origin_channel','origin_post','dest_channel']: d[col]=d[col].astype('int64')
        date=epoch(d['date']); invalid_time+=int(date.isna().sum())
        d=d.loc[date.notna()].copy(); date=date.loc[date.notna()]
        d['ts']=((date-pd.Timestamp('1970-01-01',tz='UTC'))//pd.Timedelta(seconds=1)).astype('int64')
        d=d.drop(columns='date')
        selfmask=d.origin_channel.eq(d.dest_channel); selfrows+=int(selfmask.sum())
        parts.append(d.loc[~selfmask])
        if n%10==0: print(f'chains: {raw:,} endpoint rows scanned',flush=True)
    if not parts: raise ValueError('No chain rows for candidate endpoints')
    d=pd.concat(parts,ignore_index=True); del parts
    coverage.update({'endpoint_rows_scanned':raw,'endpoint_rows_missing_keys':missing,
                     'endpoint_rows_invalid_time':invalid_time,'endpoint_self_forwards_removed':selfrows})
    before=len(d)
    d=d.groupby(['origin_channel','origin_post','dest_channel'],sort=False,as_index=False).ts.min()
    coverage['later_same_channel_post_relays_removed']=before-len(d)
    coverage['earliest_endpoint_channel_post_rows']=len(d)
    keys=pd.MultiIndex.from_frame(d[['origin_channel','origin_post']])
    codes,unique=pd.factorize(keys,sort=False)
    origins=np.asarray(unique.get_level_values(0),dtype=np.int64)
    posts=np.asarray(unique.get_level_values(1),dtype=np.int64)
    d['code']=codes
    channels={}
    for channel,g in d.groupby('dest_channel',sort=False):
        g=g.sort_values('code')
        channels[int(channel)]=(g.code.to_numpy(dtype=np.int64),g.ts.to_numpy(dtype=np.int64))
    coverage['distinct_original_posts_in_endpoint_corpus']=len(unique)
    coverage['channels_with_valid_relays']=len(channels)
    coverage['memory_prepared_dataframe_gib']=round(d.memory_usage(deep=True).sum()/1024**3,3)
    del d,keys
    return channels,origins,posts,coverage


def common(channels,a,b):
    ca,ta=channels.get(int(a),(np.array([],dtype=np.int64),np.array([],dtype=np.int64)))
    cb,tb=channels.get(int(b),(np.array([],dtype=np.int64),np.array([],dtype=np.int64)))
    codes,ia,ib=np.intersect1d(ca,cb,assume_unique=True,return_indices=True)
    return codes,ta[ia],tb[ib],ia,ib


def feature_table(pairs,channels,origins,window,windows,min_posts,min_days):
    rows=[]
    for i,p in enumerate(pairs.itertuples(index=False)):
        a,b=int(p.channel_a),int(p.channel_b)
        codes,ta,tb,_,_=common(channels,a,b)
        gap=np.abs(tb-ta); sync=gap<=window
        n=int(sync.sum()); shared=len(codes)
        times=np.minimum(ta,tb)[sync]; signed=(tb-ta)[sync]
        days,daycounts=np.unique(times//DAY,return_counts=True)
        source,sourcecounts=np.unique(origins[codes[sync]],return_counts=True)
        counta=len(channels.get(a,([],[]))[0]); countb=len(channels.get(b,([],[]))[0])
        row={'channel_a':a,'channel_b':b,'n_sync_posts':n,'n_shared_posts':shared,'n_sync_days':len(days),
             'posts_a':counta,'posts_b':countb,'sync_share':n/shared if shared else np.nan,
             'overlap_share':shared/min(counta,countb) if min(counta,countb) else np.nan,
             'median_gap_s':float(np.median(gap[sync])) if n else np.nan,
             'p90_gap_s':float(np.quantile(gap[sync],.9)) if n else np.nan,
             'a_precedes_b_share':float((signed>0).mean()) if n else np.nan,
             'b_precedes_a_share':float((signed<0).mean()) if n else np.nan,
             'exact_tie_share':float((signed==0).mean()) if n else np.nan,
             'max_day_share':int(daycounts.max())/n if n else np.nan,
             'n_sources':len(source),'top_source_share':int(sourcecounts.max())/n if n else np.nan,
             'top_source_channel':int(source[np.argmax(sourcecounts)]) if n else None,
             'source_entropy_bits':float(-np.sum((sourcecounts/n)*np.log2(sourcecounts/n))) if n else np.nan,
             'first_sync_utc':pd.to_datetime(int(times.min()),unit='s',utc=True) if n else pd.NaT,
             'last_sync_utc':pd.to_datetime(int(times.max()),unit='s',utc=True) if n else pd.NaT,
             'eligible_at_base_window':bool(n>=min_posts and len(days)>=min_days)}
        for w in windows:
            mask=gap<=w
            row[f'n_sync_{w}s']=int(mask.sum())
            row[f'n_days_{w}s']=int(len(np.unique(np.minimum(ta,tb)[mask]//DAY)))
        rows.append(row)
        if (i+1)%1000==0: print(f'Pair profiles: {i+1:,}/{len(pairs):,}',flush=True)
    return pd.DataFrame(rows).sort_values(['n_sync_posts','sync_share','channel_a','channel_b'],ascending=[False,False,True,True]).reset_index(drop=True)


def add_context(p,communities,geo):
    p=p.copy(); cities={}
    if communities.exists():
        c=pd.read_csv(communities,usecols=['channel_id','community'])
        if c.channel_id.duplicated().any(): raise ValueError('Duplicate community assignments')
        cmap=c.set_index('channel_id').community
        for side in ['a','b']: p[f'community_{side}']=p[f'channel_{side}'].map(cmap)
        known=p.community_a.notna()&p.community_b.notna()
        p['same_community']=pd.Series(pd.NA,index=p.index,dtype='boolean')
        p.loc[known,'same_community']=p.loc[known,'community_a'].eq(p.loc[known,'community_b'])
    if geo.exists():
        g=pd.read_csv(geo); g['city_key']=g.city.astype(str)+' ('+g.country.astype(str)+')'
        # Only unambiguous geographic hypotheses are used for episode linkage.
        if 'n_cities' in g: g=g[g.n_cities.eq(1)]
        for ch,group in g.groupby('channel_id'):
            unique=group.city_key.unique()
            if len(unique)==1: cities[int(ch)]=unique[0]
        for side in ['a','b']: p[f'city_{side}']=p[f'channel_{side}'].map(cities)
    return p,cities


def details(profiles,channels,origins,posts,window,n,seed):
    sources=[]; weekly=[]; examples=[]; null_inputs={}
    for rank,p in enumerate(profiles.head(n).itertuples(index=False),1):
        a,b=int(p.channel_a),int(p.channel_b)
        codes,ta,tb,ia,ib=common(channels,a,b)
        mask=np.abs(tb-ta)<=window; k=codes[mask]; x=ta[mask]; y=tb[mask]
        if not len(k): continue
        o,counts=np.unique(origins[k],return_counts=True)
        order=np.argsort(-counts)
        for j in order[:20]: sources.append((a,b,int(o[j]),int(counts[j]),int(counts[j])/len(k)))
        dates=pd.to_datetime(np.minimum(x,y),unit='s',utc=True)
        week=dates.tz_localize(None).to_period('W-SUN').start_time
        w=pd.Series(week).value_counts().sort_index()
        for date,count in w.items(): weekly.append((a,b,date,int(count)))
        # Stratified across the observed chronology rather than just the first few rows.
        order=np.argsort(np.minimum(x,y))
        selected=order[np.unique(np.linspace(0,len(order)-1,min(60,len(order)),dtype=int))]
        for j in selected:
            examples.append((rank,a,b,int(origins[k[j]]),int(posts[k[j]]),
                pd.to_datetime(int(x[j]),unit='s',utc=True),pd.to_datetime(int(y[j]),unit='s',utc=True),int(abs(y[j]-x[j]))))
        null_inputs[a,b]=(ia,ib)
    return (pd.DataFrame(sources,columns=['channel_a','channel_b','origin_channel','n_sync_posts','source_share']),
            pd.DataFrame(weekly,columns=['channel_a','channel_b','week','n_sync_posts']),
            pd.DataFrame(examples,columns=['pair_rank','channel_a','channel_b','origin_channel','origin_post','date_a','date_b','gap_s']),null_inputs)


def day_groups(ts):
    order=np.argsort(ts//DAY,kind='stable'); d=ts[order]//DAY
    splits=np.flatnonzero(d[1:]!=d[:-1])+1
    return [x for x in np.split(order,splits) if len(x)>1]


def shuffle_times(ts,groups,rng):
    out=ts.copy()
    for idx in groups: out[idx]=rng.permutation(ts[idx])
    return out


def permutation_test(profiles,channels,inputs,window,repeats,top,seed):
    rng=np.random.default_rng(seed); rows=[]; dist=[]
    for p in profiles.head(top).itertuples(index=False):
        pair=int(p.channel_a),int(p.channel_b)
        if pair not in inputs: continue
        ia,ib=inputs[pair]; _,ta=channels[pair[0]]; _,tb=channels[pair[1]]
        ga,gb=day_groups(ta),day_groups(tb)
        null=[]
        for trial in range(repeats):
            sa=shuffle_times(ta,ga,rng); sb=shuffle_times(tb,gb,rng)
            count=int((np.abs(sa[ia]-sb[ib])<=window).sum()); null.append(count)
            dist.append((*pair,trial,count))
            if (trial+1)%50==0: print(f'Permutation {pair}: {trial+1}/{repeats}',flush=True)
        observed=int(p.n_sync_posts); baseline=float(np.mean(null))
        movable_a=sum(len(g) for g in ga if np.unique(ta[g]).size>1)
        movable_b=sum(len(g) for g in gb if np.unique(tb[g]).size>1)
        informative=bool(movable_a or movable_b)
        rows.append((*pair,observed,baseline,float(np.quantile(null,.95)),
                     observed/baseline if baseline>0 else np.nan,
                     (1+sum(v>=observed for v in null))/(repeats+1) if informative else np.nan,
                     repeats,movable_a,movable_b,informative))
        print(f'Exploratory permutations: {pair}',flush=True)
    cols=['channel_a','channel_b','observed_sync','null_mean','null_p95','observed_over_null_mean',
          'exploratory_tail_fraction','n_permutations','movable_rows_a','movable_rows_b','informative_shuffle']
    return pd.DataFrame(rows,columns=cols),pd.DataFrame(dist,columns=['channel_a','channel_b','trial','null_sync_posts'])


def network_tables(profiles,min_posts,min_days,min_sync,min_overlap):
    edges=profiles[(profiles.n_sync_posts>=min_posts)&(profiles.n_sync_days>=min_days)&
                   (profiles.sync_share>=min_sync)&(profiles.overlap_share>=min_overlap)].copy()
    parent={}
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    for p in edges.itertuples(index=False):
        a,b=find(int(p.channel_a)),find(int(p.channel_b)); parent[a]=b
    roots=defaultdict(list)
    for node in sorted(parent): roots[find(node)].append(node)
    groups=sorted(roots.values(),key=lambda x:(-len(x),x[0])); membership={ch:i+1 for i,g in enumerate(groups) for ch in g}
    edges['group_id']=edges.channel_a.map(membership)
    nodes=pd.DataFrame([(ch,g) for ch,g in sorted(membership.items())],columns=['channel_id','group_id'])
    rows=[]
    for gid,group in enumerate(groups,1):
        e=edges[edges.group_id.eq(gid)]; size=len(group)
        rows.append((gid,size,len(e),2*len(e)/(size*(size-1)),int(e.n_sync_posts.sum()),';'.join(map(str,group[:100]))))
    return edges,nodes,pd.DataFrame(rows,columns=['group_id','n_channels','n_edges','edge_density','sum_pair_sync_counts','channel_sample'])


def link_episodes(profiles,channels,cities,path,window,top):
    cols=['channel_a','channel_b','city_key','episode_start','episode_end','episode_shape','n_sync_posts_in_episode','n_sync_posts_all']
    if not path.exists(): return pd.DataFrame(columns=cols),'episode table unavailable'
    eps=pd.read_csv(path);eps['start']=pd.to_datetime(eps.start,utc=True);eps['end_exclusive']=pd.to_datetime(eps.end,utc=True)+pd.Timedelta(days=1)
    rows=[]
    for p in profiles.head(top).itertuples(index=False):
        a,b=int(p.channel_a),int(p.channel_b); _,ta,tb,_,_=common(channels,a,b)
        mask=np.abs(tb-ta)<=window
        for city in sorted({cities[x] for x in [a,b] if x in cities}):
            # Use the relay time at the endpoint linked to this city, not a global min.
            times=[]
            if cities.get(a)==city: times.append(ta[mask])
            if cities.get(b)==city: times.append(tb[mask])
            for e in eps[eps.city_key.eq(city)].itertuples(index=False):
                lo=int(e.start.timestamp()); hi=int(e.end_exclusive.timestamp())
                hit=np.zeros(int(mask.sum()),dtype=bool)
                for t in times: hit|=(t>=lo)&(t<hi)
                if hit.any(): rows.append((a,b,city,e.start,e.end_exclusive-pd.Timedelta(days=1),e.shape,int(hit.sum()),int(mask.sum())))
    return pd.DataFrame(rows,columns=cols),'selected detail pairs; unambiguous city mapping; endpoint relay times; no causal attribution or city-wide contribution denominator'


def interaction_context(clean,profiles,channels,origins,posts,window,top,min_ref,max_reference_rows):
    cols=['origin_channel','origin_post','views','forwards','reference_posts_in_channel_year',
          'forward_percentile_within_channel_year','views_percentile_within_channel_year','forward_view_ratio','valid_forward_references','valid_view_references','reference_sufficient']
    if clean is None or not clean.exists(): return pd.DataFrame(columns=cols),'clean_dedup unavailable; interaction analysis not performed'
    import pyarrow.dataset as ds
    targets=set()
    for p in profiles.head(top).itertuples(index=False):
        k,ta,tb,_,_=common(channels,int(p.channel_a),int(p.channel_b));k=k[np.abs(tb-ta)<=window]
        targets.update((int(origins[c]),int(posts[c])) for c in k)
    if not targets: return pd.DataFrame(columns=cols),'no target posts'
    source_ids=sorted({c for c,_ in targets})
    dataset=ds.dataset(str(clean),format='parquet',partitioning='hive')
    needed=['dest_channel','msg_id','ts','views','forwards']
    if set(needed)-set(dataset.schema.names): raise ValueError('clean_dedup schema missing interaction columns')
    frames=[]
    scanner=dataset.scanner(columns=needed,filter=ds.field('dest_channel').isin(source_ids),batch_size=500_000)
    reference_rows=0
    for b in scanner.to_batches():
        reference_rows+=b.num_rows
        if reference_rows>max_reference_rows:
            return pd.DataFrame(columns=cols),f'interaction analysis not performed: selected source references exceed {max_reference_rows:,} rows; raise --max-reference-rows or reduce --detail-pairs explicitly'
        frames.append(b.to_pandas(types_mapper=pd.ArrowDtype))
    if not frames: return pd.DataFrame(columns=cols),'source channels absent from clean corpus'
    d=pd.concat(frames,ignore_index=True); del frames
    d=d.dropna(subset=['dest_channel','msg_id','ts'])
    for c in ['dest_channel','msg_id','ts']: d[c]=d[c].astype('int64')
    if d.duplicated(['dest_channel','msg_id']).any(): raise ValueError('Interaction input is not deduplicated; use clean_dedup')
    for c in ['views','forwards']:
        d[c]=pd.to_numeric(d[c],errors='coerce').astype('float64')
        d.loc[d[c]<0,c]=np.nan
    d['year']=pd.to_datetime(d.ts,unit='s',utc=True,errors='coerce').dt.year
    by=d.groupby(['dest_channel','year'],dropna=False)
    d['reference_posts_in_channel_year']=by.msg_id.transform('size')
    d['valid_forward_references']=by.forwards.transform('count')
    d['valid_view_references']=by.views.transform('count')
    d['forward_percentile_within_channel_year']=by.forwards.rank(pct=True)
    d['views_percentile_within_channel_year']=by.views.rank(pct=True)
    # A descriptive count ratio, not a forwarding probability. Both counters may be missing.
    d['forward_view_ratio']=d.forwards.div(d.views.where(d.views>=100))
    key=pd.MultiIndex.from_frame(d[['dest_channel','msg_id']]); selected=key.isin(list(targets))
    out=d.loc[selected].rename(columns={'dest_channel':'origin_channel','msg_id':'origin_post'})
    out['reference_sufficient']=out.valid_forward_references>=min_ref
    return out[cols],f'original source posts of detail pairs only; {len(targets)} requested; {len(out)} found; channel-year reference, descriptive only; original post age and collection time not controlled'


def visuals(out,figdir,p,source,weekly,examples,null,dist,edges,nodes,groups,interactions,seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.tight_layout();fig.savefig(figdir/f'{name}.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));ax.scatter(p.n_sync_posts,p.sync_share,s=9,alpha=.3)
    ax.set(xlabel='Synchronized original posts',ylabel='Synchronized / shared posts',title='Repeated synchrony: selected candidate pairs');ax.set_xscale('log');save(fig,'sync_support_vs_share')
    fig,ax=plt.subplots(figsize=(9,5));ax.hist(p.median_gap_s.dropna(),bins=30)
    ax.set(xlabel='Median synchronized gap, seconds',ylabel='Candidate pairs',title='Time proximity of repeated forwards');save(fig,'pair_delay_distribution')
    fig,ax=plt.subplots(figsize=(9,5));ax.scatter(p.n_sync_days,p.max_day_share,s=9,alpha=.3)
    ax.set(xlabel='Distinct UTC days with synchrony',ylabel='Share of sync posts on busiest day',title='Persistent vs event-concentrated synchrony');save(fig,'temporal_persistence')
    if not source.empty:
        a,b=source.iloc[0][['channel_a','channel_b']].astype(int);s=source[source.channel_a.eq(a)&source.channel_b.eq(b)].head(10)
        fig,ax=plt.subplots(figsize=(9,5));ax.barh(s.origin_channel.astype(str),s.source_share)
        ax.set(xlabel='Share of synchronized posts',title=f'Source concentration: {a} / {b}');save(fig,'top_pair_sources')
    if not weekly.empty:
        fig,ax=plt.subplots(figsize=(10,5))
        for (a,b),g in weekly.groupby(['channel_a','channel_b'],sort=False):
            if len(ax.lines)>=5: break
            # Explicitly include zero weeks between observations.
            s=g.set_index('week').n_sync_posts;s=s.reindex(pd.date_range(s.index.min(),s.index.max(),freq='7D'),fill_value=0)
            ax.plot(s.index,s.values,label=f'{a}/{b}',alpha=.8)
        ax.set(xlabel='UTC week',ylabel='Synchronized original posts',title='Weekly activity of selected pairs');ax.legend(fontsize=7);save(fig,'pair_weekly_activity')
    if not examples.empty:
        a,b=examples.iloc[0][['channel_a','channel_b']].astype(int);e=examples[examples.channel_a.eq(a)&examples.channel_b.eq(b)].sort_values('date_a').head(24)
        # Relative-to-A raster avoids unreadable multi-year datetime axes.
        fig,ax=plt.subplots(figsize=(9,7))
        gap=(pd.to_datetime(e.date_b)-pd.to_datetime(e.date_a)).dt.total_seconds().to_numpy()
        yy=np.arange(len(e));ax.scatter(np.zeros(len(e)),yy,label=f'Channel {a}');ax.scatter(gap,yy,label=f'Channel {b}')
        for j,x in enumerate(gap): ax.plot([0,x],[j,j],color='grey',alpha=.4)
        labels=[f'{d:%Y-%m-%d} | {int(o)}/{int(post)}' for d,o,post in zip(pd.to_datetime(e.date_a),e.origin_channel,e.origin_post)]
        ax.set_yticks(yy,labels,fontsize=6);ax.set(xlabel='Seconds relative to relay at channel A',title='Selected synchronized examples: relative-time raster');ax.legend();save(fig,'pair_example_timeline')
    if not null.empty:
        r=null.iloc[0];d=dist[dist.channel_a.eq(r.channel_a)&dist.channel_b.eq(r.channel_b)]
        fig,ax=plt.subplots(figsize=(9,5));ax.hist(d.null_sync_posts,bins=min(25,len(d)));ax.axvline(r.observed_sync,color='red',label='Observed')
        ax.set(xlabel='Synchronized original posts',ylabel='Permutations',title='Within-day time shuffling: exploratory selected pair');ax.legend();save(fig,'observed_vs_permuted')
    if not interactions.empty:
        valid=interactions[interactions.reference_sufficient]
        fig,ax=plt.subplots(figsize=(9,5));ax.hist(valid.forward_percentile_within_channel_year.dropna(),bins=np.linspace(0,1,21))
        ax.set(xlabel='Forward-counter percentile in source channel/year',ylabel='Selected original posts',title='Interaction context (not proof of manipulated engagement)');save(fig,'interaction_context')
    # Small filtered components only; not one unreadable full-corpus graph.
    if not groups.empty:
        import plotly.graph_objects as go
        chosen=groups[groups.n_channels.le(60)].head(8)
        traces=[];buttons=[]
        rng=np.random.default_rng(seed)
        for row in chosen.itertuples(index=False):
            ns=nodes[nodes.group_id.eq(row.group_id)].channel_id.tolist();es=edges[edges.group_id.eq(row.group_id)]
            angles=np.linspace(0,2*np.pi,len(ns),endpoint=False);pos={ch:(np.cos(t),np.sin(t)) for ch,t in zip(ns,angles)}
            xx=[];yy=[]
            for edge in es.itertuples(index=False):
                xx.extend([pos[edge.channel_a][0],pos[edge.channel_b][0],None]);yy.extend([pos[edge.channel_a][1],pos[edge.channel_b][1],None])
            visible=len(traces)==0
            traces.append(go.Scatter(x=xx,y=yy,mode='lines',line=dict(width=1,color='#8899aa'),hoverinfo='skip',visible=visible,showlegend=False))
            traces.append(go.Scatter(x=[pos[ch][0] for ch in ns],y=[pos[ch][1] for ch in ns],mode='markers',text=[str(ch) for ch in ns],hovertemplate='Channel %{text}<extra></extra>',marker=dict(size=12),visible=visible,showlegend=False))
        if traces:
            for i,row in enumerate(chosen.itertuples(index=False)):
                visibility=[j//2==i for j in range(len(traces))]
                buttons.append(dict(label=f'Group {row.group_id}: {row.n_channels} channels',method='update',args=[{'visible':visibility}]))
            fig=go.Figure(traces);fig.update_layout(title='Filtered synchrony components — exploratory, not campaigns',updatemenus=[dict(buttons=buttons)],xaxis=dict(visible=False),yaxis=dict(visible=False,scaleanchor='x'))
            fig.write_html(str(figdir/'map_sync_groups.html'),include_plotlyjs=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pairs',type=Path,default=Path('bigdata_pipeline/analysis/suspicious/suspicious_pairs.csv'))
    ap.add_argument('--chains',type=Path,default=Path('telescope_pipeline_files/gold/chains'))
    ap.add_argument('--clean',type=Path,default=None,help='optional clean_dedup directory for interaction context')
    ap.add_argument('--communities',type=Path,default=Path('bigdata_pipeline/analysis/communities_v2/communities_leiden/communities_leiden.csv'))
    ap.add_argument('--geo',type=Path,default=Path('bigdata_pipeline/analysis/geo_cities/channel_geo/channel_geo.csv'))
    ap.add_argument('--episodes',type=Path,default=Path('bigdata_pipeline/analysis/anomalies/anomaly_episodes/anomaly_episodes.csv'))
    ap.add_argument('--out',type=Path,default=Path('bigdata_pipeline/analysis/suspicious_deep'))
    ap.add_argument('--figdir',type=Path,default=Path('bigdata_pipeline/analysis/figures/suspicious_deep'))
    ap.add_argument('--window',type=int,default=60)
    ap.add_argument('--windows',default='30,60,300')
    ap.add_argument('--min-posts',type=int,default=10)
    ap.add_argument('--min-days',type=int,default=3)
    ap.add_argument('--detail-pairs',type=int,default=30)
    ap.add_argument('--null-top',type=int,default=10)
    ap.add_argument('--permutations',type=int,default=200)
    ap.add_argument('--group-min-posts',type=int,default=50)
    ap.add_argument('--group-min-days',type=int,default=7)
    ap.add_argument('--group-min-sync',type=float,default=.8)
    ap.add_argument('--group-min-overlap',type=float,default=.5)
    ap.add_argument('--min-reference',type=int,default=30)
    ap.add_argument('--max-reference-rows',type=int,default=2_000_000)
    ap.add_argument('--sample-size',type=int,default=100)
    ap.add_argument('--review',type=Path)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--no-plots',action='store_true')
    ap.add_argument('--report-only',action='store_true',help='Build section 5.5 from saved CSVs; never reads chains or runs permutations')
    ap.add_argument('--metadata-out',type=Path,help='Existing suspicious_metadata output directory')
    ap.add_argument('--case-out',type=Path,help='Separate case-report destination (default: <out>/case_report)')
    ap.add_argument('--case-review',type=Path,help='Optional reviewed case template; separate from the original --review format')
    a=ap.parse_args()
    if a.report_only:
        from telescope_case_report import export_saved
        output,info=export_saved(a.out,a.metadata_out,a.case_out,a.case_review,a.sample_size,a.seed)
        print('Saved-results mode: no chains loading or permutation reruns.',flush=True)
        print(json.dumps(info,ensure_ascii=False,indent=2))
        print('Section 5.5 report:',output)
        return
    from suspicious_analysis_th5 import load_pairs, write_table, review_summary
    windows=sorted(set(map(int,a.windows.split(',')))|{a.window})
    if min(windows+[a.min_posts,a.min_days,a.detail_pairs,a.permutations,a.sample_size,a.min_reference,a.max_reference_rows,a.group_min_posts,a.group_min_days])<1 or a.null_top<0:
        ap.error('window sizes and counts must be positive; null-top may be 0')
    if not (0<=a.group_min_sync<=1 and 0<=a.group_min_overlap<=1): ap.error('group shares must be in [0,1]')
    if a.out.resolve()==a.pairs.parent.resolve(): ap.error('output must not overwrite original candidate directory')
    a.out.mkdir(parents=True,exist_ok=True)
    candidates=load_pairs(a.pairs)
    if candidates.empty: raise SystemExit('No candidates supplied')
    endpoints=set(candidates.channel_a)|set(candidates.channel_b)
    print('Preparing chains for candidate endpoints. Full selected-corpus pandas RAM is required.',flush=True)
    channels,origins,posts,coverage=load_chains(a.chains,endpoints)
    profiles=feature_table(candidates,channels,origins,a.window,windows,a.min_posts,a.min_days)
    profiles,cities=add_context(profiles,a.communities,a.geo)
    source,weekly,examples,null_inputs=details(profiles,channels,origins,posts,a.window,max(a.detail_pairs,a.null_top),a.seed)
    null,dist=permutation_test(profiles,channels,null_inputs,a.window,a.permutations,a.null_top,a.seed)
    edges,nodes,groups=network_tables(profiles,a.group_min_posts,a.group_min_days,a.group_min_sync,a.group_min_overlap)
    linkage,link_scope=link_episodes(profiles,channels,cities,a.episodes,a.window,a.detail_pairs)
    interaction,interaction_scope=interaction_context(a.clean,profiles,channels,origins,posts,a.window,a.detail_pairs,a.min_reference,a.max_reference_rows)
    # Compare old vs recomputed base-window metrics to detect stale or mismatched inputs.
    compare=candidates[['channel_a','channel_b','n_sync_posts','n_shared_posts']].merge(profiles[['channel_a','channel_b','n_sync_posts','n_shared_posts']],on=['channel_a','channel_b'],suffixes=('_input','_recomputed'),validate='one_to_one')
    compare['counts_agree']=compare.n_sync_posts_input.eq(compare.n_sync_posts_recomputed)&compare.n_shared_posts_input.eq(compare.n_shared_posts_recomputed)
    sensitivity=[]
    for w in windows:
        n=profiles[f'n_sync_{w}s']; days=profiles[f'n_days_{w}s'];eligible=(n>=a.min_posts)&(days>=a.min_days)
        sensitivity.append({'window_seconds':w,'eligible_pairs_within_input_candidates':int(eligible.sum()),'sum_pair_sync_posts':int(n.sum())})
    review_path=a.out/'manual_review_sample.csv'
    if not review_path.exists():
        sample=profiles.sample(n=min(a.sample_size,len(profiles)),random_state=a.seed).copy();sample['review_label']='';sample['review_comment']=''
        sample.to_csv(review_path,index=False,encoding='utf-8-sig')
    review=review_summary(pd.read_csv(a.review or review_path),profiles)
    summary={'input_candidate_pairs':len(candidates),'pairs_still_eligible':int(profiles.eligible_at_base_window.sum()),
             'candidate_endpoint_channels':len(endpoints),'pairs_with_count_mismatch':int((~compare.counts_agree).sum()),
             'median_sync_posts':profiles.n_sync_posts.median(),'median_sync_days':profiles.n_sync_days.median(),
             'median_sync_share':profiles.sync_share.median(),'filtered_graph_components':len(groups),
             'exploratory_null_pairs':len(null),'detail_pair_episode_links':len(linkage),
             'source_posts_with_interaction_context':len(interaction)}
    tables=[('detection_coverage',pd.DataFrame(coverage.items(),columns=['metric','value']),'Coverage: quality counts apply only to selected candidate endpoints, not the entire Telegram corpus.'),
        ('suspicious_summary',pd.DataFrame(summary.items(),columns=['metric','value']),'Descriptive counts of existing candidates; no bot/spam labels.'),
        ('pair_profiles',profiles,'Earliest relay per endpoint and original post. Direction refers to time order, not forwarding hops.'),
        ('input_reconciliation',compare,'Check old metrics against newly read chains; mismatches may indicate changed data or detector settings.'),
        ('window_sensitivity',pd.DataFrame(sensitivity),'Conditional on the supplied candidate list; not a full rediscovery at other windows. Pair-event totals are not unique corpus events.'),
        ('pair_source_profiles',source,'Top source channels for selected detail pairs only.'),
        ('pair_weekly_activity',weekly,'Synchronized posts by UTC week for selected detail pairs; missing interior weeks are zero.'),
        ('suspicious_examples',examples,'At most 60 examples per detail pair, sampled evenly across chronology. Not a random validation sample.'),
        ('permutation_summary',null,'Exploratory within-channel/day time permutation on full channel schedules; selected top pairs, no confirmatory p-values, FDR guarantee or causal conclusion.'),
        ('permutation_distribution',dist,'All exploratory permutation counts, including ties; finite simulation resolution.'),
        ('sync_group_edges',edges,'Filtered synchrony graph edges, not confirmed campaign relationships.'),
        ('sync_group_nodes',nodes,'Connected components of the filtered graph; transitive membership does not imply every pair synchronizes.'),
        ('sync_group_summary',groups,'Exploratory components. Sum of edge-event counts double-counts posts shared across pairs.'),
        ('anomaly_episode_links',linkage,link_scope),
        ('interaction_context',interaction,interaction_scope),
        ('review_summary',review,'Manual evidence assessment. False alarms / decided cases only; missing reviews are not normal or confirmed. Not corpus FPR.')]
    for name,frame,description in tables: write_table(a.out,name,frame,description)
    if not a.no_plots: visuals(a.out,a.figdir,profiles,source,weekly,examples,null,dist,edges,nodes,groups,interaction,a.seed)
    info={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'parameters':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
          'interaction_scope':interaction_scope,'episode_link_scope':link_scope,
          'null_scope':'exploratory because selection used observed synchrony; no multiplicity-adjusted inference',
          'coverage_scope':'full chains row count from metadata; cleaning and activity metrics restricted to candidate endpoints'}
    (a.out/'run_info.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    rate=review.false_alarm_share_among_decided.iloc[0]
    rv='Ручная проверка не завершена; доля ложных тревог не оценена.' if pd.isna(rate) else f'Доля ложных тревог среди случаев с определённой ручной меткой: {rate:.1%}. Неопределённые и непроверенные случаи приведены отдельно в review_summary.'
    text=f'''# 5. Анализ распространения и взаимодействий для выявления подозрительных тенденций

## Методика
Использован ранее сформированный список из {len(candidates):,} пар каналов. Для каждой пары пересчитаны общие исходные публикации и синхронные пересылки в пределах {a.window} с; взята первая пересылка каждого исходного сообщения каждым каналом. Дополнительно рассмотрены окна {', '.join(map(str,windows))} с. Анализ ограничен исходным списком кандидатов и не оценивает долю подозрительного поведения во всём Telegram.

## Повторяемость и источники
Медианное число синхронных публикаций: {summary['median_sync_posts']:g}; медианное число дней: {summary['median_sync_days']:g}; медианная доля синхронных публикаций среди общих: {summary['median_sync_share']:.1%}. Для пар рассчитаны концентрация источников, энтропия источников, временная близость, порядок публикации и концентрация по дням. Это описательные признаки, не вероятность злонамеренного поведения.

## Структура групп
При заданных фильтрах найдено {len(groups)} связных компонент. Компоненты не интерпретируются как подтверждённые кампании; транзитивная связь не доказывает согласованность всех участников.

## Проверка и контекст
Перестановочный анализ выполнен для {len(null)} отобранных пар. Он сохраняет набор публикаций и временной профиль каждого канала внутри суток, но нарушает соответствие публикации и времени. Из-за предварительного отбора пар результат носит исследовательский характер. {interaction_scope}. Связи с эпизодами части 4 исследованы только для выбранных пар и однозначно сопоставленных городов, без утверждения причинности.

## 5.5 Проверка случаев и ограничения
{rv}

Синхронность может объясняться общей редакцией, автоматическим репостингом и независимой реакцией на новости; она не доказывает ботов, спам или злоупотребление. Конкретные примеры выбираются из suspicious_examples и проверяются по нескольким публикациям и дням. Количество пар не равно количеству независимых групп. При несовпадении исходных и пересчитанных метрик необходимо проверить input_reconciliation до окончательного отчёта.
'''
    (a.out/'section_5_draft.md').write_text(text,encoding='utf-8')
    print(pd.DataFrame(summary.items(),columns=['metric','value']).to_string(index=False))
    print(f'Done. Tables: {a.out}; charts: {a.figdir}. Manual review labels are preserved.')

if __name__=='__main__': main()
