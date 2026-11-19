#!/usr/bin/env python3
"""Theme 5 metadata extension. No raw messages or chains rerun required.
Reads unchanged Source_Channel_Metadata.csv (id,data JSON), 24-hour profiles and existing candidate pairs.
Platform flags and metadata similarities are context, NOT bot/spam ground truth.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit,urlunsplit,parse_qsl,urlencode
import numpy as np
import pandas as pd

FLAGS=['scam','fake','restricted','verified']
META=['channel_id','entity_kind','title','username','about','participants_count','metadata_date',
      'recorded_creation_date','creation_field','metadata_ambiguous']+FLAGS
LINK_PATTERN=re.compile(r'https?://[^\s<>"\[\]{}]+|(?<![\w@])(?:www\.|t\.me/|telegram\.me/)[^\s<>"\[\]{}]+',re.I)
MENTION=re.compile(r'(?<![\w@])@([A-Za-z][A-Za-z0-9_]{4,31})\b')
NULL_LANG={'','nan','none','null','unknown','cannot determine','cannot_determine','und','n/a'}


def resolve_file(path):
    if path.exists(): return path
    if path.parent.exists():
        hits=[p for p in path.parent.iterdir() if p.is_file() and p.name.lower()==path.name.lower()]
        if len(hits)==1: return hits[0]
    return path


def canonical(s): return re.sub(r'[^a-z0-9]+','_',str(s).strip().lower()).strip('_')


def channel_id(value):
    if value is None: return None
    text=str(value).strip()
    if re.fullmatch(r'\d+(?:\.0+)?',text): return int(text.split('.')[0])
    stem=Path(text.replace('\\','/')).stem
    if re.fullmatch(r'\d+',stem): return int(stem)
    return None                 # Do not guess conversions of marked -100... peer IDs.


def read_dicts(path):
    size=sys.maxsize
    while True:
        try: csv.field_size_limit(size);break
        except OverflowError:size//=10
    with path.open(encoding='utf-8-sig',newline='',errors='replace') as f:
        sample=f.read(65536);f.seek(0)
        try: dialect=csv.Sniffer().sniff(sample,delimiters=',\t;')
        except csv.Error:dialect=csv.excel
        reader=csv.DictReader(f,dialect=dialect)
        for row in reader: yield row


def known_bool(value):
    if isinstance(value,bool):return value
    if isinstance(value,(int,float)) and value in (0,1):return bool(value)
    if isinstance(value,str) and value.strip().lower() in {'true','false','0','1'}:
        return value.strip().lower() in {'true','1'}
    return pd.NA


def parse_date(value):
    if value is None or str(value).strip()=='':return pd.NaT
    try:
        if isinstance(value,(int,float)) or re.fullmatch(r'\d{10}(?:\.\d+)?',str(value)):
            return pd.to_datetime(value,unit='s',utc=True,errors='coerce')
        return pd.to_datetime(str(value),utc=True,errors='coerce')
    except (TypeError,ValueError,OverflowError):return pd.NaT


def extract_channel(obj,row_id):
    if not isinstance(obj,dict):return None,'non_object',{}
    tag=str(obj.get('_',''))
    if tag in {'User','UserEmpty','UserFull','PeerUser'} or obj.get('bot') is True:
        return None,'user_or_bot',{}
    if tag in {'Chat','ChatEmpty','ChatForbidden','PeerChat'}:return None,'chat',{}
    full=obj.get('full_chat') if isinstance(obj.get('full_chat'),dict) else {}
    kind=str(full.get('_',''))
    identity=channel_id(obj.get('id')) or channel_id(full.get('id')) or row_id
    if obj.get('id') is not None and row_id is not None and channel_id(obj['id'])!=row_id:
        return None,'row_object_id_mismatch',{}
    main=obj
    if isinstance(obj.get('chats'),list) and identity is not None:
        matches=[x for x in obj['chats'] if isinstance(x,dict) and channel_id(x.get('id'))==identity and x.get('_') in {'Channel','ChannelForbidden'}]
        if len(matches)==1:main=matches[0];tag=str(main.get('_'))
    if tag in {'Channel','ChannelForbidden'}:entity_kind='channel'
    elif tag=='ChannelFull' or kind=='ChannelFull':entity_kind='channel_full'
    elif not tag and obj.get('title'):entity_kind='channel_inferred_from_title'
    else:return None,'unrecognized_or_nonchannel',{}
    if identity is None:return None,'missing_channel_id',{}
    # All lookups are in the identified channel or its full_chat, never arbitrary nested users.
    details=full or (obj if tag=='ChannelFull' else {})
    def first(key):
        if key in main:return main[key]
        return details.get(key)
    title=first('title') or ''
    username=first('username') or ''
    about=details.get('about') or main.get('about') or main.get('description') or details.get('description') or ''
    participants=pd.to_numeric(first('participants_count'),errors='coerce')
    if pd.notna(participants) and participants<0:participants=np.nan
    creation=pd.NaT;creation_field=''
    for scope,record in [('channel',main),('full_chat',details)]:
        for field in ['creation_date','created_at','created_date','date_created']:
            parsed=parse_date(record.get(field))
            if pd.notna(parsed):creation=parsed;creation_field=f'{scope}.{field}';break
        if creation_field:break
    row={'channel_id':identity,'entity_kind':entity_kind,'title':str(title),'username':str(username),
         'about':str(about),'participants_count':participants,'metadata_date':parse_date(main.get('date')),
         'recorded_creation_date':creation,'creation_field':creation_field,'metadata_ambiguous':False}
    for key in FLAGS:row[key]=known_bool(first(key))
    return row,entity_kind,main


def load_metadata(path):
    counts=Counter();fields=Counter();entries={};ambiguous=set()
    for raw in read_dicts(path):
        counts['source_rows']+=1
        if counts['source_rows']%100_000==0:print(f'Source metadata rows: {counts["source_rows"]:,}',flush=True)
        rawmap={canonical(k):v for k,v in raw.items() if k is not None}
        rid=channel_id(rawmap.get('id',rawmap.get('channel_id')))
        try:
            if 'data' in rawmap:obj=json.loads(rawmap['data'])
            else:obj=rawmap
        except (TypeError,ValueError):counts['invalid_json']+=1;continue
        row,kind,main=extract_channel(obj,rid);counts[kind]+=1
        if row is None:continue
        fields.update(main.keys())
        ident=row['channel_id']
        if ident in entries:
            counts['duplicate_channel_rows']+=1
            # Compare normalized exports, including unknown bools and timestamps.
            previous=entries[ident]
            signature=lambda x:json.dumps({k:None if pd.isna(v) else str(v) for k,v in x.items()},sort_keys=True)
            if signature(previous)!=signature(row):ambiguous.add(ident)
        else:entries[ident]=row
    for ident in ambiguous:
        row=entries[ident];row['metadata_ambiguous']=True
        for flag in FLAGS:row[flag]=pd.NA
        row['participants_count']=np.nan;row['about']='';row['recorded_creation_date']=pd.NaT;row['creation_field']=''
    counts['ambiguous_duplicate_channels']=len(ambiguous)
    frame=pd.DataFrame(entries.values(),columns=META)
    for key in FLAGS:frame[key]=frame[key].astype('boolean')
    return frame,counts,fields


def hour_number(column):
    s = str(column).strip().lower()

    # TeleScope: "0-1 hr", "7-8hr", ..., "23-24 hr"
    match = re.fullmatch(
        r"(\d{1,2})\s*[-–—]\s*(\d{1,2})\s*"
        r"(?:h|hr|hrs|hour|hours)",
        s,
    )
    if match:
        start, end = map(int, match.groups())
        if 0 <= start <= 23 and end == start + 1:
            return start
        return None

    # Các định dạng cũ: "0", "00:00", "hour_0"
    match = re.fullmatch(
        r"(?:hour[ _-]*)?(\d{1,2})(?::00(?::00)?)?",
        s,
    )
    if match:
        hour = int(match.group(1))
        if 0 <= hour <= 23:
            return hour

    return None


def load_temporal(path,hour_columns=None,creation_column=None,min_sum=0):
    d=pd.read_csv(path,sep=None,engine='python',dtype=str)
    normalized={canonical(c):c for c in d}
    idcol=next((normalized[x] for x in ['channel_id','id','file_names','file_name','filename','file'] if x in normalized),d.columns[0])
    if hour_columns:
        hours=hour_columns.split(',')
        if len(hours)!=24 or len(set(hours))!=24 or set(hours)-set(d):
            raise ValueError('--hour-columns must list exactly 24 distinct existing columns in hour 0..23 order')
    else:
        slots=defaultdict(list)
        for c in d:
            h=hour_number(c)
            if h is not None:slots[h].append(c)
        if set(slots)!=set(range(24)) or any(len(v)!=1 for v in slots.values()):
            raise ValueError('Cannot unambiguously identify 24 hour columns. Use --hour-columns with 24 actual names in order. Found: '+str(list(d.columns)))
        hours=[slots[i][0] for i in range(24)]
    if idcol in hours:raise ValueError('ID column also matched an hour column')
    ids=d[idcol].map(channel_id)
    values=d[hours].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
    totals=values.sum(axis=1)
    valid=ids.notna().to_numpy() & np.isfinite(values).all(axis=1) & (values>=0).all(axis=1) & (totals>min_sum)
    candidates={};conflicts=set();creation={}
    if creation_column:
        if creation_column not in d:raise ValueError('--creation-column is absent from temporal file')
        cc=creation_column
    else:cc=next((normalized[x] for x in ['creation_date','created_at','created_date','date_created','channel_creation_date'] if x in normalized),None)
    for i in range(len(d)):
        if pd.isna(ids.iloc[i]):continue
        ch=int(ids.iloc[i])
        if cc:
            date=parse_date(d.iloc[i][cc])
            if pd.notna(date):
                if ch in creation and creation[ch]!=date:creation[ch]=pd.NaT
                else:creation[ch]=date
        if not valid[i]:continue
        if ch in candidates and not np.array_equal(candidates[ch],values[i]):conflicts.add(ch)
        else:candidates[ch]=values[i]
    for ch in conflicts:candidates.pop(ch,None)
    profiles={ch:raw/raw.sum() for ch,raw in candidates.items()}
    rows=[]
    for ch,p in profiles.items():
        positive=p[p>0]
        rows.append({'channel_id':ch,'hour_bin_sum':float(candidates[ch].sum()),
                     'hour_entropy':float(-(positive*np.log2(positive)).sum()/np.log2(24)),
                     'peak_hour':int(np.argmax(p)),'peak_hour_share':float(p.max()),
                     'active_hour_bins':int((p>0).sum()),'share_hours_00_05':float(p[:6].sum()),
                     'hourly_cv':float(p.std()/p.mean()),'recorded_creation_date_temporal':creation.get(ch,pd.NaT)})
    cols=['channel_id','hour_bin_sum','hour_entropy','peak_hour','peak_hour_share','active_hour_bins','share_hours_00_05','hourly_cv','recorded_creation_date_temporal']
    info={'temporal_rows':len(d),'valid_unique_hour_profiles':len(profiles),'invalid_or_below_min_rows':int((~valid).sum()),
          'conflicting_hour_profiles_excluded':len(conflicts),'id_column':idcol,'hour_columns':hours,'creation_column':cc,
          'profile_scale':'normalized input bins; bin sum is not assumed to be lifetime message count'}
    return pd.DataFrame(rows,columns=cols),profiles,creation,info


def load_languages(path):
    if not path or not path.exists():return {},'language file unavailable'
    d=pd.read_csv(path,sep=None,engine='python',dtype=str)
    names={canonical(c):c for c in d}
    idcol=next((names[x] for x in ['channel_id','file_names','file_name','filename','id'] if x in names),None)
    langcol=next((names[x] for x in ['detected_language','language','lang','language_code'] if x in names),None)
    if idcol is None or langcol is None:return {},'language columns unrecognized; no language-matched references'
    possibilities=defaultdict(set)
    for ident,lang in zip(d[idcol],d[langcol]):
        ch=channel_id(ident);l=str(lang).strip().lower()
        if ch is not None and l not in NULL_LANG:possibilities[ch].add(l)
    return {ch:next(iter(v)) for ch,v in possibilities.items() if len(v)==1},'observed channel-language labels only; no inference from user lang_code'


def normalize_url(raw):
    raw=raw.rstrip('.,;:!?)、。')
    if not re.match(r'^https?://',raw,re.I):raw='https://'+raw
    try:
        p=urlsplit(raw)
        if not p.hostname or p.username or p.password:return None,None,None
        host=p.hostname.lower().encode('idna').decode('ascii').removeprefix('www.')
        # A hostname is not a registrable-domain/ownership determination.
        port=f':{p.port}' if p.port not in (None,80,443) else ''
        path=p.path or '/'
        query=urlencode(sorted((k,v) for k,v in parse_qsl(p.query,keep_blank_values=True) if not k.lower().startswith('utm_') and k.lower() not in {'fbclid','gclid'}))
        url=urlunsplit(('https',host+port,path,query,''))
        target=None
        if host in {'t.me','telegram.me'}:
            pieces=[x for x in path.split('/') if x]
            if pieces and pieces[0]=='s':pieces=pieces[1:]
            if pieces and re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{4,31}',pieces[0]) and pieces[0].lower() not in {'joinchat','addstickers','addemoji','proxy','share','login','confirmphone','contact'}:
                target=pieces[0].lower()
        return url,host,target
    except (ValueError,UnicodeError):return None,None,None


def extract_tokens(text):
    tokens=set()
    for raw in LINK_PATTERN.findall(text or ''):
        url,host,target=normalize_url(raw)
        if url:tokens.add(('url',url))
        if host and host not in {'t.me','telegram.me','telegram.org'}:tokens.add(('web_host',host))
        if target:tokens.add(('telegram_target',target))
    for mention in MENTION.findall(text or ''):tokens.add(('telegram_target',mention.lower()))
    return tokens


def read_pairs(path):
    p=pd.read_csv(path)
    if not {'channel_a','channel_b'}<=set(p):raise ValueError('Candidate CSV must contain channel_a/channel_b')
    for c in ['channel_a','channel_b']:
        raw=p[c].map(channel_id)
        if raw.isna().any():raise ValueError('Unrecognized candidate channel IDs')
        p[c]=raw.astype('int64')
    if p.duplicated(['channel_a','channel_b']).any() or p.channel_a.ge(p.channel_b).any():
        raise ValueError('Candidate pairs must be distinct and canonical (a < b)')
    if p.empty:raise ValueError('Candidate list is empty')
    # Avoid carrying stale metadata enrichment from another run.
    return p


def match_references(channels,targets,seed,bins):
    c=channels.copy()
    c['activity_stratum']=pd.qcut(np.log1p(c.hour_bin_sum),q=bins,labels=False,duplicates='drop')
    # All identical activity values produce all-NaN qcut; give them a single stratum.
    if c.activity_stratum.isna().all() and len(c):c['activity_stratum']=0
    usable=c[c.language.notna() & c.activity_stratum.notna() & ~c.metadata_ambiguous].copy()
    groups={k:g.channel_id.astype(int).tolist() for k,g in usable[~usable.channel_id.isin(targets)].groupby(['language','activity_stratum'])}
    rng=np.random.default_rng(seed)
    for pool in groups.values():rng.shuffle(pool)
    by=usable.set_index('channel_id'); rows=[]
    ordering=sorted(targets);rng.shuffle(ordering)
    for ch in ordering:
        if ch not in by.index:
            rows.append((ch,pd.NA,None,None,'missing_profile_language_or_unambiguous_metadata'));continue
        row=by.loc[ch];key=(row.language,row.activity_stratum);pool=groups.get(key,[])
        if not pool:rows.append((ch,pd.NA,row.language,int(row.activity_stratum),'no_unused_reference_in_stratum'));continue
        reference=pool.pop()
        rows.append((ch,reference,row.language,int(row.activity_stratum),'matched'))
    frame=pd.DataFrame(rows,columns=['candidate_channel','reference_channel','language','activity_stratum','match_status'])
    frame['reference_channel']=frame.reference_channel.astype('Int64')
    return frame


def similarity(x,y):
    cosine=float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))
    mean=(x+y)/2
    def kl(a):
        mask=a>0
        return float(np.sum(a[mask]*np.log2(a[mask]/mean[mask])))
    js=float(np.sqrt(max(0.,.5*(kl(x)+kl(y)))))
    return cosine,js


def pair_analysis(pairs,metadata,profiles,creation,tokens,matching):
    m=metadata.set_index('channel_id');ref={int(r.candidate_channel):int(r.reference_channel) for r in matching.itertuples(index=False) if pd.notna(r.reference_channel)}
    rows=[];shared=[]
    for i,p in enumerate(pairs.itertuples(index=False)):
        a,b=int(p.channel_a),int(p.channel_b)
        row={'channel_a':a,'channel_b':b,'has_source_metadata_a':a in m.index,'has_source_metadata_b':b in m.index,
             'has_hour_profile_a':a in profiles,'has_hour_profile_b':b in profiles}
        if a in profiles and b in profiles:
            row['hour_cosine'],row['hour_js_distance']=similarity(profiles[a],profiles[b])
            delta=abs(int(np.argmax(profiles[a]))-int(np.argmax(profiles[b])))
            row['peak_hour_circular_gap']=min(delta,24-delta)
        else:row.update(hour_cosine=np.nan,hour_js_distance=np.nan,peak_hour_circular_gap=np.nan)
        ra,rb=ref.get(a),ref.get(b);row.update(reference_a=ra,reference_b=rb)
        if ra in profiles and rb in profiles:
            row['reference_hour_cosine'],row['reference_hour_js_distance']=similarity(profiles[ra],profiles[rb])
        else:row.update(reference_hour_cosine=np.nan,reference_hour_js_distance=np.nan)
        ca,cb=creation.get(a,pd.NaT),creation.get(b,pd.NaT)
        row['recorded_creation_gap_days']=abs((ca-cb).total_seconds())/86400 if pd.notna(ca) and pd.notna(cb) else np.nan
        for flag in FLAGS:
            values=[m.at[ch,flag] for ch in [a,b] if ch in m.index and pd.notna(m.at[ch,flag])]
            row[f'{flag}_known_endpoints']=len(values)
            row[f'{flag}_positive_endpoints']=sum(bool(v) for v in values)
        common=tokens.get(a,set()) & tokens.get(b,set())
        for kind in ['url','web_host','telegram_target']:row[f'n_shared_{kind}']=sum(t==kind for t,_ in common)
        for kind,value in sorted(common):shared.append((a,b,kind,value))
        rows.append(row)
        if (i+1)%10000==0:print(f'Metadata pairs: {i+1:,}/{len(pairs):,}',flush=True)
    features=pd.DataFrame(rows)
    # Candidate detection columns retained, metadata columns originate only from this run.
    p=pairs.drop(columns=[c for c in features if c not in ['channel_a','channel_b']],errors='ignore').merge(features,on=['channel_a','channel_b'],validate='one_to_one')
    return p,pd.DataFrame(shared,columns=['channel_a','channel_b','token_type','token_value'])


def flag_comparison(metadata,matching):
    available=matching[matching.reference_channel.notna()]
    indexed=metadata.set_index('channel_id');rows=[]
    for flag in FLAGS:
        ids=available.candidate_channel.astype(int).tolist();refs=available.reference_channel.astype(int).tolist()
        a=indexed.reindex(ids)[flag];b=indexed.reindex(refs)[flag]
        # Paired-complete comparison: missing flags are not false.
        known=a.notna().to_numpy() & b.notna().to_numpy()
        av=a.to_numpy(dtype=object)[known];bv=b.to_numpy(dtype=object)[known]
        n=int(known.sum());pa=sum(bool(x) for x in av);pb=sum(bool(x) for x in bv)
        rows.append({'flag':flag,'matched_channels':len(available),'paired_known_channels':n,
                     'candidate_positive':pa,'reference_positive':pb,
                     'candidate_rate':pa/n if n else np.nan,'reference_rate':pb/n if n else np.nan,
                     'rate_ratio':pa/pb if pb else np.nan})
    return pd.DataFrame(rows)


def write_table(out,name,frame,description):
    folder=out/name;folder.mkdir(parents=True,exist_ok=True)
    frame.to_csv(folder/f'{name}.csv',index=False,encoding='utf-8-sig')
    (folder/'README.md').write_text(f'# {name}\n\n{description}\n\nRows: {len(frame):,}.\n',encoding='utf-8')


def plots(figdir,channels,pairs,flags,shared,profiles):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir.mkdir(parents=True,exist_ok=True)
    names=['metadata_flag_comparison','metadata_hour_similarity','metadata_hour_heatmap','metadata_entropy_peak','metadata_shared_links','metadata_creation_gap']
    for name in names:
        old=figdir/f'{name}.png'
        if old.exists():old.unlink()
    def save(fig,name):
        fig.tight_layout();fig.savefig(figdir/f'{name}.png',dpi=160);plt.close(fig)
    known=flags[flags.paired_known_channels.gt(0)]
    if not known.empty:
        fig,ax=plt.subplots(figsize=(9,5));x=np.arange(len(known))
        ax.bar(x-.18,known.candidate_rate*100,width=.36,label='Candidate endpoints')
        ax.bar(x+.18,known.reference_rate*100,width=.36,label='Matched unselected references')
        ax.set_xticks(x,[f'{r.flag}\npaired n={r.paired_known_channels}' for r in known.itertuples(index=False)])
        ax.set(ylabel='Platform flag positive (%)',title='Metadata flags: paired-complete descriptive comparison');ax.legend(fontsize=8);save(fig,'metadata_flag_comparison')
    comparable=pairs.dropna(subset=['hour_cosine','reference_hour_cosine'])
    if not comparable.empty:
        fig,ax=plt.subplots(figsize=(9,5));bins=np.linspace(0,1,31)
        ax.hist(comparable.hour_cosine,bins=bins,alpha=.55,label='Candidate pairs')
        ax.hist(comparable.reference_hour_cosine,bins=bins,alpha=.55,label='Matched endpoint-reference pairs')
        ax.set(xlabel='Cosine similarity of 24-bin activity profiles',ylabel='Pairs',title='Activity-profile similarity (dependent pairs, descriptive)');ax.legend(fontsize=8);save(fig,'metadata_hour_similarity')
    targets=channels[channels.cohort.eq('candidate')]
    chosen=targets[targets.channel_id.isin(profiles)].sort_values(['candidate_degree','channel_id'],ascending=[False,True]).head(16).channel_id.astype(int).tolist()
    if chosen:
        fig,ax=plt.subplots(figsize=(11,max(4,len(chosen)*.28)));im=ax.imshow(np.stack([profiles[ch] for ch in chosen]),aspect='auto',cmap='viridis')
        ax.set_xticks(range(24));ax.set_yticks(range(len(chosen)),list(map(str,chosen)),fontsize=7)
        ax.set(xlabel='Hour as recorded in temporal metadata',ylabel='Channel ID',title='Normalized 24-hour profiles: highest candidate-degree endpoints')
        fig.colorbar(im,ax=ax,label='Fraction of input-bin activity');save(fig,'metadata_hour_heatmap')
    if not channels.empty and 'hour_entropy' in channels:
        fig,ax=plt.subplots(figsize=(9,5))
        for group,label in [('candidate','Candidate endpoints'),('reference','Matched references')]:
            g=channels[channels.cohort.eq(group)].dropna(subset=['hour_entropy','peak_hour_share'])
            if len(g)>3000:g=g.sample(3000,random_state=42)
            ax.scatter(g.hour_entropy,g.peak_hour_share,s=8,alpha=.3,label=label)
        ax.set(xlabel='Normalized hourly entropy',ylabel='Activity share in peak hour',title='Hourly concentration: not an automation classifier');ax.legend(fontsize=8);save(fig,'metadata_entropy_peak')
    if not shared.empty:
        g=shared.groupby(['token_type','token_value']).size().sort_values(ascending=False).head(12)
        fig,ax=plt.subplots(figsize=(11,6));labels=[f'{k}: {v}'[:85] for k,v in g.index]
        ax.barh(labels[::-1],g.to_numpy()[::-1]);ax.set(xlabel='Candidate pairs sharing this profile token',title='Shared links/targets in channel descriptions (not post entities)');save(fig,'metadata_shared_links')
    gap=pairs.recorded_creation_gap_days.dropna()
    if len(gap):
        fig,ax=plt.subplots(figsize=(9,5));ax.hist(gap,bins=30)
        ax.set(xlabel='Absolute recorded-creation-date gap (days)',ylabel='Pairs with dates available',title='Recorded creation dates: descriptive context');save(fig,'metadata_creation_gap')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pairs',type=Path)
    ap.add_argument('--metadata',type=Path,default=Path('telescope_files/Source_Channel_Metadata.csv'))
    ap.add_argument('--temporal',type=Path,default=Path('telescope_files/Temporal_Channel_Metadata.csv'))
    ap.add_argument('--languages',type=Path,default=Path('telescope_files/Channel_Language.csv'))
    ap.add_argument('--out',type=Path,default=Path('bigdata_pipeline/analysis/suspicious_metadata'))
    ap.add_argument('--figdir',type=Path,default=Path('bigdata_pipeline/analysis/figures/suspicious_metadata'))
    ap.add_argument('--hour-columns',default=None,help='24 comma-separated actual names in hour 0..23 order')
    ap.add_argument('--creation-column',default=None,help='Explicit recorded-creation field in temporal CSV; generic date not assumed creation')
    ap.add_argument('--min-hour-sum',type=float,default=0.)
    ap.add_argument('--activity-bins',type=int,default=5)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--no-plots',action='store_true')
    a=ap.parse_args()
    if a.activity_bins<1 or a.min_hour_sum<0:ap.error('activity-bins >=1 and min-hour-sum >=0 required')
    if a.pairs is None:
        a.pairs=next((x for x in [Path('bigdata_pipeline/analysis/suspicious_deep/pair_profiles/pair_profiles.csv'),Path('bigdata_pipeline/analysis/suspicious/suspicious_pairs.csv')] if x.exists()),None)
        if a.pairs is None:ap.error('No candidate CSV found. Supply --pairs PATH.')
    for attr in ['metadata','temporal','languages']:setattr(a,attr,resolve_file(getattr(a,attr)))
    for path in [a.pairs,a.metadata,a.temporal]:
        if not path.is_file():ap.error(f'File not found: {path}')
    if a.out.resolve()==a.pairs.parent.resolve():ap.error('Choose a different output directory from the candidate CSV')
    print('Reading metadata (channel constructors only; user/bot objects excluded)...',flush=True)
    metadata,kinds,rawfields=load_metadata(a.metadata)
    if metadata.empty:raise SystemExit('No recognizable channel metadata rows. Check exported schema; see parser supported constructors.')
    temporal,profiles,temporal_creation,timeinfo=load_temporal(a.temporal,a.hour_columns,a.creation_column,a.min_hour_sum)
    languages,langscope=load_languages(a.languages)
    pairs=read_pairs(a.pairs);targets=set(pairs.channel_a)|set(pairs.channel_b)
    endpoints=Counter(pd.concat([pairs.channel_a,pairs.channel_b]).astype(int))
    channels=metadata.merge(temporal,on='channel_id',how='left',validate='one_to_one')
    channels['language']=channels.channel_id.map(languages)
    # Require channel metadata to establish namespace when using temporal IDs.
    metadata_ids=set(metadata.channel_id)
    profiles={ch:v for ch,v in profiles.items() if ch in metadata_ids}
    # Temporal creation only fills explicit creation values; generic Channel.date stays separate.
    creation={};creation_status={};creation_origin={}
    for row in channels.itertuples(index=False):
        mdate=row.recorded_creation_date;tdate=temporal_creation.get(int(row.channel_id),pd.NaT)
        ch=int(row.channel_id)
        if pd.notna(mdate) and pd.notna(tdate) and mdate!=tdate:
            creation_status[ch]='conflicting_explicit_dates';creation_origin[ch]='';continue
        date=mdate if pd.notna(mdate) else tdate
        if pd.notna(date):
            creation[ch]=date
            creation_status[ch]='both_agree' if pd.notna(mdate) and pd.notna(tdate) else ('source_metadata' if pd.notna(mdate) else 'temporal_metadata')
            creation_origin[ch]=row.creation_field if pd.notna(mdate) else f'temporal.{timeinfo["creation_column"]}'
        else:creation_status[ch]='unavailable';creation_origin[ch]=''
    channels['recorded_creation_date']=channels.channel_id.map(creation)
    channels['creation_status']=channels.channel_id.map(creation_status)
    channels['creation_field_final']=channels.channel_id.map(creation_origin)
    channels['candidate_degree']=channels.channel_id.map(endpoints).fillna(0).astype(int)
    eligible=channels[channels.channel_id.isin(profiles)].copy()
    matching=match_references(eligible,targets,a.seed,a.activity_bins)
    reference_ids=set(matching.reference_channel.dropna().astype(int))
    selected=channels[channels.channel_id.isin(targets|reference_ids)].copy()
    selected['cohort']=np.where(selected.channel_id.isin(targets),'candidate','reference')
    # Raw about text remains in the input; export only selected identity and derived tokens.
    tokens={}
    for i,row in enumerate(channels.itertuples(index=False),1):
        if not row.metadata_ambiguous:tokens[int(row.channel_id)]=extract_tokens(row.about)
        if i%100_000==0:print(f'Description profiles inspected: {i:,}',flush=True)
    token_channels=defaultdict(set)
    for ch,items in tokens.items():
        for token in items:token_channels[token].add(ch)
    enriched,shared=pair_analysis(pairs,metadata,profiles,creation,tokens,matching)
    if not shared.empty:
        shared['metadata_channel_frequency']=[len(token_channels[k,v]) for k,v in zip(shared.token_type,shared.token_value)]
        shared['idf_context']=[math.log((1+len(metadata))/(1+n))+1 for n in shared.metadata_channel_frequency]
    flags=flag_comparison(metadata,matching)
    prevalence=[]
    indexed=metadata.set_index('channel_id')
    for cohort,ids in [('candidate',targets),('matched_reference',reference_ids)]:
        group=indexed.reindex(sorted(ids))
        for flag in FLAGS:
            values=group[flag];known=int(values.notna().sum());positive=int(values.fillna(False).sum())
            prevalence.append({'cohort':cohort,'flag':flag,'total_channels':len(ids),'known_channels':known,
                               'positive_channels':positive,'unknown_channels':len(ids)-known,
                               'positive_share_known':positive/known if known else np.nan})
    fieldcoverage=[]
    for field in ['title','username','about','participants_count','recorded_creation_date']+FLAGS:
        values=channels[field]
        known=values.notna() & (values.astype(str).str.strip().ne('') if field in ['title','username','about'] else True)
        fieldcoverage.append({'field':field,'channel_rows':len(channels),'known_rows':int(known.sum()),'known_share':float(known.mean())})
    summary={'input_candidate_pairs':len(pairs),'candidate_endpoint_channels':len(targets),'unique_channel_metadata_records':len(metadata),
             'candidate_endpoints_with_metadata':int(metadata.channel_id.isin(targets).sum()),
             'candidate_endpoints_with_hour_profile':len(targets&set(profiles)),
             'matched_reference_channels':len(reference_ids),'candidate_pairs_with_both_hour_profiles':int(enriched.hour_cosine.notna().sum()),
             'pairs_with_matched_reference_similarity':int(enriched.reference_hour_cosine.notna().sum()),
             'candidate_pairs_sharing_profile_url':int(enriched.n_shared_url.gt(0).sum()),
             'candidate_pairs_sharing_profile_target':int(enriched.n_shared_telegram_target.gt(0).sum()),
             'pairs_with_recorded_creation_gap':int(enriched.recorded_creation_gap_days.notna().sum())}
    tokenrows=[]
    for ch in sorted(targets|reference_ids):
        for kind,value in sorted(tokens.get(ch,set())):
            tokenrows.append((ch,'candidate' if ch in targets else 'reference',kind,value,len(token_channels[kind,value])))
    tables=[('metadata_summary',pd.DataFrame(summary.items(),columns=['metric','value']),'Counts are restricted to available metadata and supplied candidate pairs; no bot/spam labels.'),
            ('metadata_kind_audit',pd.DataFrame(kinds.items(),columns=['kind_or_measure','rows']),'Source constructor audit. User and bot records are not channel labels. Categories include processing counters, not a partition to sum.'),
            ('metadata_field_coverage',pd.DataFrame(fieldcoverage),'Missing flags are unknown, never automatically false. Creation excludes generic Channel.date.'),
            ('metadata_raw_field_counts',pd.DataFrame(rawfields.items(),columns=['field','channel_records_with_field']),'Presence in selected channel constructors, including duplicate rows; not a guarantee of populated values.'),
            ('metadata_channel_profiles',selected.drop(columns='about'),'Selected candidate/reference channels. Hour profiles are aggregated, not day-by-day traces; metadata_date is not assumed creation.'),
            ('metadata_pair_profiles',enriched,'Candidate detection columns plus platform-flag availability, activity similarity, shared description tokens and recorded-creation gaps. No abuse probability.'),
            ('metadata_reference_matching',matching,'One reference without replacement per available endpoint; same observed language and input-hour-sum quantile. References exclude all candidate endpoints but are not verified normal.'),
            ('metadata_flag_prevalence',pd.DataFrame(prevalence),'Standalone counts per cohort, including endpoints absent from metadata as unknown. Cohorts may differ in missingness; not the paired comparison or bot prevalence.'),
            ('metadata_flag_comparison',flags,'Paired-complete candidate/reference channel comparison; same denominator per flag. No missing-as-false, significance test, FPR or causal inference.'),
            ('metadata_profile_tokens',pd.DataFrame(tokenrows,columns=['channel_id','cohort','token_type','token_value','metadata_channel_frequency']),'Tokens from channel about/description only, not messages. No network requests; redirects and content safety unverified. Hostnames are not ownership/registrable-domain proof.'),
            ('metadata_shared_tokens',shared,'Shared tokens for candidate pairs; corpus frequency/IDF is descriptive context, not an abuse score.')]
    for name,frame,description in tables:write_table(a.out,name,frame,description)
    if not a.no_plots:plots(a.figdir,selected,enriched,flags,shared,profiles)
    info={'parameters':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'temporal_schema':timeinfo,
          'language_scope':langscope,'matching':'observed language + normalized-input-bin-sum quantile, one channel reference without replacement',
          'time_scope':'hour labels as supplied; no local-night interpretation without timezone confirmation',
          'limitations':['Snapshot metadata is not a historical timeline.','Pair similarities are statistically dependent through shared endpoints.',
                         'Unselected references are not established benign channels.','Flags are contextual platform annotations, not ground truth for synchronized abuse.',
                         'Original descriptions are not exported; extracted URLs/mentions may still contain public identifying information.']}
    (a.out/'run_info_metadata.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    rate=enriched.hour_cosine.median()
    rate_text=f'{rate:.3f}' if pd.notna(rate) else 'не рассчитана (нет пригодных пар)'
    text=f'''# Дополнение к разделу 5: метаданные каналов

Для {len(pairs):,} ранее отобранных пар использованы исходные метаданные каналов и почасовые профили активности. Из {len(targets):,} каналов-кандидатов {summary['candidate_endpoints_with_metadata']:,} сопоставлены с метаданными; {summary['candidate_endpoints_with_hour_profile']:,} имеют пригодный профиль 24 часов. Объекты User/ботов исключены из сопоставления с каналами по числовому ID.

Сходство почасовых профилей вычислено для {summary['candidate_pairs_with_both_hour_profiles']:,} пар; медианная косинусная близость: {rate_text}. Профили нормированы на сумму входных 24 значений. Для сравнения сопоставлено {len(reference_ids):,} каналов вне списка кандидатов по наблюдаемому языку и квантилю объёма входных почасовых значений, без повторного использования каналов. Отсутствующие соответствия и флаги отражены в таблицах; контрольные каналы не считаются заведомо нормальными.

Общая ссылка в описании найдена у {summary['candidate_pairs_sharing_profile_url']:,} пар, общий Telegram-target — у {summary['candidate_pairs_sharing_profile_target']:,}. Это ссылки в профилях, не сущности сообщений. Общие контакты могут указывать на редакционную или организационную связь, но сами по себе не доказывают её и не устанавливают фишинг.

Сопоставление scam/fake/restricted/verified представлено с числом пар каналов, для которых оба значения известны. Эти признаки не определяют долю ложных срабатываний детектора. Почасовое сходство, возраст и ссылки используются как дополнительный контекст к временной синхронности, а не как самостоятельное доказательство ботов, спама или злонамеренной координации. generic Channel.date не интерпретируется автоматически как дата создания.
'''
    (a.out/'section_5_metadata_draft.md').write_text(text,encoding='utf-8')
    print(pd.DataFrame(summary.items(),columns=['metric','value']).to_string(index=False))
    print(f'Done. Tables: {a.out}; figures: {a.figdir}. No raw-message rerun needed.')

if __name__=='__main__':main()
