"""Short captions (fr / en / ru) for the charts and maps of themes 1 to 4.

Each entry has a one-line description (what the chart shows) and, when the tables are there, a second line
with figures and percentages computed from them. Nothing is typed by hand: if a table or a column is missing,
the second line is simply left out.

    what, facts = describe("community_purity", "fr", get_table)

`get_table(name)` returns the DataFrame of a table or None.
"""
from __future__ import annotations

import re

import pandas as pd


# ----------------------------------------------------------------- number formats
def num(x, lang: str) -> str:
    """Integer with a space as thousands separator."""
    return f"{int(round(float(x))):,}".replace(",", " ")


def dec(x, lang: str, d: int = 2) -> str:
    s = f"{float(x):.{d}f}"
    return s if lang == "en" else s.replace(".", ",")


def pct(share, lang: str, d: int = 1) -> str:
    """Share between 0 and 1 as a percentage."""
    return dec(float(share) * 100, lang, d) + " %"


def duration(seconds, lang: str) -> str:
    s = float(seconds)
    units = {"fr": ("s", "min", "h", "j"), "en": ("s", "min", "h", "d"), "ru": ("с", "мин", "ч", "сут")}[lang]
    for div, u in ((86400, units[3]), (3600, units[2]), (60, units[1])):
        if s >= div:
            return f"{dec(s / div, lang, 1)} {u}"
    return f"{dec(s, lang, 0)} {units[0]}"


def col(df: pd.DataFrame, *names: str):
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(names)


def need(T, name: str) -> pd.DataFrame:
    d = T(name)
    if d is None or d.empty:
        raise KeyError(name)
    return d


def summary_value(df: pd.DataFrame, startswith: str, key: str = "metric", value: str = "value"):
    k = col(df, key, "measure")
    row = df[df[k].astype(str).str.lower().str.startswith(startswith.lower())]
    if row.empty:
        raise KeyError(startswith)
    return row.iloc[0]


# ----------------------------------------------------------------- theme 1: communities
def f_sizes(T, L):
    d = need(T, "community_profiles_leiden")
    s = d[col(d, "size")].astype(float).sort_values(ascending=False)
    return dict(n=num(len(s), L), largest=num(s.iloc[0], L), p=pct(s.iloc[0] / s.sum(), L), med=num(s.median(), L))


def f_top20(T, L):
    d = need(T, "community_profiles_leiden").sort_values("size", ascending=False).head(20)
    allc = need(T, "community_profiles_leiden")
    out = dict(p=pct(d["size"].sum() / allc["size"].sum(), L), n=num(len(allc), L))
    if "forwards_out" in allc:
        out["pf"] = pct(d["forwards_out"].sum() / allc["forwards_out"].sum(), L)
    else:
        raise KeyError("forwards_out")
    return out


def f_inside(T, L):
    d = need(T, "community_profiles_leiden")
    return dict(p=pct(d["forwards_inside"].sum() / d["forwards_out"].sum(), L), n=num(d["forwards_out"].sum(), L))


def f_graph(T, L):
    d = need(T, "community_summary")
    r = d[d[col(d, "algorithm")].astype(str).str.lower().str.contains("leiden")]
    r = (r if len(r) else d).iloc[0]
    return dict(nodes=num(r["n_nodes"], L), edges=num(r["n_edges_directed"], L), comms=num(r["n_communities"], L),
                mod=dec(r["modularity"], L, 2))


def f_resolution(T, L):
    d = need(T, "resolution_curve")
    b = d.loc[d["modularity"].astype(float).idxmax()]
    return dict(res=dec(b["resolution"], L, 2), mod=dec(b["modularity"], L, 2), n=num(b["n_communities"], L),
                lo=num(d["n_communities"].min(), L), hi=num(d["n_communities"].max(), L))


def f_yearly(T, L):
    d = need(T, "yearly_communities").sort_values("year")
    a, b = d.iloc[0], d.iloc[-1]
    cc = col(d, "n_communities", "communities")
    return dict(y0=int(a["year"]), y1=int(b["year"]), c0=num(a[cc], L), c1=num(b[cc], L),
                m0=dec(a["modularity"], L, 2), m1=dec(b["modularity"], L, 2))


def f_stability(T, L):
    d = need(T, "yearly_communities")
    c = next(c for c in d.columns if re.match(r"nmi", str(c)))
    v = d[c].dropna().astype(float)
    return dict(mean=dec(v.mean(), L, 2), lo=dec(v.min(), L, 2), hi=dec(v.max(), L, 2))


# ----------------------------------------------------------------- theme 2: languages
def f_purity(T, L):
    d = need(T, "community_language_purity")
    w, p = d["size"].astype(float), d["purity"].astype(float)
    return dict(avg=pct((w * p).sum() / w.sum(), L), hi=pct((p >= 0.8).mean(), L), n=num(len(d), L))


def f_langdist(T, L):
    d = need(T, "language_spread").sort_values("channels", ascending=False)
    tot = d["channels"].sum()
    top = [f"{r.language_name} {pct(r.channels / tot, L, 0)}" for r in d.head(3).itertuples()]
    return dict(tot=num(tot, L), top=", ".join(top))


def f_homophily(T, L):
    d = need(T, "language_homophily")
    w = d["forwards_out"].astype(float)
    same, exp = (d["same_language_share"] * w).sum() / w.sum(), (d["expected_share"] * w).sum() / w.sum()
    return dict(same=pct(same, L), exp=pct(exp, L), ratio=dec(same / exp, L, 1))


def f_flows(T, L):
    d = need(T, "language_flows")
    d = d[d["src"] != d["dst"]].sort_values("forwards", ascending=False)
    r = d.iloc[0]
    return dict(a=r["src_name"], b=r["dst_name"], n=num(r["forwards"], L), p=pct(r["share_of_source_out"], L))


def f_bridges(T, L):
    d = need(T, "language_bridges")
    r = d.loc[d["cross_share"].astype(float).idxmax()]
    return dict(n=num(len(d), L), ch=str(int(r["channel_id"])), lang=r["language_name"], p=pct(r["cross_share"], L))


def f_activity(T, L):
    d = need(T, "language_activity")
    nm = col(d, "language_name", "language")
    y = col(d, "forwarded_per_channel")
    if "channels" in d:
        d = d[d["channels"] >= 50]
    r = d.loc[d[y].astype(float).idxmax()]
    return dict(lang=r[nm], v=dec(r[y], L, 1))


def f_inference(T, L):
    d = need(T, "inference_accuracy")
    rule = d[d["row"].astype(str).str.contains("rule", case=False)]
    r = (rule if len(rule) else d).iloc[-1]
    return dict(acc=pct(r["accuracy"], L), n=num(r["held_out_channels"], L), base=str(r["baseline_majority_language"]))


def _first_last(T, L, name, column, d=2, as_pct=False):
    t = need(T, name).sort_values("year")
    a, b = t.iloc[0], t.iloc[-1]
    f = (lambda v: pct(v, L)) if as_pct else (lambda v: dec(v, L, d))
    return dict(y0=int(a["year"]), y1=int(b["year"]), v0=f(a[column]), v1=f(b[column]))


def f_cohesion(T, L):
    return _first_last(T, L, "language_yearly", "same_language_share", as_pct=True)


def f_diversity(T, L):
    return _first_last(T, L, "language_yearly", "effective_languages", d=1)


def f_assort(T, L):
    return _first_last(T, L, "language_yearly", "assortativity")


def f_shares(T, L):
    t = need(T, "language_yearly").sort_values("year")
    a, b = t.iloc[0], t.iloc[-1]
    return dict(y0=int(a["year"]), y1=int(b["year"]), l0=a["top_language"], l1=b["top_language"],
                v0=pct(a["top_language_share"], L), v1=pct(b["top_language_share"], L))


# ----------------------------------------------------------------- theme 3: tracing
def f_chain(T, L):
    d = need(T, "chain_size_distribution")
    a, b = d.iloc[0], d.iloc[-1]
    return dict(c0=str(a["size_class"]), p0=pct(a["share_of_posts"], L), c1=str(b["size_class"]),
                p1=pct(b["share_of_relays"], L), n=num(d["original_posts"].sum(), L))


def f_delay(T, L):
    d = need(T, "delay_distribution")
    a = d.iloc[0]
    m = d.loc[d["share_of_relays"].astype(float).idxmax()]
    return dict(c0=str(a["delay_class"]), p0=pct(a["share_of_first_relays"], L), cm=str(m["delay_class"]),
                pm=pct(m["share_of_relays"], L))


def f_delay_year(T, L):
    d = need(T, "delay_by_year").sort_values("year")
    a, b = d.iloc[0], d.iloc[-1]
    return dict(y0=int(a["year"]), y1=int(b["year"]), m0=duration(a["median_delay_s"], L),
                m1=duration(b["median_delay_s"], L), d1=pct(b["share_within_1day"], L))


def f_concentration(T, L):
    d = need(T, "source_concentration").sort_values("top_share_of_sources")
    f = d["top_share_of_sources"].astype(float)

    def near(x):
        return d.loc[(f - x).abs().idxmin()]
    a, b = near(0.01), near(0.10)
    return dict(f0=pct(a["top_share_of_sources"], L, 0), r0=pct(a["share_of_relays"], L),
                f1=pct(b["top_share_of_sources"], L, 0), r1=pct(b["share_of_relays"], L))


def f_verif(T, L):
    d = need(T, "source_verification").head(3)
    return dict(txt="; ".join(f"{r.measure}: {pct(r.share_of_posts, L)}" for r in d.itertuples()))


def f_topsources(T, L):
    d = need(T, "top_sources")
    c = col(d, "share_of_relays")
    d = d.sort_values(c, ascending=False)
    return dict(p1=pct(d[c].iloc[0], L), ch=str(int(d.iloc[0][col(d, "origin_channel", "channel_id")])),
                p10=pct(d[c].head(10).sum(), L))


def f_levels(T, L):
    d = need(T, "traceability_levels")
    return dict(txt=", ".join(f"{r.level} {pct(r.share_of_forwards, L)}" for r in d.itertuples()))


# ----------------------------------------------------------------- theme 4: cities and anomalies
def _geo(T, key):
    return summary_value(need(T, "geo_summary"), key)


def f_geo_cov(T, L):
    t = _geo(T, "channels with a city")
    share = next(c for c in t.index if str(c).startswith("share_of"))
    return dict(n=num(t["value"], L), p=pct(t[share], L), nm=num(_geo(T, "metadata")["value"], L))


def f_methods(T, L):
    tot = float(_geo(T, "channels with a city")["value"])
    nb = float(_geo(T, "city from neighbours")["value"])
    return dict(nb=num(nb, L), pnb=pct(nb / tot, L), di=num(tot - nb, L), pdi=pct((tot - nb) / tot, L))


def f_topcities(T, L):
    d = need(T, "city_totals").sort_values("n_channels", ascending=False)
    tot = d["n_channels"].sum()
    top = ", ".join(f"{r.city} {num(r.n_channels, L)} ({pct(r.n_channels / tot, L, 0)})" for r in d.head(4).itertuples())
    return dict(top=top, n=num(len(d), L))


def f_sync_days(T, L):
    s = need(T, "synchronized_peaks")
    b = s.loc[s["n_cities"].idxmax()]
    return dict(n=num(len(s), L), c=num(b["n_cities"], L), a=str(b["start"])[:10], b=str(b["end"])[:10],
                g=num(int(s["global"].sum()), L) if "global" in s else "0")


def f_series(T, L):
    d = need(T, "anomaly_summary")

    def v(k):
        return summary_value(d, k)["value"]
    return dict(c=num(v("cities analysed"), L), days=num(v("days"), L), cd=num(v("anomalous city-days"), L),
                ep=num(v("episodes"), L))


def _shape_shares(T, L):
    e = need(T, "anomaly_episodes")
    sh = e["shape"].value_counts()
    return e, {k: sh.get(k, 0) / len(e) for k in ("spike", "short", "sustained")}


def f_episodes_month(T, L):
    e, s = _shape_shares(T, L)
    m = pd.to_datetime(e["start"]).dt.to_period("M").value_counts()
    return dict(n=num(len(e), L), sp=pct(s["spike"], L, 0), sh=pct(s["short"], L, 0), su=pct(s["sustained"], L, 0),
                top=str(m.index[0]), tn=num(m.iloc[0], L))


def f_map_activity(T, L):
    e = need(T, "anomaly_episodes")
    g = e.groupby("city_key").excess_posts.sum().sort_values(ascending=False)
    return dict(n=num(g.size, L), top=", ".join(f"{k} {pct(v / g.sum(), L, 0)}" for k, v in g.head(3).items()))


def f_map_corr(T, L):
    d = need(T, "city_pair_correlation")
    lag = (d["best_lag_days"] != 0).mean()
    b = d.loc[d["best_corr"].idxmax()]
    return dict(n=num(len(d), L), lag=pct(lag, L, 0), a=b["city_a"], b=b["city_b"], c=dec(b["best_corr"], L, 2))


def f_map_links(T, L):
    d = need(T, "city_pair_correlation")
    cnt = pd.concat([d.city_a, d.city_b]).value_counts()
    return dict(n=num(cnt.size, L), top=cnt.index[0], k=num(cnt.iloc[0], L))


def f_map_sync(T, L):
    s = need(T, "synchronized_peaks")
    out = dict(n=num(len(s), L))
    if "lead_stable" in s:
        out["st"] = pct(s["lead_stable"].astype(bool).mean(), L, 0)
    else:
        raise KeyError("lead_stable")
    if "onset_spread_hours" in s:
        out["sp"] = dec(s["onset_spread_hours"].median(), L, 0)
    else:
        raise KeyError("onset_spread_hours")
    return out


# ----------------------------------------------------------------- registry
def L3(fr: str, en: str, ru: str) -> dict:
    return {"fr": fr, "en": en, "ru": ru}


REGISTRY: dict[str, dict] = {
    # ---- theme 1
    "community_size_distribution": dict(
        what=L3("Nombre de communautés par taille (échelle log) : beaucoup de petites, très peu de grandes.",
                "Number of communities by size (log scale): many small ones, very few large ones.",
                "Число сообществ по размеру (лог. шкала): много малых, очень мало крупных."),
        facts=f_sizes,
        tpl=L3("{n} communautés ; la plus grande : {largest} canaux ({p} des canaux) ; taille médiane : {med}.",
               "{n} communities; the largest has {largest} channels ({p} of channels); median size: {med}.",
               "{n} сообществ; крупнейшее: {largest} каналов ({p} всех каналов); медианный размер: {med}.")),
    "degree_distribution": dict(
        what=L3("Nombre de canaux selon leur nombre de liens : quelques canaux très connectés, la majorité très peu.",
                "Channels by number of links: a few very connected channels, most with very few.",
                "Каналы по числу связей: немного очень связанных, у большинства связей мало."),
        facts=f_graph,
        tpl=L3("Graphe : {nodes} canaux, {edges} liens orientés, {comms} communautés (modularité {mod}).",
               "Graph: {nodes} channels, {edges} directed links, {comms} communities (modularity {mod}).",
               "Граф: {nodes} каналов, {edges} направленных связей, {comms} сообществ (модулярность {mod}).")),
    "resolution_curve": dict(
        what=L3("Nombre de communautés selon la résolution de Leiden : plus elle monte, plus les communautés sont petites.",
                "Number of communities by Leiden resolution: the higher it is, the smaller the communities.",
                "Число сообществ при разной резолюции Leiden: чем она выше, тем сообщества мельче."),
        facts=f_resolution,
        tpl=L3("Meilleure modularité {mod} à la résolution {res} ({n} communautés) ; de {lo} à {hi} communautés sur la courbe.",
               "Best modularity {mod} at resolution {res} ({n} communities); from {lo} to {hi} communities on the curve.",
               "Лучшая модулярность {mod} при резолюции {res} ({n} сообществ); на кривой от {lo} до {hi} сообществ.")),
    "community_size_rank": dict(
        what=L3("Taille de chaque communauté, de la plus grande à la plus petite, une ligne par algorithme.",
                "Size of each community, from largest to smallest, one line per algorithm.",
                "Размер каждого сообщества от крупнейшего к мельчайшему, по линии на алгоритм."),
        facts=f_sizes,
        tpl=L3("La plus grande communauté regroupe {p} des canaux ; {n} communautés au total.",
               "The largest community holds {p} of the channels; {n} communities in total.",
               "Крупнейшее сообщество включает {p} каналов; всего {n} сообществ.")),
    "top_20_communities": dict(
        what=L3("Les 20 plus grandes communautés.", "The 20 largest communities.", "20 крупнейших сообществ."),
        facts=f_top20,
        tpl=L3("Elles regroupent {p} des canaux et {pf} des transferts émis (sur {n} communautés).",
               "They hold {p} of the channels and {pf} of the forwards sent (out of {n} communities).",
               "В них {p} каналов и {pf} отправленных пересылок (из {n} сообществ).")),
    "flows_between_communities": dict(
        what=L3("Où vont les transferts de chaque communauté : vers elle-même ou vers les autres.",
                "Where each community's forwards go: to itself or to the others.",
                "Куда идут пересылки каждого сообщества: внутрь себя или в другие."),
        facts=f_inside,
        tpl=L3("{p} des {n} transferts restent à l'intérieur de leur communauté.",
               "{p} of the {n} forwards stay inside their own community.",
               "{p} из {n} пересылок остаются внутри своего сообщества.")),
    "yearly_structure": dict(
        what=L3("Canaux actifs, communautés et modularité, année par année.",
                "Active channels, communities and modularity, year by year.",
                "Активные каналы, сообщества и модулярность по годам."),
        facts=f_yearly,
        tpl=L3("{y0} : {c0} communautés (modularité {m0}) ; {y1} : {c1} communautés (modularité {m1}).",
               "{y0}: {c0} communities (modularity {m0}); {y1}: {c1} communities (modularity {m1}).",
               "{y0}: {c0} сообществ (модулярность {m0}); {y1}: {c1} сообществ (модулярность {m1}).")),
    "yearly_stability": dict(
        what=L3("Stabilité des communautés d'une année à l'autre (NMI : 1 = identiques).",
                "Stability of the communities from one year to the next (NMI: 1 = identical).",
                "Устойчивость сообществ от года к году (NMI: 1 = одинаковые)."),
        facts=f_stability,
        tpl=L3("NMI moyen {mean} (de {lo} à {hi}).", "Mean NMI {mean} (from {lo} to {hi}).",
               "Средний NMI {mean} (от {lo} до {hi}).")),
    # ---- theme 2
    "community_purity": dict(
        what=L3("Part de la langue dominante dans chaque communauté : plus elle est haute, plus la communauté est monolingue.",
                "Share of the dominant language in each community: the higher, the more single-language the community.",
                "Доля преобладающего языка в сообществе: чем выше, тем сообщество одноязычнее."),
        facts=f_purity,
        tpl=L3("Pureté moyenne (pondérée par la taille) : {avg} ; {hi} des {n} communautés ont une pureté d'au moins 80 %.",
               "Average purity (size-weighted): {avg}; {hi} of the {n} communities have a purity of at least 80%.",
               "Средняя чистота (с весом по размеру): {avg}; у {hi} из {n} сообществ чистота не ниже 80 %.")),
    "language_distribution": dict(
        what=L3("Nombre de canaux par langue détectée.", "Number of channels per detected language.",
                "Число каналов по определённому языку."),
        facts=f_langdist,
        tpl=L3("{tot} canaux avec langue ; les plus nombreuses : {top}.",
               "{tot} channels with a language; the most frequent: {top}.",
               "{tot} каналов с языком; самые частые: {top}.")),
    "language_homophily": dict(
        what=L3("Part des transferts qui restent dans la même langue, comparée à ce que donnerait le hasard.",
                "Share of forwards that stay in the same language, compared with what chance would give.",
                "Доля пересылок внутри одного языка в сравнении с ожидаемой при случайности."),
        facts=f_homophily,
        tpl=L3("{same} des transferts restent dans la même langue, contre {exp} attendus au hasard (×{ratio}).",
               "{same} of forwards stay in the same language, versus {exp} expected by chance (×{ratio}).",
               "{same} пересылок остаются в том же языке против {exp} при случайности (×{ratio}).")),
    "language_flows_heatmap": dict(
        what=L3("Transferts d'une langue vers une autre ; la diagonale correspond à la même langue.",
                "Forwards from one language to another; the diagonal is the same language.",
                "Пересылки из одного языка в другой; диагональ — тот же язык."),
        facts=f_flows,
        tpl=L3("Plus grand flux entre langues : {a} → {b}, {n} transferts ({p} des sorties de {a}).",
               "Largest flow between languages: {a} → {b}, {n} forwards ({p} of {a}'s outgoing).",
               "Крупнейший поток между языками: {a} → {b}, {n} пересылок ({p} исходящих {a}).")),
    "language_bridges": dict(
        what=L3("Canaux dont les transferts passent le plus d'une langue à l'autre.",
                "Channels whose forwards cross languages the most.",
                "Каналы, чьи пересылки чаще всего переходят между языками."),
        facts=f_bridges,
        tpl=L3("{n} canaux-ponts listés ; le plus marqué : {ch} ({lang}), {p} de ses transferts traversent une langue.",
               "{n} bridge channels listed; the strongest: {ch} ({lang}), {p} of its forwards cross a language.",
               "Перечислено {n} каналов-мостов; самый выраженный: {ch} ({lang}), {p} его пересылок пересекают язык.")),
    "language_activity": dict(
        what=L3("Activité de chaque langue : combien ses canaux sont relayés par rapport à leur nombre.",
                "Activity of each language: how much its channels are forwarded relative to their number.",
                "Активность каждого языка: насколько часто пересылают его каналы относительно их числа."),
        facts=f_activity,
        tpl=L3("Langue la plus relayée par canal : {lang} ({v} fois par canal).",
               "Most forwarded language per channel: {lang} ({v} times per channel).",
               "Язык, который пересылают чаще всего на канал: {lang} ({v} раз на канал).")),
    "language_inference_accuracy": dict(
        what=L3("Exactitude de la langue déduite des canaux inconnus selon la part de vote des voisins (test sur des canaux cachés).",
                "Accuracy of the inferred language of unknown channels by neighbour vote share (tested on hidden channels).",
                "Точность выведенного языка неизвестных каналов по доле голосов соседей (проверка на скрытых каналах)."),
        facts=f_inference,
        tpl=L3("Règle retenue : {acc} de bonnes réponses sur {n} canaux cachés (référence « langue majoritaire » : {base}).",
               "Rule used: {acc} correct on {n} hidden channels (baseline 'majority language': {base}).",
               "Выбранное правило: {acc} верных на {n} скрытых каналах (база «преобладающий язык»: {base}).")),
    "language_yearly_cohesion": dict(
        what=L3("Part des transferts entre canaux de même langue, par année.",
                "Share of forwards between channels of the same language, by year.",
                "Доля пересылок между каналами одного языка по годам."),
        facts=f_cohesion,
        tpl=L3("{y0} : {v0} ; {y1} : {v1}.", "{y0}: {v0}; {y1}: {v1}.", "{y0}: {v0}; {y1}: {v1}.")),
    "language_yearly_diversity": dict(
        what=L3("Nombre effectif de langues actives, par année.", "Effective number of active languages, by year.",
                "Эффективное число активных языков по годам."),
        facts=f_diversity,
        tpl=L3("{y0} : {v0} langues effectives ; {y1} : {v1}.", "{y0}: {v0} effective languages; {y1}: {v1}.",
               "{y0}: {v0} эффективных языков; {y1}: {v1}.")),
    "language_yearly_homophily": dict(
        what=L3("Assortativité par langue (lien entre la langue de l'émetteur et celle du récepteur), par année.",
                "Assortativity by language (link between the sender's and the receiver's language), by year.",
                "Ассортативность по языку (связь языка отправителя и получателя) по годам."),
        facts=f_assort,
        tpl=L3("{y0} : {v0} ; {y1} : {v1} (1 = les canaux ne relaient que leur propre langue).",
               "{y0}: {v0}; {y1}: {v1} (1 = channels only relay their own language).",
               "{y0}: {v0}; {y1}: {v1} (1 = каналы пересылают только свой язык).")),
    "language_yearly_shares": dict(
        what=L3("Part de chaque langue parmi les canaux actifs, par année.",
                "Share of each language among active channels, by year.",
                "Доля каждого языка среди активных каналов по годам."),
        facts=f_shares,
        tpl=L3("{y0} : {l0} en tête ({v0}) ; {y1} : {l1} en tête ({v1}).",
               "{y0}: {l0} leads ({v0}); {y1}: {l1} leads ({v1}).",
               "{y0}: лидирует {l0} ({v0}); {y1}: лидирует {l1} ({v1}).")),
    # ---- theme 3
    "chain_size_distribution": dict(
        what=L3("Nombre de relais par publication d'origine : la plupart sont peu relayées.",
                "Number of relays per original post: most are relayed only a little.",
                "Число ретрансляций на исходный пост: большинство пересылают мало."),
        facts=f_chain,
        tpl=L3("{n} publications d'origine ; classe « {c0} » : {p0} des publications ; classe « {c1} » : {p1} des relais.",
               "{n} original posts; class '{c0}': {p0} of posts; class '{c1}': {p1} of relays.",
               "{n} исходных постов; класс «{c0}»: {p0} постов; класс «{c1}»: {p1} ретрансляций.")),
    "delay_distribution": dict(
        what=L3("Délai entre la publication d'origine et ses relais.",
                "Delay between the original post and its relays.",
                "Задержка между исходным постом и его ретрансляциями."),
        facts=f_delay,
        tpl=L3("{p0} des premiers relais arrivent dans la classe « {c0} » ; la classe la plus fréquente : « {cm} » ({pm} des relais).",
               "{p0} of first relays fall in class '{c0}'; the most frequent class: '{cm}' ({pm} of relays).",
               "{p0} первых ретрансляций приходятся на класс «{c0}»; самый частый класс: «{cm}» ({pm} ретрансляций).")),
    "delay_by_year": dict(
        what=L3("Rapidité des relais par année : délai médian et part relayée en moins d'un jour.",
                "Speed of relaying by year: median delay and share relayed within a day.",
                "Скорость ретрансляции по годам: медианная задержка и доля в пределах суток."),
        facts=f_delay_year,
        tpl=L3("Délai médian : {m0} en {y0}, {m1} en {y1} ; {d1} des relais de {y1} en moins d'un jour.",
               "Median delay: {m0} in {y0}, {m1} in {y1}; {d1} of {y1} relays within a day.",
               "Медианная задержка: {m0} в {y0}, {m1} в {y1}; {d1} ретрансляций {y1} в течение суток.")),
    "source_concentration": dict(
        what=L3("Part des relais détenue par les sources les plus relayées.",
                "Share of relays held by the most relayed sources.",
                "Доля ретрансляций у самых пересылаемых источников."),
        facts=f_concentration,
        tpl=L3("Les {f0} de sources les plus relayées détiennent {r0} des relais ; les {f1} : {r1}.",
               "The top {f0} of sources hold {r0} of relays; the top {f1}: {r1}.",
               "Топ {f0} источников держит {r0} ретрансляций; топ {f1}: {r1}.")),
    "source_verification": dict(
        what=L3("Part des publications et des relais dont la source est retrouvée dans le corpus.",
                "Share of posts and relays whose source is found in the corpus.",
                "Доля постов и ретрансляций, источник которых найден в корпусе."),
        facts=f_verif,
        tpl=L3("Parts des publications : {txt}.", "Share of posts: {txt}.", "Доли постов: {txt}.")),
    "top_sources": dict(
        what=L3("Les canaux sources les plus relayés.", "The most relayed source channels.",
                "Самые пересылаемые каналы-источники."),
        facts=f_topsources,
        tpl=L3("La première source (canal {ch}) détient {p1} des relais ; les 10 premières : {p10}.",
               "The top source (channel {ch}) holds {p1} of relays; the top 10: {p10}.",
               "Первый источник (канал {ch}) держит {p1} ретрансляций; первые 10: {p10}.")),
    "traceability_levels": dict(
        what=L3("Niveaux de traçabilité A/B/C/D : jusqu'où on retrouve la source d'un transfert.",
                "Traceability levels A/B/C/D: how far the source of a forward can be traced.",
                "Уровни прослеживаемости A/B/C/D: насколько удаётся найти источник пересылки."),
        facts=f_levels,
        tpl=L3("Part des transferts par niveau : {txt}.", "Share of forwards by level: {txt}.",
               "Доля пересылок по уровням: {txt}.")),
    # ---- theme 4: geolocation
    "city_coverage_by_language": dict(
        what=L3("Part des canaux de chaque langue pour lesquels une ville a été trouvée.",
                "Share of the channels of each language for which a city was found.",
                "Доля каналов каждого языка, для которых найден город."),
        facts=f_geo_cov,
        tpl=L3("{n} canaux ont une ville, soit {p} des {nm} lignes de métadonnées (profils et bots inclus).",
               "{n} channels have a city, i.e. {p} of the {nm} metadata rows (profiles and bots included).",
               "Город найден у {n} каналов, это {p} из {nm} строк метаданных (включая профили и ботов).")),
    "top_cities": dict(
        what=L3("Villes avec le plus de canaux (chaque ville une seule fois).",
                "Cities with the most channels (each city once).",
                "Города с наибольшим числом каналов (каждый город один раз)."),
        facts=f_topcities,
        tpl=L3("{n} villes ; en tête : {top}.", "{n} cities; at the top: {top}.", "{n} городов; в начале списка: {top}.")),
    "coverage_by_method": dict(
        what=L3("Canaux avec une ville, selon la méthode : texte du canal ou vote des canaux voisins.",
                "Channels with a city, by method: the channel's own text or the vote of neighbouring channels.",
                "Каналы с городом по методу: текст канала или голосование соседних каналов."),
        facts=f_methods,
        tpl=L3("Voisins : {nb} ({pnb}) ; texte du canal : {di} ({pdi}).",
               "Neighbours: {nb} ({pnb}); channel text: {di} ({pdi}).",
               "Соседи: {nb} ({pnb}); текст канала: {di} ({pdi}).")),
    # ---- theme 4: anomalies
    "cities_anomalous_per_day": dict(
        what=L3("Nombre de villes en pic chaque jour ; les zones colorées sont les pics synchronisés (plusieurs villes ensemble).",
                "Number of cities in a peak each day; shaded areas are synchronized peaks (several cities together).",
                "Число городов с пиком по дням; закрашенные зоны — синхронные пики (несколько городов вместе)."),
        facts=f_sync_days,
        tpl=L3("{n} pics synchronisés ; le plus large : {c} villes, du {a} au {b} ; {g} marqués « globaux ».",
               "{n} synchronized peaks; the widest: {c} cities, from {a} to {b}; {g} flagged 'global'.",
               "{n} синхронных пиков; самый широкий: {c} городов, с {a} по {b}; «глобальных»: {g}.")),
    "top_cities_series": dict(
        what=L3("Messages par jour des 6 villes les plus actives ; points rouges = jours en pic.",
                "Daily posts of the 6 most active cities; red dots = peak days.",
                "Сообщений в день в 6 самых активных городах; красные точки — дни пиков."),
        facts=f_series,
        tpl=L3("{c} villes sur {days} jours ; {cd} jours-villes en pic, {ep} épisodes.",
               "{c} cities over {days} days; {cd} city-days in a peak, {ep} episodes.",
               "{c} городов за {days} дней; {cd} городо-дней с пиком, {ep} эпизодов.")),
    "episodes_per_month": dict(
        what=L3("Épisodes de pic commencés chaque mois : brefs (1-2 jours), courts (3-7) ou prolongés (plus de 7).",
                "Peak episodes started each month: spikes (1-2 days), short (3-7) or sustained (over 7).",
                "Эпизоды пиков по месяцам: краткие (1–2 дня), короткие (3–7) и затяжные (более 7)."),
        facts=f_episodes_month,
        tpl=L3("{n} épisodes : {sp} brefs, {sh} courts, {su} prolongés ; mois le plus chargé : {top} ({tn}).",
               "{n} episodes: {sp} spikes, {sh} short, {su} sustained; busiest month: {top} ({tn}).",
               "{n} эпизодов: {sp} кратких, {sh} коротких, {su} затяжных; самый насыщенный месяц: {top} ({tn}).")),
    # ---- theme 4: maps
    "map_anomaly_activity": dict(
        title=L3("Où sont les anomalies", "Where the anomalies are", "Где аномалии"),
        what=L3("Une bulle par ville : taille = nombre d'épisodes, couleur = messages en excès (échelle log). Survol : détail.",
                "One bubble per city: size = number of episodes, colour = excess posts (log scale). Hover for details.",
                "Пузырь на город: размер — число эпизодов, цвет — избыточные сообщения (лог. шкала). Наведите для деталей."),
        facts=f_map_activity,
        tpl=L3("{n} villes avec au moins un épisode ; part des messages en excès : {top}.",
               "{n} cities with at least one episode; share of excess posts: {top}.",
               "{n} городов с хотя бы одним эпизодом; доля избыточных сообщений: {top}.")),
    "map_city_correlation": dict(
        title=L3("Villes qui évoluent ensemble", "Cities that move together", "Города, которые меняются вместе"),
        what=L3("Lignes entre les villes les plus corrélées : orange = l'une suit l'autre de 1 à 3 jours, bleu = même jour. Menu : filtre.",
                "Lines between the most correlated cities: orange = one follows the other by 1-3 days, blue = same day. Menu: filter.",
                "Линии между самыми коррелированными городами: оранжевая — один следует за другим на 1–3 дня, синяя — в тот же день. Меню: фильтр."),
        facts=f_map_corr,
        tpl=L3("{n} paires corrélées ; {lag} avec un décalage de jours ; la plus forte : {a} – {b} ({c}).",
               "{n} correlated pairs; {lag} with a lag of days; strongest: {a} – {b} ({c}).",
               "{n} коррелированных пар; {lag} со сдвигом в днях; самая сильная: {a} – {b} ({c}).")),
    "map_city_links": dict(
        title=L3("Une ville à la fois", "One city at a time", "Один город за раз"),
        what=L3("Menu : choisissez une ville ; la carte montre ses partenaires les plus corrélés. Taille du point = force du lien.",
                "Menu: pick a city; the map shows its most correlated partners. Dot size = strength of the link.",
                "Меню: выберите город; на карте — самые коррелированные партнёры. Размер точки — сила связи."),
        facts=f_map_links,
        tpl=L3("{n} villes ont au moins un partenaire ; la plus liée : {top} ({k} partenaires).",
               "{n} cities have at least one partner; the most linked: {top} ({k} partners).",
               "{n} городов имеют хотя бы одного партнёра; больше всего связей у {top} ({k} партнёров).")),
    "map_synchronized_peaks": dict(
        title=L3("Propagation d'un pic synchronisé", "Spread of a synchronized peak", "Распространение синхронного пика"),
        what=L3("Menu : choisissez un pic. Couleur = heures après la première ville ; étoile rouge = première ville à réagir.",
                "Menu: pick a peak. Colour = hours after the first city; red star = first city to react.",
                "Меню: выберите пик. Цвет — часы после первого города; красная звезда — первый город."),
        facts=f_map_sync,
        tpl=L3("{n} pics ; première ville stable pour {st} d'entre eux ; écart médian entre première et dernière ville : {sp} h.",
               "{n} peaks; first city stable for {st} of them; median gap between first and last city: {sp} h.",
               "{n} пиков; первый город устойчив у {st}; медианный разрыв между первым и последним городом: {sp} ч.")),
}


REGISTRY.update({
    "sync_support_vs_share": dict(
        title=L3("Synchronisation répétée", "Repeated synchrony", "Повторяющаяся синхронность"),
        what=L3("Nombre de publications synchronisées et leur part parmi les publications communes. Candidats uniquement, pas une preuve d'abus.", "Synchronized-post counts and their share among shared posts. Candidates only, not proof of abuse.", "Число синхронных публикаций и их доля среди общих. Только кандидаты, не доказательство злоупотребления.")),
    "candidate_overlap": dict(
        title=L3("Recouvrement des publications", "Shared-post overlap", "Пересечение публикаций"),
        what=L3("Publications communes divisées par le nombre de publications du canal le moins actif.", "Shared posts divided by the number of posts of the less active channel.", "Общие публикации, делённые на число публикаций менее активного канала.")),
})



REGISTRY.update({
    "pair_delay_distribution": dict(title=L3("Délais des paires", "Pair delays", "Задержки пар"), what=L3("Distribution des écarts médians des paires candidates.", "Distribution of median gaps for candidate pairs.", "Распределение медианных временных разрывов пар-кандидатов.")),
    "temporal_persistence": dict(title=L3("Persistance temporelle", "Temporal persistence", "Устойчивость во времени"), what=L3("Nombre de jours et concentration sur le jour le plus actif.", "Active days versus concentration on the busiest day.", "Число дней и концентрация синхронных публикаций в самый активный день.")),
    "top_pair_sources": dict(title=L3("Sources communes", "Shared sources", "Общие источники"), what=L3("Sources principales de la première paire sélectionnée.", "Largest sources for the first selected pair.", "Основные источники первой выбранной пары.")),
    "pair_weekly_activity": dict(title=L3("Activité hebdomadaire", "Weekly activity", "Активность по неделям"), what=L3("Publications synchronisées par semaine pour cinq paires sélectionnées au maximum.", "Synchronized posts per week for up to five selected pairs.", "Синхронные публикации по неделям для максимум пяти выбранных пар.")),
    "pair_example_timeline": dict(title=L3("Exemples temporels", "Example timeline", "Временные примеры"), what=L3("Chaque ligne est une publication commune ; temps relatif au canal A.", "Each row is a shared post; times are relative to channel A.", "Каждая строка — общая публикация; время относительно канала A.")),
    "observed_vs_permuted": dict(title=L3("Observé et permutations", "Observed vs shuffled", "Наблюдение и перестановки"), what=L3("Exploration après sélection : pas un test confirmatoire d'abus.", "Exploration after selection: not a confirmatory abuse test.", "Исследовательская проверка после отбора, не подтверждающий тест злоупотребления.")),
    "interaction_context": dict(title=L3("Contexte des interactions", "Interaction context", "Контекст взаимодействий"), what=L3("Rang du compteur de transferts dans le canal source et l'année ; moment de collecte non contrôlé.", "Forward-counter rank within source channel and year; collection timing not controlled.", "Ранг счётчика пересылок внутри канала-источника и года; время сбора не контролируется.")),
})



REGISTRY.update({
    "metadata_flag_comparison": dict(title=L3("Indicateurs de la plateforme", "Platform flags", "Метки платформы"), what=L3("Comparaison descriptive des canaux candidats et des références appariées ; les valeurs manquantes ne sont pas fausses.", "Descriptive paired candidate/reference comparison; missing flags are not false.", "Описательное сравнение кандидатов и сопоставленных каналов; отсутствующие флаги не считаются ложными.")),
    "metadata_hour_similarity": dict(title=L3("Similarité horaire", "Hourly similarity", "Сходство почасовых профилей"), what=L3("Similarité cosinus des profils 24 heures pour les paires ayant aussi des références appariées.", "Cosine similarity of 24-hour profiles on pairs with matched endpoint references.", "Косинусная близость профилей 24 часов для пар с сопоставленными каналами сравнения.")),
    "metadata_hour_heatmap": dict(title=L3("Profils de 24 heures", "24-hour profiles", "Профили 24 часов"), what=L3("Profils normalisés des canaux les plus connectés dans la liste de paires candidates.", "Normalized hourly profiles of endpoints with the highest candidate-pair degree.", "Нормированные профили каналов с наибольшим числом связей в списке пар-кандидатов.")),
    "metadata_entropy_peak": dict(title=L3("Concentration horaire", "Hourly concentration", "Почасовая концентрация"), what=L3("Entropie horaire et part de l'heure maximale ; pas un classifieur de bots.", "Hourly entropy and peak-hour share; not a bot classifier.", "Почасовая энтропия и доля максимального часа; это не классификатор ботов.")),
    "metadata_shared_links": dict(title=L3("Liens communs des profils", "Shared profile links", "Общие ссылки в профилях"), what=L3("Liens et cibles communs dans les descriptions des canaux, pas dans les messages.", "Shared links/targets in channel descriptions, not in posts.", "Общие ссылки и упоминания в описаниях каналов, не в сообщениях.")),
    "metadata_creation_gap": dict(title=L3("Dates de création enregistrées", "Recorded creation dates", "Записанные даты создания"), what=L3("Écart entre dates explicitement enregistrées comme création ; date générique du canal non utilisée.", "Gaps between explicitly recorded creation dates; generic channel date is not used.", "Разрывы между явно записанными датами создания; общее поле date канала не используется.")),
})


def describe(stem: str, lang: str, get_table) -> tuple[str, str]:
    """(what the chart shows, key figures) for a chart or map file stem; empty strings when unknown."""
    e = REGISTRY.get(stem)
    if e is None:
        return "", ""
    facts = ""
    if e.get("facts"):
        try:
            facts = e["tpl"][lang].format(**e["facts"](get_table, lang))
        except Exception:                       # missing table or column: no figures, never an error
            facts = ""
    return e["what"][lang], facts


def title_of(stem: str, lang: str) -> str:
    e = REGISTRY.get(stem, {})
    return e.get("title", {}).get(lang) or stem.replace("_", " ")