"""Standalone, offline HTML view of a saved H4 search-map index.

Only presentation: no signal, Q3, price or optimization code is called.
"""
from html import escape
import json
from urllib.parse import quote

METHODS = {'FULL_PF': 'Full PF', 'TRUNCATED_PF': 'B0I', 'PR_PAULI': 'PR-Pauli', 'PR_DF': 'PR-DF'}
FORMULAS = {'2nd': 'S2', '4th': '標準Yoshida4', '4th(m5_best)': 'M5'}


def source_url(source):
    if not isinstance(source, dict) or not source.get('commit') or not source.get('path'):
        return None
    return 'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/' + quote(str(source['commit']), safe='') + '/' + quote(str(source['path']), safe='/')


def _money(value):
    if value is None:
        return '価格未接続'
    if isinstance(value, dict) and 'numerator' in value and 'denominator' in value:
        from decimal import Decimal, localcontext
        with localcontext() as context:
            context.prec = 28
            return f"{Decimal(str(value['numerator'])) / Decimal(str(value['denominator'])):,.2f} RZ"
    return str(value)


def _source(source):
    url = source_url(source)
    label = str(source.get('commit', ''))[:9] + ':' + str(source.get('path', '')) if isinstance(source, dict) else str(source)
    return ('<a href="' + escape(url, quote=True) + '" target="_blank" rel="noopener">' + escape(label) + '</a>') if url else escape(label)


def _validation_sources(index, point):
    sources = {}; models = index.get('price_validation_models') or {}
    def visit(value):
        if isinstance(value, dict):
            if source_url(value):
                sources[json.dumps(value, sort_keys=True)] = value
            else:
                for item in value.values(): visit(item)
        elif isinstance(value, list):
            for item in value: visit(item)
    for view in point.get('price_validations') or []:
        visit(models.get(view.get('model_id')) or {})
        visit(view.get('source'))
    return sources.values()


def _overview(index):
    from fractions import Fraction

    cells = next((v['cells'] for v in index.get('views', []) if v.get('mode') == 'ALL'), [])
    points = {p['id']: p for p in index.get('points', [])}
    winners = {}
    for method in METHODS:
        priced = [c for c in cells if c['method'] == method and c.get('min_cost') is not None
                  and c.get('min_point_ids')]
        if priced:
            def value(c):
                v = c['min_cost']
                return Fraction(int(v['numerator']), int(v['denominator']))
            best = min(priced, key=value)
            winners[method] = best, value(best)
    maximum = max((v for _, v in winners.values()), default=1)
    cards = []
    for method, name in METHODS.items():
        if method not in winners:
            cards.append(f'<div class="result unavailable"><span class="method">{escape(name)}</span>'
                         '<strong>価格未確定</strong></div>')
            continue
        cell, amount = winners[method]
        point = points.get(cell['min_point_ids'][0], {})
        formula = FORMULAS.get(point.get('formula'), point.get('formula'))
        conditions = list(dict.fromkeys(str(x) for x in (cell['row_label'], formula, cell['width_label'].split(' [')[0]) if x))
        cards.append('<button type="button" class="result" data-method="' + escape(method) +
                     '" data-point="' + escape(cell['min_point_ids'][0], quote=True) + '">' +
                     '<span class="method">' + escape(name) + '</span><strong>' +
                     escape(_money(cell['min_cost']).removesuffix(' RZ')) + '<small> RZ</small></strong>' +
                     '<span class="track"><i style="width:' + f'{100 * float(amount / maximum) if maximum else 0:.3f}%' +
                     '"></i></span><span class="conditions">' + escape(' · '.join(conditions)) +
                     '</span><span class="inspect">最安の配分を見る →</span></button>')
    return ''.join(cards)


def render(index):
    """Render one self-contained document; the embedded index remains exact."""
    # Escape script-closing strings without altering decoded JSON strings.
    data = json.dumps(index, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    points = index.get('points', [])
    fallback = []
    for p in points:
        refs = p.get('sources') or []
        source = next((s for s in refs if source_url(s)), None)
        fallback.append('<tr>' + ''.join('<td>' + escape(str(v)) + '</td>' for v in (
            METHODS.get(p.get('method'), p.get('method')), p.get('row_label', ''), p.get('width_label', ''),
            p.get('configuration_id', ''), p.get('q3_status') or p.get('status', ''),
            _money(p.get('conditional_cost')) + (' · ' + ' / '.join(v.get('label') or v.get('status') or '検証未完了' for v in p.get('price_validations', [])) if p.get('price_validations') else ''),
            p.get('N_X'), p.get('N_Y'))) + '<td>' + (_source(source) if source else '未取込') +
            ''.join('<br>価格検証：' + _source(value) for value in _validation_sources(index, p)) + '</td></tr>')
    return _PAGE.replace('<!--OVERVIEW-->', _overview(index)).replace('<!--FALLBACK-->', ''.join(fallback)).replace('<!--DATA-->', data)


_PAGE = r'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>H₄ RPE 探索マップ</title><style>
:root{color-scheme:light;--ink:#17263c;--muted:#52667d;--line:#d7e0ea;--paper:#fff;--blue:#185cba;--pale:#eef4fd;--good:#e7f5ed;--warn:#fff1d4;--bad:#fcecef}
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:var(--ink);font:15px/1.6 system-ui,-apple-system,"Noto Sans",sans-serif}main{max-width:1480px;margin:auto;padding:30px 26px 70px}h1{font-size:30px;line-height:1.2;margin:5px 0 12px}h2{font-size:21px;margin:0 0 14px}h3{font-size:17px;margin:15px 0 8px}p{margin:8px 0}.eyebrow{letter-spacing:.1em;color:var(--blue);font-size:12px;font-weight:700}.sub{color:var(--muted);max-width:1000px}.notice{background:var(--pale);border-left:4px solid var(--blue);padding:10px 14px;margin:18px 0}.card{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:20px;margin:16px 0;box-shadow:0 2px 8px #17263c04}.bar{display:flex;gap:12px;align-items:end;flex-wrap:wrap}.bar label{font-weight:600;font-size:13px;display:block}select,input,button{font:inherit;border:1px solid #b8c8dc;border-radius:7px;background:white;color:var(--ink);padding:9px 12px}select{max-width:100%}input{min-width:240px}button{cursor:pointer}button:focus-visible,a:focus-visible,select:focus-visible{outline:3px solid #f6b839;outline-offset:2px}.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:20px 0 14px}.tabs button.active{background:var(--blue);color:white;border-color:var(--blue)}.metrics{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:10px;margin:16px 0}.metric{background:#fff;border:1px solid var(--line);border-radius:9px;padding:12px 16px}.metric strong{display:block;font-size:25px;letter-spacing:-.02em}.metric span{font-size:12px;color:var(--muted)}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px 10px;border:1px solid var(--line);text-align:left;vertical-align:top}th{background:#f5f8fc;position:sticky;top:0;z-index:1}th:first-child{white-space:nowrap}.map{table-layout:fixed;min-width:650px}.map td{padding:0;min-width:165px}.cell{display:block;width:100%;min-height:120px;text-align:left;border:0;border-radius:0;background:white;padding:12px;line-height:1.5}.cell:hover{box-shadow:inset 0 0 0 2px var(--blue)}.cell strong{display:block;color:var(--ink);font-size:16px}.cell small{display:block;color:var(--muted);font-size:12px}.cell .price{font-weight:700;color:#07573a;font-size:14px;margin:5px 0}.cell.good{background:var(--good)}.cell.wait{background:var(--warn)}.cell.bad{background:var(--bad)}.cell.absent{background:#f5f6f8;color:#657184}.pill{display:inline-block;border-radius:4px;padding:2px 6px;font-size:11px;background:#e6ecf3;margin-bottom:4px}.legend{display:flex;gap:15px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:12px}.legend b{display:inline-block;width:10px;height:10px;border:1px solid #9eacbc;margin-right:5px}.mono,code,pre{font-family:ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere}code{font-size:12px;background:#f4f6fa;padding:2px 4px}pre{font-size:12px;white-space:pre-wrap;margin:8px 0;padding:12px;background:#f6f8fc;border-radius:7px;max-height:340px;overflow:auto}.break{overflow-wrap:anywhere}a{color:#145cb9;text-decoration:none}a:hover{text-decoration:underline}.muted{color:var(--muted)}.detail-head{display:flex;justify-content:space-between;gap:20px}.item{border-top:1px solid var(--line);padding-top:12px;margin-top:14px}details{margin:10px 0}summary{cursor:pointer;font-weight:600}.source-list li{margin:7px 0;overflow-wrap:anywhere}.empty{padding:30px;text-align:center;color:var(--muted)}.run-scope{padding:10px;background:#f5f8fc;border-radius:7px}.fallback-table td{max-width:300px;overflow-wrap:anywhere}.data-note{font-size:12px;color:var(--muted)}[hidden]{display:none!important}
@media(max-width:750px){main{padding:22px 12px 45px}.card{padding:14px}.metrics{grid-template-columns:repeat(2,1fr)}h1{font-size:25px}.detail-head{display:block}.bar>*{width:100%}input{min-width:0;width:100%}.map{min-width:600px}}

:root{--ink:#182f32;--muted:#4b6368;--blue:#126660;--pale:#e8f4f2;--line:#d5e1df;--good:#e4f3ec;--warn:#fff4d7;--bad:#f8e5df}
body{background:#f5f7f5}main{max-width:1440px;padding-top:32px}header{margin:0 0 24px}h1{font-size:clamp(26px,3.4vw,42px);letter-spacing:-.035em}
.overview{margin:0 0 28px}h2{letter-spacing:-.025em}.sub,.caption{color:var(--muted)}.overview h2{margin-bottom:2px}
.results{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:14px}.result{background:white;display:flex;flex-direction:column;align-items:stretch;text-align:left;border:1px solid var(--line);border-top:3px solid #177264;border-radius:8px;padding:15px 16px;color:var(--ink)}
.result:hover{background:#f3faf7}.result.unavailable{border-top-color:#889397}.method{font-weight:700;font-size:16px}.result strong{margin:8px 0 4px;font:650 clamp(17px,1.8vw,28px)/1.3 system-ui,sans-serif;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}.result small{font:500 11px system-ui;color:var(--muted)}.track{height:5px;background:#edf1ee;margin:5px 0 12px}.track i{display:block;height:100%;background:#308377}.conditions{font-size:13px;color:var(--muted)}.inspect{color:var(--blue);font-weight:600;margin-top:10px;font-size:12px}
.map{table-layout:auto;min-width:100%}.map td{min-width:110px}.map th:first-child{position:sticky;left:0;z-index:2}.map th{line-height:1.4}.map th small{display:block;font-size:10px;color:var(--muted);font-weight:400}.cell{min-height:86px;padding:8px}.cell strong{font-size:14px}.cell.absent{display:grid;align-content:center}.cell[aria-pressed="true"]{box-shadow:inset 0 0 0 3px var(--blue)}.cell[data-state="SOURCE_NOT_IMPORTED"]{background:#f1ebf8}.cell[data-state="PRICE_WAITING"]{background:#e8f0f9}.card{box-shadow:none}.detail-note{margin:0;color:var(--muted)}#detail:focus{outline:0}.allocation{padding:10px 0}.allocation>summary{font-variant-numeric:tabular-nums}.overview p,.section-desc{font-size:14px;margin:4px 0 12px;color:var(--muted)}.map-intro{margin-top:25px}.map-intro h2{margin-bottom:2px}.tabs button.active{background:#125c55}.legend b{vertical-align:middle}.metrics{margin:10px 0}.metric strong{font-size:21px}
@media(max-width:950px){.results{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:500px){.result{padding:10px}.results{gap:7px}.result strong{font-size:16px}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}main{padding-top:18px}}
</style></head><body><main>
<header><h1>H₄ RPE 探索マップ</h1><p class="sub">目標精度を満たした保存結果について、回路方式ごとの総RZ数と評価範囲を比較します。</p></header>
<section class="overview" aria-labelledby="comparison-title"><h2 id="comparison-title">方式別の最小総RZ数</h2><p>全収録runのうち、MSE条件を満たし、価格を照合できた配分を比較。棒が短いほど低コストです。</p><div class="results" id="overview"><!--OVERVIEW--></div></section>
<div class="map-intro"><h2>探索した範囲</h2><p class="section-desc">方式を選び、分割・公式と要求幅が交わるマスを開くと、配分、価格、元記録を確認できます。</p></div>
<section class="card"><div class="bar"><div><label for="view">表示範囲</label><select id="view"></select></div><div><label for="search">候補を絞る</label><input id="search" type="search" placeholder="configuration・公式・κ・配分"></div><button id="clear">絞り込み解除</button></div><p id="coverage" class="data-note"></p><div id="scope" class="run-scope"></div></section>
<nav id="methods" class="tabs" aria-label="方式"></nav><div id="metrics" class="metrics"></div>
<section class="card"><h2 id="map-title">探索範囲</h2><div id="grid" class="scroll"></div><div class="legend"><span><b style="background:var(--good)"></b>適合点あり</span><span><b style="background:var(--bad)"></b>評価済み・適合0件</span><span><b style="background:var(--warn)"></b>対象・未評価など</span><span><b style="background:#f1ebf8"></b>元記録未取込</span><span><b style="background:#e8f0f9"></b>適合・価格待ち</span><span><b style="background:#f5f6f8"></b>対象として未確認</span></div><p class="data-note">各マス：Q3評価済み物理構成数／評価配分数／適合配分数。最安値は接続済みの条件付き費用に限定。再利用は全run累計で1件です。「そのrunで新たに評価」は起動来歴の確認できるものだけを数えます。</p></section>
<section id="detail" tabindex="-1" class="card" aria-live="polite"><h2>マスを選ぶと元結果を確認できます</h2><p class="muted">公式・κ・実schedule・X/Y配分・価格根拠・固定commitへたどれます。</p></section>
<details class="card"><summary>収録run・比較条件・取り込み待ちの記録</summary><div id="sources"></div></details>
<details class="card"><summary>全収録点の静的表（JavaScriptなしでも閲覧可能）</summary><p class="data-note">ブラウザ内検索も使えます。詳細なJSONは <a href="INDEX.json">INDEX.json</a> を参照してください。</p><div class="scroll"><table class="fallback-table"><thead><tr><th>方式</th><th>公式/p/LD</th><th>要求幅</th><th>configuration</th><th>保存Q3判定</th><th>条件付き費用</th><th>N_X</th><th>N_Y</th><th>固定SOURCE</th></tr></thead><tbody><!--FALLBACK--></tbody></table></div></details>
<footer class="data-note">Cの比較カタログと公開探索記録を読取専用で結合。ロードマップとは別の探索coverage成果物です。</footer>
</main><script id="index-data" type="application/json"><!--DATA--></script><script>
'use strict';
const D=JSON.parse(document.getElementById('index-data').textContent), E=id=>document.getElementById(id), labels={FULL_PF:'Full PF',TRUNCATED_PF:'B0I',PR_PAULI:'PR-Pauli',PR_DF:'PR-DF'}, formulas={'2nd':'S2','4th':'標準Yoshida4','4th(m5_best)':'M5'};
const states={TARGET_UNEVALUATED:'探索対象・未評価',EVALUATED_NO_FEASIBLE:'評価済み・適合0件',SOURCE_NOT_IMPORTED:'元記録が未取込',NOT_TARGETED:'対象として未確認',MIXED:'評価済み＋未取込/未評価',FEASIBLE:'適合点あり',EVALUATED:'評価済み',PRICE_WAITING:'Q3適合・価格待ち'};
let method='FULL_PF'; const allPoints=new Map((D.points||[]).map(p=>[p.id,p])), allTargets=new Map((D.targets||[]).map(p=>[p.id,p]));
function el(tag,text,cls){let n=document.createElement(tag);if(text!==undefined&&text!==null)n.textContent=String(text);if(cls)n.className=cls;return n}
function fmt(x){if(x===undefined||x===null)return '—';if(typeof x==='object'&&'numerator'in x&&'denominator'in x){let n=Number(x.numerator)/Number(x.denominator);return Number.isFinite(n)?n.toLocaleString('ja-JP',{maximumFractionDigits:2}):String(x.numerator)+'/'+String(x.denominator)}return typeof x==='object'?JSON.stringify(x):String(x)}
function exact(x){return x&&typeof x==='object'&&'numerator'in x?String(x.numerator)+'/'+String(x.denominator):fmt(x)}
function percent(x){if(x===null||x===undefined)return 'UNKNOWN';let n=(typeof x==='object'&&'numerator'in x)?Number(x.numerator)/Number(x.denominator):Number(x);return Number.isFinite(n)?(n>=0?'+':'')+(100*n).toFixed(6)+'%':exact(x)}
function link(s){let n=el('span');if(!s||!s.commit||!s.path){n.textContent=fmt(s);return n}let a=el('a',String(s.commit).slice(0,9)+':'+s.path);a.href='https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'+encodeURIComponent(s.commit)+'/'+s.path.split('/').map(encodeURIComponent).join('/');a.target='_blank';a.rel='noopener';n.append(a);if(s.selector&&s.selector.length)n.append(el('code',JSON.stringify(s.selector)));return n}
function jsonBlock(value){return el('pre',JSON.stringify(value,null,2))}
function configuration(p){return(D.configuration_details||{})[p.configuration_detail_key]||{}}
function field(parent,label,value){let p=el('p');p.append(el('strong',label+'：'),el('span',fmt(value)));parent.append(p)}
function validationDetails(parent,p){
 for(const view of p.price_validations||[]){const model=(D.price_validation_models||{})[view.model_id]||{};let box=el('details');box.append(el('summary','価格モデル簡易検証 · '+(view.label||view.status||'検証未完了')));field(box,'モデルID',view.model_id);box.append(el('p','簡易検証は保存した実測点での診断です。適用先の全ラウンドやlaw期待値を認定せず、係数・費用・順位へ補正を加えません。','data-note'));field(box,'今回の適用点 (m,q,R)',view.application_points);let identity=el('details');identity.append(el('summary','凍結係数・SOURCE・適用条件'),jsonBlock(model.identity||model));box.append(identity);
 for(const point of model.validation_points||[]){let item=el('div',null,'item');field(item,'検証ラウンド / q / R',fmt(point.m)+' / '+fmt(point.q)+' / '+fmt(point.R));field(item,'独立経路数（X/Yは同じ経路を共有）',point.independent_paths);let table=el('table'),head=el('tr');for(const h of ['軸','事前予測 RZ/shot','実測平均 RZ/shot','符号付き相対差 (予測−実測)/実測'])head.append(el('th',h));table.append(head);for(const axis of ['X','Y']){let tr=el('tr');for(const val of [axis,fmt(point.predicted_RZ?.[axis])+' (exact '+exact(point.predicted_RZ?.[axis])+')',fmt(point.observed_mean_RZ?.[axis]),percent(point.signed_relative_difference?.[axis])+' (exact '+exact(point.signed_relative_difference?.[axis])+')'])tr.append(el('td',val));table.append(tr)}item.append(table);let observed=el('table'),oh=el('tr');for(const h of ['経路','軸','実測 RZ/shot','元receipt'])oh.append(el('th',h));observed.append(oh);for(const row of point.observations||[]){let tr=el('tr');tr.append(el('td',row.path_id),el('td',row.axis),el('td',exact(row.compiled_RZ)));let source=el('td');source.append(link(row.source));tr.append(source);observed.append(tr)}item.append(observed);if(point.source)item.append(link(point.source));const share=(view.validation_round_contributions||[]).find(v=>v.m===point.m);if(share){field(item,'この配分での当該roundのモデル総額比 (%)',share.percent_of_model_total===null?'UNKNOWN（対応する全軸価格または配分なし）':exact(share.percent_of_model_total)+' ≈ '+fmt(share.percent_of_model_total));field(item,'実測点と今回の適用点が同じ(q,R)',share.same_application_point?'はい':'いいえ（検証点と適用点を区別）')}box.append(item)}
 const refs=el('ul',null,'source-list'),seen=new Set();function collect(value){if(!value||typeof value!=='object')return;if(value.commit&&value.path){let k=JSON.stringify(value);if(!seen.has(k)){seen.add(k);let li=el('li');li.append(link(value));refs.append(li)}return}for(const child of Object.values(value))collect(child)}collect(model);collect(view.source);box.append(el('strong','較正・凍結・独立検証の固定参照'),refs);parent.append(box)}
}
function selected(){return(D.views||[]).find(v=>v.id===E('view').value)||(D.views||[])[0]||{cells:[],mode:'ALL'}}
function passes(p,q){return !q||JSON.stringify([p.configuration_id,p.formula,p.L_D,p.p,p.kappa,p.N_X,p.N_Y,p.requested_width]).toLowerCase().includes(q)}
function points(c){const q=E('search').value.trim().toLowerCase();return(c.point_ids||[]).map(id=>allPoints.get(id)).filter(Boolean).filter(p=>passes(p,q))}
function showCell(c, topPoint){let d=E('detail');d.replaceChildren();let h=el('div',null,'detail-head');h.append(el('h2',labels[method]+' · '+c.row_label+' × '+c.width_label),el('span',states[c.state]||c.state,'pill'));d.append(h);field(d,'Q3構成 / 配分 / 適合',c.physical_evaluated+' / '+c.evaluated_allocations+' / '+c.feasible_allocations);if(c.min_cost!==null&&c.min_cost!==undefined){field(d,'最安条件付き総RZ（表示用）',fmt(c.min_cost));field(d,'比較に使うexact値',exact(c.min_cost))}
 const bestIds=new Set(c.min_point_ids||[]), ps=points(c).sort((a,b)=>Number(b.id===topPoint)-Number(a.id===topPoint)||Number(bestIds.has(b.id))-Number(bestIds.has(a.id)));if(!ps.length)d.append(el('p','該当する評価点はありません。以下の対象・欠測記録を確認してください。','muted'));
 const progress=el('p',null,'data-note'),articles=el('div'),more=el('button','次の50件');let shown=0;d.append(progress,articles,more);
 function appendPage(){const end=Math.min(shown+50,ps.length);
 for(const p of ps.slice(shown,end)){let wrap=el('details',null,'item allocation');let fit=p.q3_status==='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'?'適合':p.q3_status==='MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD'?'基準超過':(p.q3_status||p.status||'判定未収録');wrap.open=p.id===topPoint;wrap.append(el('summary',(bestIds.has(p.id)?'最安配分 · ':'')+(formulas[p.formula]||p.formula||'公式未同定')+' · '+fit+' · '+(p.conditional_cost==null?'価格未接続':fmt(p.conditional_cost)+' RZ')));field(wrap,'configuration',p.configuration_id);field(wrap,'分割',p.p!==null&&p.p!==undefined?'p='+p.p:'L_D='+p.L_D);field(wrap,'要求幅',p.requested_width);field(wrap,'κ',p.kappa);field(wrap,'条件付き総RZ',p.conditional_cost===null?'未接続':exact(p.conditional_cost)+' ≈ '+fmt(p.conditional_cost));field(wrap,'N_X',p.N_X);field(wrap,'N_Y',p.N_Y);validationDetails(wrap,p);
 const cfg=configuration(p);let sections=[['実schedule・physical identity',p.schedule||cfg.schedule],['固定回路条件・law',{profile:p.profile||cfg.profile,law:p.law||cfg.law,split:p.split||cfg.split}],['保存MSE上下界・状態',{lower:p.MSE_lower,upper:p.MSE_upper,status:p.q3_status,evidence:p.evidence_level,issues:p.issues}],['価格根拠・qualification',p.price_evidence],['run別の新規評価／再利用と実行host',p.occurrences]];
 for(const [title,value]of sections){let x=el('details');x.append(el('summary',title),jsonBlock(value===undefined?null:value));wrap.append(x)}
 let refs=el('ul',null,'source-list');for(const s of [...(p.sources||[]),...(cfg.configuration_source?[cfg.configuration_source]:[])]){let li=el('li');li.append(link(s));refs.append(li)}wrap.append(el('strong','元結果への固定参照'),refs);articles.append(wrap)}
 shown=end;progress.textContent='表示 '+shown+' / '+ps.length+' 配分（一覧・件数・検索対象は全件）';more.hidden=shown>=ps.length;}
 more.onclick=appendPage;appendPage();
 const ts=(c.target_ids||[]).map(id=>allTargets.get(id)).filter(Boolean);if(ts.length){let x=el('details');x.open=!ps.length;x.append(el('summary','探索対象・選択・不足記録 ('+ts.length+')'));for(const t of ts){let item=el('div',null,'item');field(item,'状態',t.status);field(item,'構成',t.configuration_id);field(item,'不足',t.missing);if(t.source)item.append(link(t.source));x.append(item)}d.append(x)}
}
function redraw(){E('detail').replaceChildren(el('h2','評価点の詳細'),el('p','上のマスを選ぶと配分と価格の根拠が表示されます。','detail-note'));const v=selected(), q=E('search').value.trim().toLowerCase();let cells=(v.cells||[]).filter(c=>c.method===method), filtered=new Set();if(q)cells=cells.filter(c=>{let ps=points(c);ps.forEach(p=>filtered.add(p.id));return ps.length||String(c.row_label+' '+c.width_label).toLowerCase().includes(q)});
 E('map-title').textContent=labels[method]+' · '+({FULL_PF:'公式 × 要求幅',TRUNCATED_PF:'切り捨て位置p × 要求幅',PR_PAULI:'L_D × 要求幅',PR_DF:'L_D × 要求幅'}[method]);
 let pointsIds=new Set(cells.flatMap(c=>c.point_ids||[])), ps=[...pointsIds].map(id=>allPoints.get(id)).filter(Boolean).filter(p=>passes(p,q));let configs=new Set(ps.filter(p=>p.q3_status).map(p=>p.configuration_id));let metrics=[['Q3評価済み物理構成',configs.size],['評価した配分',ps.filter(p=>p.q3_status).length],['適合配分',ps.filter(p=>p.q3_status==='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD').length],['記録未取込の対象',cells.reduce((n,c)=>n+(c.missing_count||0),0)]];E('metrics').replaceChildren(...metrics.map(([l,n])=>{let e=el('div',null,'metric');e.append(el('strong',n),el('span',l));return e}));
 E('scope').replaceChildren();let run=(D.runs||[]).find(r=>r.id===v.run_id);E('scope').append(el('strong',v.label||v.id||'全run累計'));if(run){field(E('scope'),'停止・進行状態',run.status);field(E('scope'),'実行host（保存metadata）',run.hosts);if(run.source)E('scope').append(link(run.source));if(run.missing?.length||run.issues?.length)field(E('scope'),'不足',run.missing||run.issues)}else E('scope').append(el('p','同じ構成・配分・評価契約の再検出／別runでの再利用は1件。母集合全体を網羅したとの主張ではありません。','data-note'));
 const rows=new Map(), cols=new Map();for(const c of cells){rows.set(c.row_key,c.row_label);cols.set(c.width_key,c.width_label)}let host=E('grid');host.replaceChildren();if(!cells.length){host.append(el('p','この範囲に公開対象・評価記録はありません。','empty'));return}let table=el('table',null,'map'),head=el('thead'),tr=el('tr');tr.append(el('th',method==='FULL_PF'?'公式':method==='TRUNCATED_PF'?'p':'L_D'));const widthOrder=label=>{const m=/^(\d+)(?:\/(\d+))?h/.exec(label);return m?Number(m[1])/Number(m[2]||1):Infinity};const sortedCols=[...cols].sort((a,b)=>widthOrder(a[1])-widthOrder(b[1])||a[1].localeCompare(b[1],'ja'));for(const [,label]of sortedCols){const th=el('th'),parts=label.split(' [');th.append(el('span',parts[0]));if(parts[1])th.append(el('small',parts[1].replace(/\]$/,'')));tr.append(th)}head.append(tr);table.append(head);let body=el('tbody');for(const [rk,rl]of rows){let row=el('tr');row.append(el('th',rl));for(const [wk]of sortedCols){let td=el('td'),c=cells.find(c=>c.row_key===rk&&c.width_key===wk);if(!c){td.append(el('div','対象として未確認','cell absent'))}else{let cls=c.feasible_allocations?'good':c.evaluated_allocations&&!c.missing_count?'bad':'wait',b=el('button',null,'cell '+cls);b.append(el('span',states[c.state]||c.state||'状態未同定','pill'),el('strong',c.physical_evaluated+'構成 / '+c.evaluated_allocations+'配分'),el('small','適合 '+c.feasible_allocations+'配分'));if(c.min_cost!==null&&c.min_cost!==undefined)b.append(el('div',fmt(c.min_cost)+' RZ','price'));else b.append(el('small','適合価格 '+(c.feasible_allocations?'未接続':'なし')));for(const label of c.price_validation_labels||[])b.append(el('span',label,'pill'));if(c.missing_count)b.append(el('small','記録未取込 '+c.missing_count));b.dataset.state=c.state;b.setAttribute('aria-label',rl+' '+c.width_label+' '+(states[c.state]||c.state));b.setAttribute('aria-pressed','false');b.onclick=()=>{for(const x of host.querySelectorAll('button[aria-pressed="true"]'))x.setAttribute('aria-pressed','false');b.setAttribute('aria-pressed','true');showCell(c);E('detail').focus()};td.append(b)}row.append(td)}body.append(row)}table.append(body);host.append(table)}
for(const [m,label]of Object.entries(labels)){let b=el('button',label,m===method?'active':'');b.onclick=()=>{method=m;for(const x of E('methods').children)x.classList.toggle('active',x===b);redraw()};E('methods').append(b)}
for(const v of D.views||[]){let option=el('option',v.label||v.id);option.value=v.id;E('view').append(option)}E('view').onchange=redraw;E('search').oninput=redraw;E('clear').onclick=()=>{E('search').value='';redraw()};E('coverage').textContent='収録 '+(D.runs||[]).length+' run / '+(D.points||[]).length+' 一意配分点 / '+(D.sources||[]).length+' 固定SOURCE。表示用の丸め値から価格・判定を再計算しません。';
for(const button of E('overview').querySelectorAll('button[data-point]')){button.onclick=()=>{const v=(D.views||[]).find(v=>v.mode==='ALL');if(!v)return;method=button.dataset.method;E('view').value=v.id;E('search').value='';for(const x of E('methods').children)x.classList.toggle('active',x.textContent===labels[method]);redraw();const c=v.cells.find(c=>c.method===method&&(c.min_point_ids||[]).includes(button.dataset.point));if(c){const b=[...E('grid').querySelectorAll('button')].find(b=>b.getAttribute('aria-label')===c.row_label+' '+c.width_label+' '+(states[c.state]||c.state));if(b)b.setAttribute('aria-pressed','true');showCell(c,button.dataset.point);E('detail').focus()}}}
let sourceHost=E('sources');let comp=el('details');comp.append(el('summary','共通snapshot・精度・比較条件'),jsonBlock(D.comparison));sourceHost.append(comp);for(const r of D.runs||[]){let item=el('details');item.append(el('summary',(r.label||r.id)+' · '+(r.status||'状態未同定')));field(item,'run ID',r.id);field(item,'host',r.hosts);field(item,'未取込/不明',r.missing||r.issues);if(r.source)item.append(link(r.source));sourceHost.append(item)}let src=el('details');src.append(el('summary','固定SOURCE一覧'));let list=el('ul',null,'source-list');for(const s of D.sources||[]){let li=el('li');li.append(link(s));list.append(li)}src.append(list);sourceHost.append(src);let issue=el('details');issue.append(el('summary','取込み待ち・不足情報'),jsonBlock(D.issues||[]));sourceHost.append(issue);redraw();
</script></body></html>'''
