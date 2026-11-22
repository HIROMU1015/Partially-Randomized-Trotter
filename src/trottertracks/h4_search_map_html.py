"""Standalone, offline HTML view of a saved H4 search-map index.

Only presentation: no signal, Q3, price or optimization code is called.
"""
import base64
import gzip
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


def index_text(index):
    """Canonical compact JSON shared by the browser and the machine-readable map."""
    return json.dumps(index, ensure_ascii=False, separators=(',', ':'))


def render(index):
    """Render one self-contained document; the embedded index remains exact."""
    # Escape script-closing strings without altering decoded JSON strings.
    raw = index_text(index).encode('utf-8')
    if len(raw) >= 32 * 1024**2:
        data = base64.b64encode(gzip.compress(raw, compresslevel=4, mtime=0)).decode('ascii')
        data_type = 'application/gzip'
    else:
        data = raw.decode('utf-8').replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
        data_type = 'application/json'
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
    return _PAGE.replace('<!--DATA_TYPE-->', data_type).replace('<!--DATA-->', data).replace('<!--FALLBACK-->', ''.join(fallback))


_PAGE = r'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>H₄ RPE 探索マップ</title><style>
:root{color-scheme:light;--ink:#17263c;--muted:#52667d;--line:#d7e0ea;--paper:#fff;--blue:#185cba;--pale:#eef4fd;--good:#e7f5ed;--warn:#fff1d4;--bad:#fcecef}
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:var(--ink);font:15px/1.6 system-ui,-apple-system,"Noto Sans",sans-serif}main{max-width:1480px;margin:auto;padding:30px 26px 70px}h1{font-size:30px;line-height:1.2;margin:5px 0 12px}h2{font-size:21px;margin:0 0 14px}h3{font-size:17px;margin:15px 0 8px}p{margin:8px 0}.eyebrow{letter-spacing:.1em;color:var(--blue);font-size:12px;font-weight:700}.sub{color:var(--muted);max-width:1000px}.notice{background:var(--pale);border-left:4px solid var(--blue);padding:10px 14px;margin:18px 0}.card{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:20px;margin:16px 0;box-shadow:0 2px 8px #17263c04}.bar{display:flex;gap:12px;align-items:end;flex-wrap:wrap}.bar label{font-weight:600;font-size:13px;display:block}select,input,button{font:inherit;border:1px solid #b8c8dc;border-radius:7px;background:white;color:var(--ink);padding:9px 12px}select{max-width:100%}input{min-width:240px}button{cursor:pointer}button:focus-visible,a:focus-visible,select:focus-visible{outline:3px solid #f6b839;outline-offset:2px}.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:20px 0 14px}.tabs button.active{background:var(--blue);color:white;border-color:var(--blue)}.metrics{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:10px;margin:16px 0}.metric{background:#fff;border:1px solid var(--line);border-radius:9px;padding:12px 16px}.metric strong{display:block;font-size:25px;letter-spacing:-.02em}.metric span{font-size:12px;color:var(--muted)}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px 10px;border:1px solid var(--line);text-align:left;vertical-align:top}th{background:#f5f8fc;position:sticky;top:0;z-index:1}th:first-child{white-space:nowrap}.map{table-layout:fixed;min-width:650px}.map td{padding:0;min-width:165px}.cell{display:block;width:100%;min-height:120px;text-align:left;border:0;border-radius:0;background:white;padding:12px;line-height:1.5}.cell:hover{box-shadow:inset 0 0 0 2px var(--blue)}.cell strong{display:block;color:var(--ink);font-size:16px}.cell small{display:block;color:var(--muted);font-size:12px}.cell .price{font-weight:700;color:#07573a;font-size:14px;margin:5px 0}.cell.good{background:var(--good)}.cell.wait{background:var(--warn)}.cell.bad{background:var(--bad)}.cell.absent{background:#f5f6f8;color:#657184}.pill{display:inline-block;border-radius:4px;padding:2px 6px;font-size:11px;background:#e6ecf3;margin-bottom:4px}.legend{display:flex;gap:15px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:12px}.legend b{display:inline-block;width:10px;height:10px;border:1px solid #9eacbc;margin-right:5px}.mono,code,pre{font-family:ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere}code{font-size:12px;background:#f4f6fa;padding:2px 4px}pre{font-size:12px;white-space:pre-wrap;margin:8px 0;padding:12px;background:#f6f8fc;border-radius:7px;max-height:340px;overflow:auto}.break{overflow-wrap:anywhere}a{color:#145cb9;text-decoration:none}a:hover{text-decoration:underline}.muted{color:var(--muted)}.detail-head{display:flex;justify-content:space-between;gap:20px}.item{border-top:1px solid var(--line);padding-top:12px;margin-top:14px}details{margin:10px 0}summary{cursor:pointer;font-weight:600}.source-list li{margin:7px 0;overflow-wrap:anywhere}.empty{padding:30px;text-align:center;color:var(--muted)}.run-scope{padding:10px;background:#f5f8fc;border-radius:7px}.fallback-table td{max-width:300px;overflow-wrap:anywhere}.data-note{font-size:12px;color:var(--muted)}[hidden]{display:none!important}
@media(max-width:750px){main{padding:22px 12px 45px}.card{padding:14px}.metrics{grid-template-columns:repeat(2,1fr)}h1{font-size:25px}.detail-head{display:block}.bar>*{width:100%}input{min-width:0;width:100%}.map{min-width:600px}}
</style></head><body><main>
<header><div class="eyebrow">SAVED EVIDENCE · H₄ / RPE</div><h1>実際に探索した範囲を見る</h1><p class="sub">方式・要求幅・分割を切り替え、公開された物理構成とshot配分の評価範囲を確認します。価格は保存済みの条件付き参照費用です。</p></header>
<div class="notice">結論の範囲は「収録した保存候補集合内」。全lawの平均compiled価格や正式G・winnerの新しい資格は与えません。新しい信号・Q3・compileは起動しません。</div>
<section class="card"><div class="bar"><div><label for="view">表示範囲</label><select id="view"></select></div><div><label for="search">候補を絞る</label><input id="search" type="search" placeholder="configuration・公式・κ・配分"></div><button id="clear">絞り込み解除</button></div><p id="coverage" class="data-note"></p><div id="scope" class="run-scope"></div></section>
<nav id="methods" class="tabs" aria-label="方式"></nav><details class="card" id="scaled-panel" hidden><summary>倍率格子の探索結果</summary>
<p class="data-note">保存コントローラーの条件付き総RZ。格子内・比較専用・従来の自由配分を分けて表示します。</p>
<div class="bar"><label>集約条件（L_D・公式・要求幅・M）<select id="scaled-group"></select></label></div>
<div id="scaled-summary"></div><div id="scaled-rows" class="scroll"></div></details><section class="card parameter-panel"><h2 id="range-title">探索範囲</h2>
<div class="bar"><label><span id="dimension-label">分割</span><select id="dimension-filter"></select></label><label>要求PF幅<select id="width-filter"></select></label><label>PF公式<select id="formula-filter"></select></label></div>
<div class="joint-tools" role="group" aria-label="表示する記録"><button type="button" id="joint-measured" aria-pressed="true">Q3評価済み</button><button type="button" id="joint-targets" aria-pressed="false">登録候補</button></div>
<div class="joint-head"><h3 id="joint-title">実際に評価した組み合わせ</h3><span class="joint-stats" id="joint-count" aria-live="polite"></span></div>
<p class="data-note" id="joint-guide"></p><div id="joint-tree"></div>
<details id="axis-summary"><summary>各パラメータの指定範囲と評価値</summary><p id="subset-note" class="data-note"></p><div id="parameter-table" class="scroll"></div><details><summary>ラウンド別の実評価範囲（q、R、X/Yショット）</summary><div id="round-ranges" class="scroll"></div></details></details>
</section><div id="metrics" class="metrics"></div>
<section class="card"><h2>探索の継続・停止と費用の方向</h2>
<details><summary>探索の継続・停止と次の候補</summary><div id="search-progress"></div></details>
<details><summary>一つのパラメータを変えたときの費用差</summary>
<p class="data-note">保存点の比較。差額は変更後 − 変更前の条件付き総RZ。適合済みの最安配分を各点から使い、未評価の途中の値は推定しません。</p>
<div id="search-directions" class="scroll"></div></details></section>
<details class="card" id="coverage-matrix"><summary id="map-title">組合せごとの評価記録</summary><p class="data-note">「対象登録あり・評価記録なし」は対象の記録だけがある組合せ。「—」は対象登録も評価記録も見つからない組合せです。「元結果未取込」は、対応する結果が索引に揃っていないことを表します。</p><div id="grid" class="scroll"></div></details>
<section id="detail" tabindex="-1" class="card" aria-live="polite"><h2>評価記録の詳細</h2><p class="muted">公式・κ・実schedule・X/Y配分・価格根拠・固定commitへたどれます。</p></section>
<details class="card"><summary>収録run・比較条件・取り込み待ちの記録</summary><div id="sources"></div></details>
<details class="card"><summary>全収録点の静的表（JavaScriptなしでも閲覧可能）</summary><p class="data-note">ブラウザ内検索も使えます。詳細なJSONは <a href="INDEX.json">INDEX.json</a> を参照してください。</p><div class="scroll"><table class="fallback-table"><thead><tr><th>方式</th><th>公式/p/LD</th><th>要求幅</th><th>configuration</th><th>保存Q3判定</th><th>条件付き費用</th><th>N_X</th><th>N_Y</th><th>固定SOURCE</th></tr></thead><tbody><!--FALLBACK--></tbody></table></div></details>
<footer class="data-note">Cの比較カタログと公開探索記録を読取専用で結合。ロードマップとは別の探索coverage成果物です。</footer>
</main><script id="index-data" type="<!--DATA_TYPE-->"><!--DATA--></script><script>
'use strict';
async function loadMap() {
const source=document.getElementById('index-data');
if(source.type==='application/json')return JSON.parse(source.textContent);
const compressed=Uint8Array.from(atob(source.textContent), c=>c.charCodeAt(0));
return new Response(new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'))).json();
}
function displayMap(D) {
const E=id=>document.getElementById(id), labels={FULL_PF:'Full PF',TRUNCATED_PF:'B0I',PR_PAULI:'PR-Pauli',PR_DF:'PR-DF'}, formulas={'2nd':'S2','4th':'標準Yoshida4','4th(m5_best)':'M5'};
const states={TARGET_UNEVALUATED:'探索対象・未評価',EVALUATED_NO_FEASIBLE:'評価済み・適合0件',SOURCE_NOT_IMPORTED:'元記録が未取込',NOT_TARGETED:'対象として未確認',MIXED:'評価済み＋未取込/未評価',FEASIBLE:'適合点あり',EVALUATED:'評価済み',PRICE_WAITING:'Q3適合・価格待ち'};
let method='FULL_PF'; const allPoints=new Map((D.points||[]).map(p=>[p.id,p])), allTargets=new Map((D.targets||[]).map(p=>[p.id,p]));
function el(tag,text,cls){let n=document.createElement(tag);if(text!==undefined&&text!==null)n.textContent=String(text);if(cls)n.className=cls;return n}
function fmt(x){if(x===undefined||x===null)return '—';if(typeof x==='object'&&'numerator'in x&&'denominator'in x){let n=Number(x.numerator)/Number(x.denominator);return Number.isFinite(n)?n.toLocaleString('ja-JP',{maximumFractionDigits:2}):String(x.numerator)+'/'+String(x.denominator)}return typeof x==='object'?JSON.stringify(x):String(x)}
function exact(x){return x&&typeof x==='object'&&'numerator'in x?String(x.numerator)+'/'+String(x.denominator):fmt(x)}
function link(s){let n=el('span');if(!s||!s.commit||!s.path){n.textContent=fmt(s);return n}let a=el('a',String(s.commit).slice(0,9)+':'+s.path);a.href='https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'+encodeURIComponent(s.commit)+'/'+s.path.split('/').map(encodeURIComponent).join('/');a.target='_blank';a.rel='noopener';n.append(a);if(s.selector&&s.selector.length)n.append(el('code',JSON.stringify(s.selector)));return n}
function jsonBlock(value){return el('pre',JSON.stringify(value,null,2))}
function configuration(p){return(D.configuration_details||{})[p.configuration_detail_key]||{}}
function field(parent,label,value){let p=el('p');p.append(el('strong',label+'：'),el('span',fmt(value)));parent.append(p)}
function validationDetails(parent,p){
 for(const view of p.price_validations||[]){const model=(D.price_validation_models||{})[view.model_id]||{};let box=el('details');box.append(el('summary','価格モデル簡易検証 · '+(view.label||view.status||'検証未完了')));field(box,'モデルID',view.model_id);box.append(el('p','簡易検証は保存した実測点での診断です。適用先の全ラウンドやlaw期待値を認定せず、係数・費用・順位へ補正を加えません。','data-note'));field(box,'今回の適用点 (m,q,R)',view.application_points);let identity=el('details');identity.append(el('summary','凍結係数・SOURCE・適用条件'),jsonBlock(model.identity||model));box.append(identity);
 for(const point of model.validation_points||[]){let item=el('div',null,'item');field(item,'検証ラウンド / q / R',fmt(point.m)+' / '+fmt(point.q)+' / '+fmt(point.R));field(item,'独立経路数（X/Yは同じ経路を共有）',point.independent_paths);let table=el('table'),head=el('tr');for(const h of ['軸','事前予測 RZ/shot','実測平均 RZ/shot','符号付き相対差 (予測−実測)/実測'])head.append(el('th',h));table.append(head);for(const axis of ['X','Y']){let tr=el('tr');for(const val of [axis,point.predicted_RZ?.[axis],point.observed_mean_RZ?.[axis],point.signed_relative_difference?.[axis]])tr.append(el('td',exact(val)));table.append(tr)}item.append(table);let observed=el('table'),oh=el('tr');for(const h of ['経路','軸','実測 RZ/shot','元receipt'])oh.append(el('th',h));observed.append(oh);for(const row of point.observations||[]){let tr=el('tr');tr.append(el('td',row.path_id),el('td',row.axis),el('td',exact(row.compiled_RZ)));let source=el('td');source.append(link(row.source));tr.append(source);observed.append(tr)}item.append(observed);if(point.source)item.append(link(point.source));const share=(view.validation_round_contributions||[]).find(v=>v.m===point.m);if(share){field(item,'この配分での当該roundのモデル総額比 (%)',share.percent_of_model_total===null?'UNKNOWN（対応する全軸価格または配分なし）':exact(share.percent_of_model_total)+' ≈ '+fmt(share.percent_of_model_total));field(item,'実測点と今回の適用点が同じ(q,R)',share.same_application_point?'はい':'いいえ（検証点と適用点を区別）')}box.append(item)}
 const refs=el('ul',null,'source-list'),seen=new Set();function collect(value){if(!value||typeof value!=='object')return;if(value.commit&&value.path){let k=JSON.stringify(value);if(!seen.has(k)){seen.add(k);let li=el('li');li.append(link(value));refs.append(li)}return}for(const child of Object.values(value))collect(child)}collect(model);collect(view.source);box.append(el('strong','較正・凍結・独立検証の固定参照'),refs);parent.append(box)}
}
function selected(){return(D.views||[]).find(v=>v.id===E('view').value)||(D.views||[])[0]||{cells:[],mode:'ALL'}}
function passes(p,q){return !q||JSON.stringify([p.configuration_id,p.formula,p.L_D,p.p,p.kappa,p.N_X,p.N_Y,p.requested_width]).toLowerCase().includes(q)}
function points(c){const q=E('search').value.trim().toLowerCase();return(c.point_ids||[]).map(id=>allPoints.get(id)).filter(Boolean).filter(p=>passes(p,q))}
function showCell(c){let d=E('detail');d.replaceChildren();let h=el('div',null,'detail-head');h.append(el('h2',labels[method]+' · '+c.row_label+' × '+c.width_label),el('span',states[c.state]||c.state,'pill'));d.append(h);field(d,'Q3構成 / 配分 / 適合',c.physical_evaluated+' / '+c.evaluated_allocations+' / '+c.feasible_allocations);if(c.min_cost!==null&&c.min_cost!==undefined){field(d,'最安条件付き総RZ（表示用）',fmt(c.min_cost));field(d,'比較に使うexact値',exact(c.min_cost))}
 const ps=points(c);if(!ps.length)d.append(el('p','該当する評価点はありません。以下の対象・欠測記録を確認してください。','muted'));
 const progress=el('p',null,'data-note'),articles=el('div'),more=el('button','次の50件');let shown=0;d.append(progress,articles,more);
 function appendPage(){const end=Math.min(shown+50,ps.length);
 for(const p of ps.slice(shown,end)){let wrap=el('article',null,'item');wrap.append(el('h3',(formulas[p.formula]||p.formula||'公式未同定')+' · '+(p.q3_status||p.status||'状態未同定')));field(wrap,'configuration',p.configuration_id);field(wrap,'分割',p.p!==null&&p.p!==undefined?'p='+p.p:'L_D='+p.L_D);field(wrap,'要求幅',p.requested_width);field(wrap,'κ',p.kappa);field(wrap,'条件付き総RZ',p.conditional_cost===null?'未接続':exact(p.conditional_cost)+' ≈ '+fmt(p.conditional_cost));field(wrap,'N_X',p.N_X);field(wrap,'N_Y',p.N_Y);validationDetails(wrap,p);
 const cfg=configuration(p);let sections=[['実schedule・physical identity',p.schedule||cfg.schedule],['固定回路条件・law',{profile:p.profile||cfg.profile,law:p.law||cfg.law,split:p.split||cfg.split}],['保存MSE上下界・状態',{lower:p.MSE_lower,upper:p.MSE_upper,status:p.q3_status,evidence:p.evidence_level,issues:p.issues}],['価格根拠・qualification',p.price_evidence],['run別の新規評価／再利用と実行host',p.occurrences]];
 for(const [title,value]of sections){let x=el('details');x.append(el('summary',title),jsonBlock(value===undefined?null:value));wrap.append(x)}
 let refs=el('ul',null,'source-list');for(const s of [...(p.sources||[]),...(cfg.configuration_source?[cfg.configuration_source]:[])]){let li=el('li');li.append(link(s));refs.append(li)}wrap.append(el('strong','元結果への固定参照'),refs);articles.append(wrap)}
 shown=end;progress.textContent='表示 '+shown+' / '+ps.length+' 配分（一覧・件数・検索対象は全件）';more.hidden=shown>=ps.length;}
 more.onclick=appendPage;appendPage();
 const ts=(c.target_ids||[]).map(id=>allTargets.get(id)).filter(Boolean);if(ts.length){let x=el('details');x.open=!ps.length;x.append(el('summary','探索対象・選択・不足記録 ('+ts.length+')'));for(const t of ts){let item=el('div',null,'item');field(item,'状態',t.status);field(item,'構成',t.configuration_id);field(item,'不足',t.missing);if(t.source)item.append(link(t.source));x.append(item)}d.append(x)}
}
const gridStatus={MINIMUM_WITHIN_REGISTERED_CANDIDATES:'登録済み回路の範囲で格子内確定',GRID_MINIMUM_CONFIRMED:'この回路とMで格子内確定',EVALUATED_MINIMUM:'評価済み最安',
 NO_GRID_POINT_MEETS_LIMIT:'格子内適合なし',NO_CONFIRMED_GRID_POINT:'適合未確認'};
const scaledStops={SCALED_SHOT_GRID_COMPLETE:'指定範囲の倍率探索が完了',
 Q3_BUDGET_EXHAUSTED:'新規Q3の予算上限',STEP_LIMIT:'探索step上限',WORK_SLICE_YIELDED:'次の探索区間に継続',
 UNRESOLVED_SCALED_SCOPE:'未評価・入力待ち・判定保留が残る', USER_REQUESTED_STOP:'利用者による停止', UNKNOWN_RESULT:'結果照合が必要'};
const scaledDirections={COST_DOWN:'費用減',COST_UP:'費用増',EQUAL:'同額',UNRESOLVED:'費用差は未確定'};
function scaledView(view){
 const panel=E('scaled-panel'), filter=E('scaled-group'), host=E('scaled-rows'), summary=E('scaled-summary');
 const runs=(D.runs||[]).filter(r=>(!view.run_id||r.id===view.run_id)&&r.scaled_shots?.[method]);
 panel.hidden=!runs.length;host.replaceChildren();summary.replaceChildren();
 if(!runs.length)return;
 const groupLabel=r=>{
  const c=r.configuration||{};
  return [c.p==null?null:'p='+c.p,c.L_D==null?null:'L_D='+c.L_D,c.formula,
   c.delta_pf==null?null:'Δ='+exact(c.delta_pf),c.delta_pf!=null||c.delta_over_h==null?null:'Δ/h='+c.delta_over_h,'M='+r.M].filter(Boolean).join(' · ');
 };
 const groups=[...new Set(runs.flatMap(r=>r.scaled_shots[method].configurations.map(groupLabel)))].sort();
 const previous=filter.value;
 filter.replaceChildren(el('option','すべて'));filter.firstChild.value='';
 for(const label of groups){const option=el('option',label);option.value=label;filter.append(option)}
 if(groups.includes(previous))filter.value=previous;
 filter.onchange=()=>scaledView(view);
 const q=E('search').value.trim().toLowerCase();
 const table=el('table'), header=el('tr');
 for(const name of ['run／回路条件','絶対κ','評価済み最小適合倍率','条件付き総RZ','状態','比較専用点・入力不足']){
  header.append(el('th',name));
 }
 table.append(header);
 for(const run of runs){
  const {summary:s,configurations,stop_reason,horizon_directions}=run.scaled_shots[method];
  const card=el('details');card.append(el('summary',run.label));
  field(card,'停止・継続理由',scaledStops[stop_reason]||stop_reason);
  field(card,'指定探索範囲',s.policy);
  field(card,'M=12の格子内',s.M12_best?.cost);
  field(card,'全Mの格子内',s.all_M_best?.cost);
  field(card,'集約状態',gridStatus[s.all_M_minimum_status]||s.all_M_minimum_status);
  field(card,'比較専用の最安／格子内との差',s.comparison_only_best?fmt(s.comparison_only_best.cost)+' / '+fmt(s.comparison_only_delta_RZ):null);
  field(card,'従来の保存配分（参照）',s.legacy_free_allocation_reference?.cost);
  if(horizon_directions?.length){
   const trend=el('details');trend.append(el('summary','同じ回路でMを変えたときの費用差'));
   trend.append(el('p','指定範囲の評価済み最小適合配分を両端で比較。配分はMごとに生成。'));
   for(const d of horizon_directions){
    const item=el('div',null,'item');
    field(item,'回路',d.source_configuration_id);
    field(item,'M='+d.from_M+' → '+d.to_M,scaledDirections[d.direction]+(d.delta_RZ?' '+exact(d.delta_RZ)+' RZ':''));
    field(item,'両端の状態',[gridStatus[d.from_status]||d.from_status,gridStatus[d.to_status]||d.to_status]);
    item.append(link(d.from_source),link(d.to_source));
    trend.append(item);
   }
   card.append(trend);
  }
  summary.append(card);
  for(const row of configurations){
   const label=groupLabel(row), c=row.configuration||{}, best=row.best, comp=row.comparison_best;
   if((filter.value&&label!==filter.value)||(q&&!JSON.stringify([label,c.absolute_kappa,row.configuration_id]).toLowerCase().includes(q)))continue;
   const tr=el('tr');tr.append(el('td',run.label+' · '+label),el('td',c.absolute_kappa??'—'));
   const aliases=(best?.aliases||[]).filter(a=>a.kind==='GRID').map(a=>a.value);
   const fit=el('td',aliases[0]||'—');
   if(best){
    const detail=el('details');detail.append(el('summary','配分・倍率の由来'));
    for(const [key,value] of [['X',best.N_X],['Y',best.N_Y],['MSE下界',best.MSE_lower],
       ['MSE上界',best.MSE_upper],['保存・新規',best.evidence],['倍率',best.aliases]])field(detail,key,value);
    fit.append(detail);
   }
   tr.append(fit,el('td',fmt(best?.cost)),el('td',gridStatus[row.minimum_status]||row.minimum_status));
   const notes=el('td');
   if(comp)field(notes,'比較専用 ' + (comp.aliases||[]).map(a=>a.value).join('・'),
       fmt(comp.cost)+' RZ （格子との差 '+fmt(row.comparison_delta_RZ)+'）');
   if(row.needs?.length)field(notes,'準備待ち',row.needs);
   if(row.required_rounds?.length)field(notes,'不足ラウンド',row.required_rounds);
   if(row.allocation_states)field(notes,'配分の状態',row.allocation_states);
   const reasons=Object.entries(row.allocation_reasons||{}).map(([reason,count])=>
      [(reason==='OWN_GRID_PASS_COST_OR_TIE'?'同じ回路・Mの適合点より安くならず、同額でより小さい倍率にもならない':reason),count]);
   if(reasons.length)field(notes,'省略・保留理由',Object.fromEntries(reasons));
   if(row.source)notes.append(link(row.source));
   tr.append(notes);table.append(tr);
  }
 }
 host.append(table);
}
const parameterNames={M:'最終ラウンド M',formula:'PF公式',width:'要求PF幅 Δ',p:'保持数 p',L_D:'決定論側 L_D',kappa:'κ'};
const boundaryReasons={RANGE_LIMIT:'設定した探索上限',PROVEN_COST_FLOOR:'価格下限で改善不能',
 EMPIRICAL_UPPER_STOP:'観測傾向で打ち切り',NO_FEASIBILITY_PROGRESS:'適合への進展が見られず保留'};
const controllerStops={USER_REQUESTED_STOP:'手動停止・途中',Q3_BUDGET_EXHAUSTED:'予算上限・途中',
 WAITING_FOR_INPUTS:'入力待ち',SCOPE_HELD_INPUTS:'入力保留',REQUESTED_WIDTH_RULE_EXHAUSTED:'今回の探索規則で終了',
 MAX_CYCLES:'サイクル上限・途中',RULE_EXHAUSTED:'今回の探索規則で終了',WAIT_FOR_PENDING:'候補処理待ち'};
const candidateStates={WAITING_INPUTS:'入力待ち',HELD:'保留',EVALUATED:'評価済み',COMPLETION:'準備候補',
 PREPARED:'準備済み',REGISTERED:'登録済み',PRUNED:'探索対象外に移動',READY_METADATA:'条件の登録済み'};
const differenceLabels={COST_DOWN:'費用低下',COST_UP:'費用上昇',EQUAL:'同額',UNRESOLVED:'費用差は未確定'};
const endpointLabels={FEASIBLE:'適合',PRICE_WAITING:'Q3適合・価格待ち',INFEASIBLE:'試した配分は基準超過',
 PENDING:'判定保留',Q3_WAITING:'Q3未評価',MISMATCH:'条件不一致'};
function tableHeader(titles){const table=el('table'),row=el('tr');for(const title of titles)row.append(el('th',title));table.append(row);return table}
function progressView(run){
 const host=E('search-progress');host.replaceChildren();
 const p=run?.search_progress;
 if(!p){host.append(el('p',run?'このrunには方向別の探索判断が未収録です。':'表示範囲からrunを選ぶと、その時点の探索判断を確認できます。'));return}
 if(p.status!=='RECORDED'){field(host,'取得状況',p.status);return}
 const reason=p.controller_stop?.reason;
 field(host,'runの状態',controllerStops[reason]||reason||'停止記録なし');
 field(host,'記録したcycle / 最終計画cycle',fmt(p.controller_cycle)+' / '+fmt(p.planner_cycle));
 if(p.planner_stop)field(host,'最終計画の状態',controllerStops[p.planner_stop]||p.planner_stop);
 const rows=(p.families||[]).filter(f=>!f.method||f.method===method);
 if(!rows.length)host.append(el('p','この方式には方向別の探索判断が未収録です。'));
 else{
  const table=tableHeader(['候補群','κの評価済み範囲','下側の境界','上側の境界と根拠','候補の状態','次の候補・判断']);
  for(const f of rows){
   const title=f.method&&f.L_D!=null?'L_D='+f.L_D+' · '+(formulas[f.formula]||f.formula)+' · Δ='+exact(f.requested_delta_pf):f.family;
   const tr=el('tr');tr.append(el('th',title));
   const range=(f.kappa_range_evaluated||[]).map(v=>v==null?'未評価':exact(v)).join(' ～ ');
   const coverage=el('td',range||'未収録');
   if(f.requested_kappa_bounds?.length)coverage.append(el('div','指定範囲 '+f.requested_kappa_bounds.map(exact).join(' ～ '),'muted'));
   tr.append(coverage,
    el('td',f.floor_end==null?'判断未収録':'最小スケジュール区間の端 '+exact(f.floor_end)),
    el('td',f.upper_basis?(boundaryReasons[f.upper_basis]||f.upper_basis)+' · '+exact(f.upper_limit):'判断未収録'));
   const status=Object.entries(f.point_status||{}).map(([k,v])=>(candidateStates[k]||k)+' '+v+'件').join(' / ');
   tr.append(el('td',[candidateStates[f.status]||f.status,status,f.open_gaps==null?'':'未精査の間隔 '+f.open_gaps].filter(Boolean).join(' · ')||'未収録'));
   const next=el('td');
   const proposals=(f.next_candidates||[]).map(v=>[v.direction==='UP'?'κ↑':v.direction==='DOWN'?'κ↓':'κ',exact(v.kappa),candidateStates[v.status]||v.status,v.reason].filter(Boolean).join(' '));
   next.append(el('span',proposals.join(' / ')||'候補の記録なし'));
   if(f.decisions?.length){const decisions=el('details');decisions.append(el('summary','計画の判定根拠'),jsonBlock(f.decisions));next.append(decisions)}
   tr.append(next);table.append(tr);
  }
  host.append(table);
 }
 if(p.source)host.append(link(p.source));
}
function directionView(view,q){
 const host=E('search-directions');host.replaceChildren();
 const inView=new Set((view.cells||[]).filter(c=>c.method===method).flatMap(c=>c.point_ids||[]));
 const rows=(D.direction_records||[]).filter(row=>row.method===method
  && inView.has(row.from_point)&&inView.has(row.to_point)
  && [row.from_point,row.to_point].some(id=>passes(allPoints.get(id)||{},q)));
 if(!rows.length){host.append(el('p','この表示範囲には、条件を揃えて比較できる保存点の組がありません。'));return}
 const table=tableHeader(['変えたパラメータ','比較前','比較後','費用差','固定したパラメータ・配分']);
 const more=el('button','次の50件');host.append(table,more);let shown=0;
 function endpoint(p){
  const box=el('td');const status=endpointLabels[p.status]||p.status;
  box.append(el('div',status+(p.conditional_cost!=null&&p.status==='FEASIBLE'?' · '+fmt(p.conditional_cost)+' RZ':'')));
  const detail=el('details');detail.append(el('summary','配分と元記録'));
  field(detail,'configuration',p.configuration_id);field(detail,'N_X',p.N_X);field(detail,'N_Y',p.N_Y);
  if(p.price_evidence?.scenario)field(detail,'価格の条件',p.price_evidence.scenario);
  for(const ref of p.sources||[])detail.append(link(ref));box.append(detail);return box;
 }
 function nextPage(){
  for(const row of rows.slice(shown,shown+50)){
   const tr=el('tr'),label=parameterNames[row.parameter]||row.parameter;
   const val=x=>row.parameter==='formula'?(formulas[x]||x):x;
   const diff=el('td',differenceLabels[row.direction]||row.direction);
   if(row.delta_RZ!=null){const sign=row.direction==='COST_UP'?'+':'';diff.append(el('div',sign+fmt(row.delta_RZ)+' RZ'));
    diff.title=sign+exact(row.delta_RZ)+' RZ'}
   const fixed=Object.entries(row.fixed||{}).map(([k,v])=>(parameterNames[k]||k)+'='+v).join(' · ');
   tr.append(el('th',label+' '+val(row.from_value)+' → '+val(row.to_value)),
    endpoint(allPoints.get(row.from_point)),endpoint(allPoints.get(row.to_point)),diff,
    el('td',fixed+(row.same_allocation?' · 同じ配分':' · 各点の保存配分')));
   table.append(tr);
  }
  shown+=50;more.hidden=shown>=rows.length;
 }
 more.onclick=nextPage;nextPage();
}
function redraw(){E('detail').replaceChildren(el('h2','評価点の詳細'),el('p','組み合わせから配分を選び、「元記録を見る」を開いてください。','detail-note'));const v=selected();scaledView(v);const q=E('search').value.trim().toLowerCase();let sourceCells=(v.cells||[]).filter(c=>c.method===method);
 E('dimension-label').textContent=method==='FULL_PF'?'公式':method==='TRUNCATED_PF'?'p':'L_D';
 selectOptions('dimension-filter',sourceCells.map(c=>[c.row_key,c.row_label]));
 E('dimension-filter').parentElement.hidden=method==='FULL_PF';
 selectOptions('width-filter',sourceCells.map(c=>[c.width_key,c.width_label]));
 const records=sourceCells.flatMap(c=>[...(c.point_ids||[]).map(i=>allPoints.get(i)),...(c.target_ids||[]).map(i=>allTargets.get(i))]).filter(Boolean);
 selectOptions('formula-filter',records.filter(r=>r.formula).map(r=>[r.formula,formulas[r.formula]||r.formula]));
 const getRecords=(ids,map)=>(ids||[]).map(i=>map.get(i)).filter(r=>r&&passes(r,q));
 const ts=records.filter(r=>allTargets.has(r.id)&&passes(r,q)),evaluated=records.filter(r=>allPoints.has(r.id)&&passes(r,q)&&r.q3_evaluated);
 buildRanges(v,ts,evaluated);
 buildCombinations(ts,evaluated);
 let cells=sourceCells.filter(c=>getRecords(c.point_ids,allPoints).length||getRecords(c.target_ids,allTargets).length);
 E('map-title').textContent='組合せごとの評価記録を開く · '+labels[method]+' · '+({FULL_PF:'公式 × 要求幅',TRUNCATED_PF:'切り捨て位置p × 要求幅',PR_PAULI:'L_D × 要求幅',PR_DF:'L_D × 要求幅'}[method]);
 let pointsIds=new Set(cells.flatMap(c=>c.point_ids||[])), ps=[...pointsIds].map(id=>allPoints.get(id)).filter(Boolean).filter(p=>passes(p,q));let configs=new Set(ps.filter(p=>p.q3_status).map(p=>p.configuration_id));let metrics=[['Q3評価済み物理構成',configs.size],['評価した配分',ps.filter(p=>p.q3_status).length],['適合配分',ps.filter(p=>p.q3_status==='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD').length],['記録未取込の対象',cells.reduce((n,c)=>n+(c.missing_count||0),0)]];E('metrics').replaceChildren(...metrics.map(([l,n])=>{let e=el('div',null,'metric');e.append(el('strong',n),el('span',l));return e}));
 E('scope').replaceChildren();let run=(D.runs||[]).find(r=>r.id===v.run_id);E('scope').append(el('strong',v.label||v.id||'全run累計'));if(run){field(E('scope'),'停止・進行状態',run.status);field(E('scope'),'実行host（保存metadata）',run.hosts);if(run.source)E('scope').append(link(run.source));if(run.missing?.length||run.issues?.length)field(E('scope'),'不足',run.missing||run.issues)}else E('scope').append(el('p','同じ構成・配分・評価契約の再検出／別runでの再利用は1件。母集合全体を網羅したとの主張ではありません。','data-note'));
 progressView(run);directionView(v,q);
 const rows=new Map(), cols=new Map();for(const c of cells){rows.set(c.row_key,c.row_label);cols.set(c.width_key,c.width_label)}let host=E('grid');host.replaceChildren();if(!cells.length){host.append(el('p','この範囲に公開対象・評価記録はありません。','empty'));return}let table=el('table',null,'map'),head=el('thead'),tr=el('tr');tr.append(el('th',method==='FULL_PF'?'公式':method==='TRUNCATED_PF'?'p':'L_D'));const widthOrder=label=>{const m=/^(\d+)(?:\/(\d+))?h/.exec(label);return m?Number(m[1])/Number(m[2]||1):Infinity};const sortedCols=[...cols].sort((a,b)=>widthOrder(a[1])-widthOrder(b[1])||a[1].localeCompare(b[1],'ja'));for(const [,label]of sortedCols){const th=el('th'),parts=label.split(' [');th.append(el('span',parts[0]));if(parts[1])th.append(el('small',parts[1].replace(/\]$/,'')));tr.append(th)}head.append(tr);table.append(head);let body=el('tbody');for(const [rk,rl]of rows){let row=el('tr');row.append(el('th',rl));for(const [wk]of sortedCols){let td=el('td'),c=cells.find(c=>c.row_key===rk&&c.width_key===wk);if(!c){const blank=el('span','—','cell absent');blank.title='対象登録も評価記録も見つかりません';td.append(blank)}else{const count=getRecords(c.point_ids,allPoints).filter(r=>r.q3_evaluated).length;
 const cls=count?'good':c.missing_count?'absent':'wait',b=el('button',null,'cell '+cls);
 b.append(el('strong',count?'Q3評価 '+count+'配分':c.missing_count?'元結果未取込':'対象登録あり'),el('small',count?'配分の記録を見る':c.missing_count?'評価の有無を判定できず':v.mode==='NEW'?'このrunの新規評価記録なし':'Q3評価記録なし'));
 b.dataset.state=c.state;b.setAttribute('aria-label',rl+' '+c.width_label+' '+(states[c.state]||c.state));b.setAttribute('aria-pressed','false');b.onclick=()=>{for(const x of host.querySelectorAll('button[aria-pressed="true"]'))x.setAttribute('aria-pressed','false');b.setAttribute('aria-pressed','true');showCell(c);E('detail').focus()};td.append(b)}row.append(td)}body.append(row)}table.append(body);host.append(table)}
function clearDimensions(){for(const id of ['dimension-filter','width-filter','formula-filter'])E(id).value=''}
for(const id of ['dimension-filter','width-filter','formula-filter'])E(id).onchange=redraw;
for(const [m,label]of Object.entries(labels)){let b=el('button',label,m===method?'active':'');b.onclick=()=>{clearDimensions();method=m;for(const x of E('methods').children)x.classList.toggle('active',x===b);redraw()};E('methods').append(b)}
for(const v of D.views||[]){let option=el('option',v.label||v.id);option.value=v.id;E('view').append(option)}E('view').onchange=()=>{clearDimensions();redraw()};E('search').oninput=redraw;E('clear').onclick=()=>{clearDimensions();E('search').value='';redraw()};E('coverage').textContent='収録 '+(D.runs||[]).length+' run / '+(D.points||[]).length+' 一意配分点 / '+(D.sources||[]).length+' 固定SOURCE。表示用の丸め値から価格・判定を再計算しません。';
for(const button of E('overview').querySelectorAll('button[data-point]')){button.onclick=()=>{const v=(D.views||[]).find(v=>v.mode==='ALL');if(!v)return;clearDimensions();method=button.dataset.method;E('view').value=v.id;E('search').value='';for(const x of E('methods').children)x.classList.toggle('active',x.textContent===labels[method]);redraw();const c=v.cells.find(c=>c.method===method&&(c.min_point_ids||[]).includes(button.dataset.point));if(c){const b=[...E('grid').querySelectorAll('button')].find(b=>b.getAttribute('aria-label')===c.row_label+' '+c.width_label+' '+(states[c.state]||c.state));if(b)b.setAttribute('aria-pressed','true');showCell(c,button.dataset.point);E('detail').focus()}}}
let sourceHost=E('sources');let comp=el('details');comp.append(el('summary','共通snapshot・精度・比較条件'),jsonBlock(D.comparison));sourceHost.append(comp);for(const r of D.runs||[]){let item=el('details');item.append(el('summary',(r.label||r.id)+' · '+(r.status||'状態未同定')));field(item,'run ID',r.id);field(item,'host',r.hosts);field(item,'未取込/不明',r.missing||r.issues);if(r.source)item.append(link(r.source));sourceHost.append(item)}let src=el('details');src.append(el('summary','固定SOURCE一覧'));let list=el('ul',null,'source-list');for(const s of D.sources||[]){let li=el('li');li.append(link(s));list.append(li)}src.append(list);sourceHost.append(src);let issue=el('details');issue.append(el('summary','取込み待ち・不足情報'),jsonBlock(D.issues||[]));sourceHost.append(issue);redraw();
}
loadMap().then(displayMap).catch(error=>{
 const note=document.createElement('p');note.textContent='対話表示を開けません。下の静的表を参照してください。 '+error.message;
 document.querySelector('main').prepend(note);
});
</script></body></html>'''
