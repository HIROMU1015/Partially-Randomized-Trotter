#!/usr/bin/env python3
"""Document the saved-coordinate display packet; no numerical science."""
from pathlib import Path
import json

P=Path(__file__).resolve().parent
data=json.loads((P/'figure_data.json').read_text())
figs=data['figures'];records=data['records'];cov=json.loads((P/'coverage_audit.json').read_text())['rows']
EPS=('0.05','0.01','0.005','0.001','0.0005');MODES=('USUAL_SIGNED','NONCANCELLATION');SYS=('H2','H4','H6','H8')
def write(name,s):(P/name).write_text(s.rstrip()+'\n')
def k(domain,mode,e,s,series):return '|'.join((domain,mode,e,s,series))
def stem(g,m,e,kind):return f'{g}_{m.lower()}_{kind}_eps_{e.replace(".","p")}'
def links(b):return f'[PNG]({b}.png) / [SVG]({b}.svg) / [PDF]({b}.pdf)'

common='''直鎖H2/H4/H6、隣接1 Å、STO-3G、tol-only1e-8の保存DF rank3/10/19、T=.8/P=0。
新outer32 PRはprefix1/3/5、K4、q=1/2/4/8、δ=.8/.4/.2/.1、各outerの全piece合計32 short steps。
S2・3段Yoshida4・固定binary64 m5_bestを区別する。通常signed・経験的allowance倍率1・α_axis=.025の保存会計を使用。
costは保存解析shot込みのfull cosine/sine Hadamard-wrapper compiled RZ総費用。εは有限時間複素信号の要求精度で、energy精度ではない。
状態準備、実量子shot、noise、FT T count、全RPE/QPE round総費用は含まない。
新探索のnは候補ごとに33〜255で異なる。pilot・旧n8・確認cohortとpoolしない。
独立確認は固定commit c7c1d822で準備段階。提案n_Cを取得済み標本数へ読み替えない。'''

scope_text={
 'A':'''**Aは旧全方式と新outer32を別パネルに分けた拡張overview**。
旧強化S2/S4 PFとのidentity結合は固定報告で未実施と明記されているため、1軸8系列の直接比較や
高次PR/旧PF比は作らない。左は公開commit 5d536985の7系列・元140枠の値をそのまま表示。
右はouter32 PR S2と、Yoshida4/m5_bestの保存formula bestのうち低い**高次だけ**の値。
high bestにS2を含めない。絶対図は左右のy範囲を共通にして参考併置する。
相対図の分母は左がbest full PF（S2/S4）、右が同じouter32のPR S2で、左右の比を直接比較しない。
図タイトル・軸・caption・auditでこの区別を保持する。H8右パネルは未保存。''',
 'B':'''**Bは同じouter32系列内のPR内部比較**。
H2/H4/H6についてPR S2 / PR Yoshida4 / PR m5_bestを表示。各formulaの4q内の保存採用点を転記し、
新しい候補順位を作らない。相対費用は同じ系・精度・bias modeのouter32 PR S2を分母にする。
通常signedとNONCANCELLATIONを別図に保存する。''',
 'C':'''**Cは同じnative系列に結合済みのH4/H6・m5決定論部分集合との比較**。
PR S2 / PR high best / full PF m5 / B0 m5 / B0+I m5を表示する。
このm5はS4の一つであり、旧全方式図のPF S4と同一候補域・同一費用政策の系列ではない。
通常signedの相対図の分母は保存されたm5部分集合内の最良費用。全強化PFの最良とは呼ばない。
非相殺ではB0/B0+IのPF/discard成分が未保存なので点を描かず、full PF m5だけとの診断図とする。
その分母はfull PF m5であり、欠測候補を除外した値をm5部分集合全体の最良として代用していない。
H2は同じidentityで結合した決定論signal集合がなく、H8はouter32未保存のためCのx軸に含めない。
同一系列のS2決定論対照も未保存・未結合であり、旧PFを代入しない。'''}

readme='''# 高次PRを含む保存済み比較図

作成：2026-10-22 JST。**52図・156出力（各PNG/SVG/PDF）**。
通常signedのA/B/C各10図に加え、B/Cの非相殺診断各10図、D監査図2図を保存した。
対象精度は全図群で0.05 / 0.01 / 0.005 / 0.001 / 0.0005。

## 最初に確認する区別

'''+scope_text['A']+'\n\n'+scope_text['B']+'\n\n'+scope_text['C']+'''

既存の2次PR全方式図10枚と、その生成packetは保全した。旧図を誤りとして置換するものではない。

## スライド検討用の主3精度

| ε | A：保存域を分けた絶対費用 | B：PR内部の相対費用 | C：m5部分集合の相対費用 |
|---:|---|---|---|
'''
for e in ('0.05','0.005','0.0005'):
 readme+=f'| {e} | {links(stem("A","USUAL_SIGNED",e,"absolute"))} | {links(stem("B","USUAL_SIGNED",e,"relative"))} | {links(stem("C","USUAL_SIGNED",e,"relative"))} |\n'
readme+='\n## 全50費用図\n\n'
for group,mode in [('A','USUAL_SIGNED'),('B','USUAL_SIGNED'),('B','NONCANCELLATION'),('C','USUAL_SIGNED'),('C','NONCANCELLATION')]:
 readme+=f'### {group} / {mode}\n\n| ε | 絶対総費用 | 相対費用 |\n|---:|---|---|\n'
 for e in EPS:readme+=f'| {e} | {links(stem(group,mode,e,"absolute"))} | {links(stem(group,mode,e,"relative"))} |\n'
 readme+='\n'
readme+='''## D：カバレッジ・確認状態

| 図 | 出力 |
|---|---|
| 旧方式と高次PRを直接比較できる範囲 | '''+links('D_coverage_legacy_vs_higher')+''' |
| outer32の独立確認状況 | '''+links('D_higher_pr_confirmation_coverage')+''' |

[coverage_audit.json](coverage_audit.json) / [coverage_matrix.csv](coverage_matrix.csv) /
[coverage_audit.md](coverage_audit.md)に、方式・系・精度・mode・scope別600枠を保存。
plotted / saved-but-ineligible / missing / not comparableと、独立確認状態を別columnで記録する。
Aの高次点は旧PFとnot comparableだが、別パネルではplotted。状態の軸を混同しない。

## 条件と保存値の採用

'''+common+'''

既存A系列はH2/H4/H6/H8のDF3/10/19/28、旧登録域・費用政策を継承。
旧PR S2とouter32 PR S2は異なる固定条件の探索値であり、同一のS2対照として混ぜていない。
条件・候補域・各採用点のprefix/q/δ/nと解釈は[figure_notes_ja.md](figure_notes_ja.md)。

固定結果は旧全方式5d536985、新保存値評価8613b3ec、独立確認準備c7c1d822。
取得結果a05f4ed19（H2）/605ca07a6（H4/H6）へ、元SOURCE・hashで追跡できる。
[input_sources.json](input_sources.json)と[sources/](sources/)は実読取Git blobのcommit/path/SHAと同一bytesのコピー。
sources中のMarkdownの相対リンクは原配置を前提とするため、元GitHub URLから読む。

## 描画data・検証

- [figure_data.json](figure_data.json)：420保存表示record、50図の構成と740描画枠。exact fraction・元candidate・SOURCE・record locator・分母を保持。
- [figure_data.csv](figure_data.csv)：740描画枠のflat表。nullは未保存・未結合・不適格の理由付きで、0ではない。
- [group_minima_audit.json](group_minima_audit.json)：保存formula選択の転記と、許可された方式別最小の表示用抽出。
- [compatibility_audit.json](compatibility_audit.json)：四つの入力hash一致と、旧強化PFの完全identity結合が未認定であることを分離。
- [captions_ja.md](captions_ja.md) / [figure_mapping.md](figure_mapping.md)：全52図のcaption・使用域・描画行対応。
- [verification.json](verification.json) / [rendering_checks.json](rendering_checks.json) / [display_check.json](display_check.json)：出典値・保存exact比60件・画像形式・配置・表示確認。
- [provenance.json](provenance.json) / [preservation_check.json](preservation_check.json) / [STOP.json](STOP.json)：来歴、旧成果物保全、終了状態。

新規science sampling / signal / compile / fit / bootstrap / shot生成 / 総費用再会計 / 科学的再ランキング / 再最適化は0。
新たに行った数値操作は、保存totalの表示用最小抽出、分数からfloatへの変換、指定分母での表示用除算のみ。
保存PRの平均・分散・SE・適格性・shotは再計算していない。表示SEを母平均CIに変換していない。
これはCodex自身のローカル技術確認で、独立科学レビュー・外部再現・immutable CI・最終freezeではない。

## 再描画

Python + [requirements.txt](requirements.txt)の専用表示環境で、科学moduleをimportせず再現できる。

```bash
python build_figures.py
python build_figures.py --check
python write_documents.py
```

CPU32・約62.5 GiB RAM・T1000 4 GiBを確認。保存表読取と表示を行い、1図のpreflight後にCPU32 workerで49図、監査2図を出力した。
CPU/RAM/workerの恣意的な低い上限は設けず、GPUはこの描画には使わなかった。speedup倍率は測定していない。
作業用環境は既存の専用`/tmp/pr-all-methods-20261022-venv`、font cacheはこのdirectory内だけ。

## 保存・公開状態と次の担当

ローカル未commit・未push。今回の新しい図群の公開は実施していない。元の公開10図は保持した。
本依頼に残る必須作業はなし。次段階の提案は、**ユーザー**による採用図・公開範囲の指定。
**Codex**は独立確認の新取得や未結合PF比較を始めず、成果物保存・検証後 **mandatory STOP**。
'''
write('README.md',readme)

captions='# 全52図の日本語caption\n\n'+common+'\n\n'
for f in figs:
 captions+=f"## {f['stem']}\n\n[図]({f['stem']}.png)。ε={f['epsilon']}、{f['mode']}、{'絶対総費用' if f['kind']=='absolute' else '相対費用'}。\n\n"
 captions+=scope_text[f['group']]+'\n\n'
 if f['group']=='A':captions+='左はH2/H4/H6/H8、右の保存点はH2/H4/H6。右のPR S2は固定prefix1/3/5・K4・outer32で、左の元採用S2とは別候補域。\n\n'
 if f['group']=='B':captions+='H2/H4/H6のDF3/10/19・prefix1/3/5・K4・outer32、各formulaのq1/2/4/8内保存採用点。\n\n'
 if f['group']=='C':captions+='H4/H6のDF10/19。PRのprefix3/5・K4・outer32、決定論m5の登録prefix域はH4が3/8/9/full10、H6が5/7/11/15/full19、q1/2/4/8。\n\n'
 captions+='全費用軸はlog。線は離散点のguideで、補間・fit・サイズscalingを示さない。小さい固定横offsetは表示上のずらしで、原子数の差ではない。選択効果、有限標本、未観測rare eventを含む母平均の不確かさは残る。\n\n'
captions+='''## D_coverage_legacy_vs_higher

[図](D_coverage_legacy_vs_higher.png)。5精度×H2/H4/H6/H8×要求8系列のカバレッジ。
OK=plotted、I=saved-but-ineligible、M=missing、NC=not comparable。
NCの高次PR値はAの別outer32パネルで表示するが、旧PFに対する直接比較・比は未作成。
H2のε=.01以下のB0/B0+I両次数は登録候補が保存判定で全て不適格。H8の高次PRは未保存。

## D_higher_pr_confirmation_coverage

[図](D_higher_pr_confirmation_coverage.png)。固定c7c1d822時点のouter32独立確認状態。
H2/H4/H6の高次PRは探索取得と保存値評価があるが、独立確認は未実行。H8は探索自体が未保存。
ε=.0005の3比較の提案n_CおよびH4 ε=.005補助案を、取得完了として数えない。
'''
write('captions_ja.md',captions)

notes='''# 読み方・保存範囲・残る注意点

## 研究上の二つの比較

「同じouter32 PR内でS2から4次へ変える効果」と「PRが決定論PFより低費用か」は別の問い。
Bは前者を扱い、Cは後者のうち同じnative系列のm5部分集合だけを扱う。
Aは旧全方式比較と新系列を保存域別に並べた参考overviewで、両系列のS2を同一対照に置き換えていない。

'''+common+'''

## PR高次化の既報値を再現する分母

ε=.0005の報告値は、次の固定条件を持つ**同じouter32系列内**の保存選択どうしの比である。
以下の改善率は保存評価のフィールドを転記し、統計再解析をしていない。

| 系 / DF rank / prefix | 基準 | 高次側 | δ（基準 / 高次） | 保存削減率 |
|---|---|---|---|---:|
| H2 / 3 / 1 | S2/q4 | Yoshida4/q2 | .2 / .4 | 47.122758% |
| H4 / 10 / 3 | S2/q8 | m5_best/q1 | .1 / .8 | 53.303863% |
| H6 / 19 / 5 | S2/q8 | m5_best/q1 | .1 / .8 | 63.123232% |

これらを旧全方式図のS2費用に対する削減率として表示してはいけない。
旧Aはprefix/q/r/Kを含む別の登録候補域から選択され、outer32のS2はprefix1/3/5・K4・固定32政策である。
特にH4の厳しい精度の旧S2採用点はprefix2で、新outer32はprefix3。同じ「PR S2」の文字だけで交換しない。

## 互換性の監査

固定8613b3ecのinput_provenanceは旧強化S2/S4対照を
`PRESERVED_NOT_REPLACED_NOT_IDENTITY_JOINED`、H2の決定論比較を
`NULL_NO_IDENTITY_BOUND_SAME_SERIES_SIGNAL_SET`と明記する。
Hamiltonian/state/snapshot/compiler hashの4項目はH2/H4/H6で旧図と一致するが、
それだけで全native/normalization/shot/実runtime政策の同一性を認定したことにはならない。
今回の互換性監査はその保存metadataを整理したもので、未結合対照の科学的比較を新たに認定していない。

このためAは2パネル。絶対図は左右共通y範囲で値を併置するが、点をパネル間で結んだり、
高次PR/旧PFの比を算出したりしない。相対図は左が旧best full PF、右がouter32 PR S2。
右の1未満はPR内部高次化の点費用減であり、旧PFへの優位性を意味しない。

H2の既存compiler位相帳簿guardや旧PF側のprofile/runtime留保も保持する。
元Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、symmetric_directionalとscalar onceが記録されているが、
同じcompiler SOURCEや実runtimeで一括再実験したとは表現しない。

## Cの決定論候補域

| 系 | m5のmethod | 登録prefix | q | 保存状態 |
|---|---|---|---|---|
| H4 DF10 | B0/B0+I | 3,8,9 | 1,2,4,8 | 24候補。8候補は全対象精度で不適格、残り16候補に単価 |
| H4 DF10 | full | 10 | 1,2,4,8 | 4候補に単価 |
| H6 DF19 | B0/B0+I | 5,7,11,15 | 1,2,4,8 | 32候補。一部不適格、保存適格候補だけ表示 |
| H6 DF19 | full | 19 | 1,2,4,8 | 4候補に単価 |

全64候補中48候補の96軸が直接単価区分（92新軸＋4保存再利用）。残る16候補の単価未保存と不適格は別情報。
全適格行で保存Gが存在することを確認し、未保存単価を補完しない。候補を新たに追加していない。
方式別の最小G抽出は依頼で許可された表示操作だけで、新しい科学的順位表・winner認定を作っていない。
通常signedのm5部分集合最良は、保存summaryの既存候補ID・exact費用と一致する。
この部分集合の結果から、旧D190/強化136/184を含む全PF最良に対するPR優位性を主張しない。

非相殺ではB0/B0IのPF/discard分離成分が未保存。1000行表の280行をMISSING_COMPONENTS・費用nullとして保持する。
C非相殺のfull PF m5は別の部分診断であり、全m5部分集合最良の代替ではない。
同じnative系列のS2決定論集合はこの固定保存評価にないので、Cへ旧S2の値を入れない。

## 表示したouter32採用点

下表の全点は通常signedの新探索標本。全てT=.8/P=0、K4、outer32、tol-only1e-8、上記rank/prefixである。
非相殺の採用候補IDは同じだが整数shot・費用にはroundoff級の差があるため、保存NONCANCELLATION値を別に使用する。

| ε | 系 | formula | candidate | prefix | q | δ | n |
|---:|---|---|---|---:|---:|---|---:|
'''
for e in EPS:
 for s in SYS[:3]:
  for name in ('PR S2','PR Yoshida4','PR m5_best'):
   r=records[k('OUTER32','USUAL_SIGNED',e,s,name)]
   notes+=f"| {e} | {s} | {r['formula']} | `{r['candidate_id']}` | {r['prefix']} | {r['q']} | {r['delta_exact']} | {r['n']} |\n"
notes+='''
各図の高次bestは、Yoshida4とm5_bestの保存formula採用点のうち低い方で、S2を含まない。
同値の候補はexact_tiesへ保持する。他の高次公式を旧pilotから混合しない。
今回固定された現在の36 PR群はS2/Yoshida4/m5_bestだけで、追加の同条件formulaはない。

## 不確かさ・独立確認

保存費用はpaired cosine/sineのG平均。保存SEには同一trajectory内の共分散を含むが、今回再計算していない。
同じ候補の5精度は同じtrajectoryを共有するため、5個の独立な実験ではない。
formula/q選択は同じ探索標本上の点最良で、選択効果補正・母平均CI・rare層上界はない。
全36群でTaylor S4層は未観測。これはouter PFの4次を未実行という意味ではなく、RTE eventの次数層の話である。
S2_multiが1観測の群以外にも未観測層があり、0観測費用・分散を0に補完しない。
保存observed_strata_144.csvのnullを保持した証拠コピーから追える。

c7c1d822の独立確認は準備完了・science_started=false・有効grantなし。
主要3比較の提案n_C、H4補助ε=.005の設計、fixed_shotsは準備情報だけで、費用の取得済み確認結果ではない。
図のnに提案n_Cを使わず、独立確認曲線・CI・新標本設計を作らない。
旧H2/H4/H6/H8の独立確認Pも、新outer32の独立確認の代替にはしていない。

## カバレッジと「作れる範囲」

要求5精度×保存method×適用scopeを全列挙した。A10図、B通常10＋非相殺10、C通常10＋非相殺full-only10、D2図。
scopeを変えた無制限な組合せや未登録q/prefixの補完を「すべて」に含めない。
未作成の比較は次の理由を持つ：旧全方式PFと高次PRの直接overlay/比は未結合、
H2の同一系列決定論対照は未保存、H8outer32は未保存、非相殺B0/B0Iは成分欠測、
同一系列S2決定論対照は未保存・未結合、独立確認は準備段階。
旧H2のε=.01以下の打切りPFは「保存済み不適格」で、未保存と区別する。

statusは表示・比較可能性・独立確認の複数軸を持つ。A高次はnot comparableとplottedの両方が正しいが、
対象はそれぞれ「旧PFと直接比較」と「別outer32パネルでの表示」である。
監査図DのOKはplotted、Iはsaved-but-ineligible、Mはmissing、NCはnot comparable。
全ての理由・source・candidate・費用はcoverage_matrix.csvの600行で追える。

## 保存・検証範囲

原10図・元の主表・SOURCE・STOP・provenanceを変更せず、別directoryへ保存した。
新たに行った演算は、保存fractionの読取・表示用最小抽出・分母が明示された表示用費用比だけ。
新しいmean/SE/shot/総費用・signal・sampling・compile・fit・bootstrap・科学的再ランキングはない。
表示前にexact tokenを元JSONへ照合し、Bの表示比60件を保存exact比へ、C通常の分母を保存m5最良へ照合した。
PNG/SVG/PDFの形式・canvas配置・PDF実renderとPNGのcontact sheetを確認する。
これらは作者Codex自身のローカル技術確認で、独立科学レビュー・immutable CIや最終freezeではない。
'''
write('figure_notes_ja.md',notes)

mapping='''# 図と保存recordの対応

全費用図はfigure_data.jsonのrecords / figures / display_pointsを使用する。
figure_data.csvはdisplay_pointsの同一行をflat化したもの。元sourceは各record.sourceの固定commit/path/SHA/locator。
ratioはdisplay_exact、元totalはcost_exact。nullは理由付きの未表示枠で、0ではない。

| 図base（各PNG/SVG/PDF） | scope | mode | ε | CSV描画行（1起点） | 分母 |
|---|---|---|---:|---|---|
'''
offset=0
for f in figs:
 n=sum(len(p['points']) for p in f['panels']);den=' / '.join(p['denominator'] for p in f['panels']) if f['kind']=='relative' else 'absolute total'
 mapping+=f"| `{f['stem']}` | {f['group']} | {f['mode']} | {f['epsilon']} | {offset+1}–{offset+n} | {den} |\n";offset+=n
mapping+='''
各panel内の行順はseries順、その内部はH-chain順。A左/右はdomainを分ける。
Dの2図はcoverage_audit.json（Aの160枠）および固定readiness情報を使い、費用CSVの行を持たない。
sourceのcommitは保存blobの所在、scientific_SOURCEは既存実装の識別子で別の役割を持つ。
''';write('figure_mapping.md',mapping)

audit='''# 表示・比較・独立確認の監査

600枠の完全な情報はcoverage_matrix.csv / coverage_audit.json。以下はstatusの一覧。
OK=plotted、I=saved-but-ineligible、M=missing、NC=not comparable。
AのNC高次点は別outer32パネルでplottedであり、旧PFとの直接比較・旧PF基準の比を作っていない。
独立確認は別columnで管理し、新outer32はH2/H4/H6ともnot yet independently confirmed。
H8は未保存。旧Aの探索点を示すことは、旧Pの有無や判定を新たに変更するものではない。

'''
code={'plotted':'OK','saved-but-ineligible':'I','missing':'M','not comparable':'NC'}
for scope,mode in [('A','USUAL_SIGNED'),('B','USUAL_SIGNED'),('B','NONCANCELLATION'),('C','USUAL_SIGNED'),('C','NONCANCELLATION')]:
 rr=[r for r in cov if r['scope']==scope and r['mode']==mode];methods=list(dict.fromkeys(r['method'] for r in rr))
 audit+=f'## {scope} / {mode}\n\n| H-chain / ε | '+' | '.join(methods)+' |\n|---|'+'---|'*len(methods)+'\n'
 for e in EPS:
  for s in SYS:
   audit+=f'| {s} / {e} | '+' | '.join(code[next(r for r in rr if r['system']==s and r['epsilon']==e and r['method']==m)['status']] for m in methods)+' |\n'
 audit+='\n'
audit+='''## 未完了の独立確認

固定c7c1d822は実行準備の記録。H2/H4/H6 ε=.0005の主要3比較はn_C提案と固定shotがあるが科学実行前。
H4 ε=.005は補助設計で主要grantから除外。その他精度も独立確認完了を示す保存結果はこの固定資料にない。
提案n_C、pilot、旧n8、旧Pを新outer32の独立確認完了として数えていない。
''';write('coverage_audit.md',audit)
write('requirements.txt','matplotlib==3.10.9\nnumpy==2.2.6\npillow==12.3.0')
write('.gitignore','.mpl-cache/\n__pycache__/')
print('Wrote README/captions/notes/mapping/coverage Markdown/requirements')
