#!/usr/bin/env python3
"""Write display documentation from the already exported figure coordinates."""
from pathlib import Path
from fractions import Fraction as F
import csv
import json

P = Path(__file__).resolve().parent
obj = json.loads((P/'figure_data.json').read_text())
rows = obj['rows']
eps = obj['epsilons']
systems = ('H2', 'H4', 'H6', 'H8')


def write(name, text):
    (P/name).write_text(text.rstrip()+'\n')


def base(e, kind='absolute'):
    return f'{kind}_cost_eps_{e.replace(".", "p")}'


def table(e, kind='absolute'):
    b = base(e, kind)
    return f'| {e} | [PNG]({b}.png) | [SVG]({b}.svg) | [PDF]({b}.pdf) |'


common = '''直鎖H2/H4/H6/H8、隣接1 Å、STO-3G、tol-only 1e-8 DF（実rank 3/10/19/28）、
有限時間複素信号T=0.8、P=0、通常signed bias、経験的allowance倍率1、保存α_axis=0.025。
縦軸はcosine/sine両軸の保存解析shotを含むfull Hadamard-wrapper compiled RZ総費用。
状態準備を含まない。εはこの有限時間複素信号の要求精度であり、基底エネルギー精度や全RPEの精度ではない。
PRは元のA探索n=8の採用点。H2/H4のPFはD70/D190の保存会計、H6/H8のPFは保存済み
ALL_SAVED458直接単価置換政策（登録136/184候補）の保存総費用・保存順位を用いる。
この単価政策には旧profileの実測軸の保持があり、runtime/profileの完全同一性、初回compileのglobal final gate、
S4 composition仮説などの元の留保を継承する。独立確認Pの標本は図に含めない。'''

readme = '''# 全方式をH-chain上で比較する保存総費用図

作成日：2026-10-22 JST。**主図はε=0.05 / 0.005 / 0.0005の絶対総費用3枚**。
全7系列を同じ軸に表示し、0.01 / 0.001も追加資料として保存した。各図PNG / SVG / PDF。
相対費用の補助図も同じ5精度について保存した。計10図・30出力。

## 主図：絶対総費用（log軸）

| ε | PNG | SVG | PDF |
|---:|---|---|---|
'''+ '\n'.join(table(e) for e in eps[:3])+'''

## 追加精度：絶対総費用

| ε | PNG | SVG | PDF |
|---:|---|---|---|
'''+ '\n'.join(table(e) for e in eps[3:])+'''

## 補助図：full PFを基準にした相対費用

分母は同じH-chain・同じεの**表示full PFのS2/S4のうち低い総費用**。
方式ごとに異なる分母を使わず、PRも同じ分母。1の水平線がその基準を示す。
この分母はB0/B0+Iを含む最良PFとは異なる。全補助図は同じlog縦軸範囲。

| ε | PNG | SVG | PDF |
|---:|---|---|---|
'''+ '\n'.join(table(e,'relative') for e in eps)+'''

## 系列・条件・採用規則

7系列：PR、full PF S2/S4、B0 S2/S4、B0+I S2/S4。
PRは青丸、full PFは灰四角、B0は橙三角、B0+Iは緑菱形。PFのS2は実線・塗り、S4は破線・白抜き。
各Hの点に小さい固定横offsetを付けて重なりを避けた。offsetに物理的意味はない。

'''+common+'''

PFの方式・次数ごとの最小値は、保存された適格判定と総費用だけから表示する。
H6/H8は既存のreplacement ordinalを参照し、各系列に属する最初の候補を転記した。
H2/H4は依頼で指定された表示groupごとに、変更していない保存total scalarの最小を抽出した。
同額候補は全てデータに残す。新しい科学的な候補順位表、適格性、会計、PR選択を作っていない。
PRは従来の5精度絶対費用図の探索20点とID・exact totalが全て一致する。

**H2のε=0.01 / 0.005 / 0.001 / 0.0005ではB0・B0+IのS2/S4に適格な保存点がない。**
16枠をnull/空欄とし、0やfull PF aliasで埋めない。140枠中124点を描画する。
H4/H6/H8と全精度のPR/full PFは揃っている。

H6/H8のPFには2026-10-20に既に保存された単価置換版を採用したため、従来のA比較表のPF費用と
一部一致しない。対応と差分は[historical_A_PF_comparison.csv](historical_A_PF_comparison.csv)。
PRはそのままであり、Pの確認標本や旧PR確認n32との混合はない。

## 保存ファイル

- [figure_data.json](figure_data.json) / [CSV](figure_data.csv)：全140枠、exact fraction、null、候補ID、
  rank・L_D・q・δ・r・K、n、cohort、SOURCE・commit・元record locator、費用比の分母。
- [group_selection_audit.json](group_selection_audit.json)：120 PF表示groupの候補件数、保存total、既存ordinal、最小値照合。
- [input_sources.json](input_sources.json) / [sources/](sources/)：実読取Git blobのcommit/path/SHA256と同一bytesのコピー。
- [captions_ja.md](captions_ja.md)、[figure_notes_ja.md](figure_notes_ja.md)、[figure_mapping.md](figure_mapping.md)。
- [provenance.json](provenance.json)：作図checkout、入力・出力hash、保存状態。
- [verification.json](verification.json)、[display_layout_check.json](display_layout_check.json)、
  [display_check.json](display_check.json)、[preservation_check.json](preservation_check.json)：技術・表示・原資料保全確認。
- [STOP.json](STOP.json)：完了後mandatory STOP。後続実行は未許可。

sources中のMarkdownは固定Git blob原文の証拠コピー。原文中の相対リンクは元repositoryの位置を前提とするため、
input_sources.jsonの固定GitHub URLから原文を読む。今回の案内・caption・notesのローカルリンクとは区別する。

## 再描画・照合

科学moduleをimportしない[build_figures.py](build_figures.py)を使用する。科学計算runtimeは不要。
Python + [requirements.txt](requirements.txt)の表示環境だけで、保存済みsourcesから再現できる。
Git blobを取り直す`--freeze-sources`は初回用。通常はコピー済みsourceのSHAを照合して使う。

```bash
python build_figures.py
python build_figures.py --check
python write_documents.py
```

作業時の一時venvは`/tmp/pr-all-methods-20261022-venv`。
依存package・font cacheは専用領域のみ。CPU32、RAM約62.5 GiB、NVIDIA T1000 4 GiBを確認した。
主作業は小さい表の読取とMatplotlib出力で、科学workloadはない。CPU/RAM/worker上限は設けていない。
環境の詳細は[render_environment.json](render_environment.json)。

新signal / sampling / compile / fit / bootstrap / shot再計算 / 総費用再会計 / 科学的再ランキング / 再最適化は0。
PNG目視、SVG構造、PDF実render、データと出典tokenの一致、CSV/JSON一致を確認した。
Codex自身によるローカル技術確認で、独立科学レビューやimmutable CI evidenceではない。

成果物はローカルの未commit保存。今回のcommit・push・外部公開は実施していない。
本依頼の追加作業はなし。ユーザーがスライド/論文への採用や公開を希望する場合、次にその範囲を指定する。
Codexは追加科学計算へ進まず **mandatory STOP**。
'''
write('README.md',readme)

captions = '# 図caption（日本語）\n\n'+common+'\n\n'
for e in eps:
    captions += f'## 絶対総費用：ε={e}\n\n'
    captions += f'**[図]({base(e)}.png)**。要求精度ε={e}における、H2/H4/H6/H8のshot込みcompiled RZ総費用。'
    captions += 'PR、full PF S2/S4、B0 S2/S4、B0+I S2/S4を比較する。PFは保存適格候補の方式・次数別最小総費用、PRは保存された探索A採用点。'
    captions += '縦軸は対数で、絶対図の縦軸範囲は精度ごとに異なる。線は離散点を追うguideで、補間・fit・サイズscalingを示さない。'
    captions += ('H2ではB0/B0+Iの両次数に適格な保存候補がなく、点を描かない。' if e!='0.05' else '全4系・全7系列に保存適格点がある。')
    captions += '各系・各方式の候補域は同一ではない。点間の近い費用差は母平均の統計的優位を意味しない。\n\n'
    captions += f'## 相対総費用：ε={e}\n\n'
    captions += f'**[補助図]({base(e,"relative")}.png)**。上と同じ保存総費用を、同じ系・同じεのfull PF S2/S4の低い方で割った表示用費用比。'
    captions += '全方式に同じ分母を使い、1より小さい点はそのfull PF基準より低費用。分母候補・exact totalはfigure_dataに保存した。'
    captions += '全5補助図の縦軸範囲は共通。欠測・線・横offset・留保は絶対図と同じ。\n\n'
write('captions_ja.md',captions)

notes = '''# 図の条件・読み方・留保

## 何を比較したか

'''+common+'''

full PFは全DF fragmentを保持するB1、B0/B0+Iは保存された決定論的打切り方式として別系列にする。
full-rank B0/B0+Iのaliasを新しく登録しない。PRは全点S2の保存採用系列で、PF側のS4と混同しない。
Gは保存済みの`N_cos C_cos + N_sin C_sin`。この式を図作成時に再会計していない。
費用比は保存total同士の表示用除算だけを行った。n=8は古典MC費用推定の標本数で、解析的な量子測定shot数Nとは別。
実量子shot、ノイズ、FT T count、エネルギー推定の全RPE/QPE round総費用は評価対象ではない。

## 保存入力・候補域

| 系 | DF rank | PF登録域（S2/S4） | q / δ窓 |
|---|---:|---|---|
| H2 | 3 | D70：B0/B0+I p=0,1,2、full p=3、各S2/S4 | q=1,2,4,8,16 / δ=.8,.4,.2,.1,.05 |
| H4 | 10 | D190：B0/B0+I p=1..9、full p=10、各S2/S4 | 同上 |
| H6 | 19 | 強化136：B0/B0+I S2 p=5..15、S4 p=11..15、full p=19 | q=1,2,4,8 / δ=.8,.4,.2,.1 |
| H8 | 28 | 強化184：B0/B0+I S2 p=7..21、S4 p=15..21、full p=28 | 同上 |

PRはH2 A66（p=0,1,2）、H4 A50（p=1,2,3）、H6/H8の新A16（p=3,4,5 / p=6,7,8）からの保存採用点。
疎な登録q/r/K域を直積へ拡張していない。H4の旧DF12をDF10と混ぜていない。
H2/H4のDとAのHamiltonian/state/snapshot/compiler hash、H6/H8の直接価格input bindingと条件表のhash一致を確認した。
state solver・残差gate・profile・実runtime等の全生成規則の一致を認定するものではなく、サイズだけの因果効果やscalingを主張しない。
記録上のcompiler設定はQiskit1.3.0、rz/sx/x/cx、opt1、seed17、topologyなし、symmetric_directional。

### 描画したPR採用点（全て探索n=8）

| ε | 系 / DF rank | 候補ID | L_D | q | δ | r | K |
|---:|---|---|---:|---:|---|---:|---:|
'''
for r in rows:
    if r['series']=='PR':
        notes += f"| {r['epsilon']} | {r['system']} / {r['DF_rank']} | `{r['candidate_id']}` | {r['L_D']} | {r['q']} | {r['delta']} | {r['r']} | {r['K']} |\n"
notes += '''
全PF点のL_D/q/δ/候補ID・exact costはfigure_data.json/CSVに保存する。

## 探索Aと独立確認Pの分離

図のPRはEXPLORATION_N8だけを使用し、Pの固定候補・確認標本・確認費用へ差し替えていない。
H6/H8の後日PF単価置換表に含まれる旧確認n32のPR比較欄も転記していない。
H2/H4のFP n1/n32、H6/H8の旧PR確認cohortとのpoolingはない。
snapshotには別cohortの欄を含む元資料もあるが、使用行はfigure_data.result_sourceのlocatorに限定される。
独立確認Pは両端精度の固定候補の確認であり、全方式・次数の再探索ではない。
そのPにおける方向判定や受入状態は本図から更新しない。

## PF費用政策：元A比較との違い

H6/H8は科学SOURCE `19f2c5a9409db69fd8bd6500eb4228fefa00145c`の既完了70条件表から、
通常signed・倍率1・強化136/184域の既保存replacement総費用とordinalを使った。
表示のためにPF費用を作り直したり、単価置換を再実行したりしていない。
旧mixed measured/model baselineを描く代わりに、既に会計された直接価格置換政策を一貫して使用する。
元Aの比較欄に保存されたPF費用との対照は次のとおり。候補IDは全20 endpointで同じで、費用だけ一部変わる。

| 系 | ε | 元AのPF exact cost | 今回のPF exact cost | 候補 |
|---|---:|---|---|---|
'''
with (P/'historical_A_PF_comparison.csv').open() as f:
    old = list(csv.DictReader(f))
assert all(x['candidate_unchanged']=='True' for x in old)
for x in old:
    if x['cost_unchanged']=='False':
        notes += f"| {x['system']} | {x['epsilon']} | {x['original_A_PF_cost_exact']} | {x['displayed_PF_cost_exact']} | `{x['displayed_PF_id']}` |\n"
notes += '''
元A比較へのこの差はPの標本混合によるものではない。PR20点はID・費用とも従来の探索図から不変。
H2/H4のPF政策はD70/D190直接実測、H6/H8はALL_SAVED458であり、全Hで同一runtimeの一括実験ではない。

## 単価証拠と科学的な限界

ALL_SAVED458は450新軸・8追加既存軸に置換できるところを置換し、残る既存の実測軸を保持した既存政策。
「全候補が新runtimeで再compile済み」とは表現しない。H6/H8 full PFの表示点には旧profileの実測軸の保持がある。
各点のunit_evidenceにreplacement SOURCE、result SHA、旧profile flag、comparison_conditionを付けた。
初回572のglobal final gate欠測、H6 modelとcurrent profileの差、runtime/binary完全同一性未確立、
S4 composition hypothesisなどの元の留保を保持する。これらを図だけで解消したと解釈しない。
全保存候補域の外の最適性、厳密bias保証、母平均の順位、未観測rare eventの費用は未確立。
PRの保存SEをデータへ保持するが、図に新たなCIやbootstrap帯は作らない。点差は有意差検定ではない。

## 主3精度から読める位置関係

以下は指定された保存候補域・費用政策の点推定だけの比較。条件・L_D・δは上表とfigure_dataを参照。

| ε | H2 | H4 | H6 | H8 |
|---:|---|---|---|---|
| 0.05 | PRがfull PFより低い。打切りPFもこの点ではPRを下回らない | PRとB0+I S2が近い | full PFよりPRが低いが、B0/B0+I S2はPRを下回る | PRとB0+I S2が近い |
| 0.005 | PRがfull PFより低い。打切りPFは欠測 | PRが表示PF系列より低い | 同左 | 同左 |
| 0.0005 | PRがfull PFより低い。打切りPFは欠測 | PRが表示PF系列より低い | PRとB0+I S4の差が中間精度より縮む | PRとB0+I S4が近い |

「PRはfull PFに対してだけ有利か」には一律の答えでなく、H6粗精度と他の表示点の違いを見る。
打切りを許すとPFはfull PFから大きく下がり、PRとの差が縮む。粗精度ではS2が安いが、
厳しい精度のfull PFではH4/H6/H8でS4がS2を下回る。一方B0では同じ精度でもS2がS4より安い。
方式を先にまとめてPF次数を隠すと、この関係を失う。

## 表示・欠測・保存

H2のε=0.01以下のB0/B0+I両次数は、登録候補が存在するが全て保存判定で不適格。未計算やゼロ費用を意味しない。
厳密に同じ全rank HamiltonianとなるB0/B0+I aliasをfull PFから追加しない。
線はguideで、欠測をまたぐ推定・補間・fitを行わない。全7系列の凡例は欠測がある図でも保持する。
横offsetは固定の表示上のずらしで、原子数の差ではない。absolute図は精度別ylim、relative図は共通ylim。
30出力を元図とは別名・別directoryに保存し、既存主表・SOURCE・図・manifest・STOPのbytesを保全した。
root synopsisは旧PR-2系列の状況を記述するため、その旧STOPをこの表示作業の科学結果で上書きしない。
今回の保存図は最終evidence freezeや独立科学レビューの代替ではない。
'''
write('figure_notes_ja.md',notes)

mapping = '''# 図・描画行・入力の対応

全図は同じ[figure_data.json](figure_data.json)のrowsから作図する。
絶対図はcost_exact、相対図はrelative_cost_exact。float表示への変換前の有理数tokenを保持した。
各行のresult_sourceで元commit・path・SHA・record locatorへ戻れる。

| 図base名（各PNG/SVG/PDF） | 役割 | ε | data row範囲（1起点） |
|---|---|---:|---|
'''
for i,e in enumerate(eps):
    for kind in ['absolute','relative']:
        role=('主図' if i<3 else '追加精度') if kind=='absolute' else '相対費用補助図'
        mapping+=f'| `{base(e,kind)}` | {role} | {e} | {i*28+1}–{(i+1)*28} |\n'
mapping+='''
各28枠はH2/H4/H6/H8の順で、各系の7枠はPR、full PF S2、full PF S4、B0 S2、B0 S4、B0+I S2、B0+I S4。
欠測はnullの枠をそのまま保持する。source全体の別bias・倍率・cohortを描画していない。
H6/H8のresult_sourceは既保存PF総費用表のreplacement列、saved_rank_sourceは同じ政策の保存ordinal。
provenanceの科学SOURCEは計算実装識別子、commitは保存result blobの所在を示すもので別の役割を持つ。
'''
write('figure_mapping.md',mapping)

write('requirements.txt','''# Display-only environment; versions used for this packet
matplotlib==3.10.9
numpy==2.2.6
pillow==12.3.0
''')
print('Wrote README, captions, notes, mapping, requirements')
