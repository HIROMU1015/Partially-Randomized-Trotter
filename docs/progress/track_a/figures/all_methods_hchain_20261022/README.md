# 全方式をH-chain上で比較する保存総費用図

作成日：2026-10-22 JST。**主図はε=0.05 / 0.005 / 0.0005の絶対総費用3枚**。
全7系列を同じ軸に表示し、0.01 / 0.001も追加資料として保存した。各図PNG / SVG / PDF。
相対費用の補助図も同じ5精度について保存した。計10図・30出力。

## 主図：絶対総費用（log軸）

| ε | PNG | SVG | PDF |
|---:|---|---|---|
| 0.05 | [PNG](absolute_cost_eps_0p05.png) | [SVG](absolute_cost_eps_0p05.svg) | [PDF](absolute_cost_eps_0p05.pdf) |
| 0.005 | [PNG](absolute_cost_eps_0p005.png) | [SVG](absolute_cost_eps_0p005.svg) | [PDF](absolute_cost_eps_0p005.pdf) |
| 0.0005 | [PNG](absolute_cost_eps_0p0005.png) | [SVG](absolute_cost_eps_0p0005.svg) | [PDF](absolute_cost_eps_0p0005.pdf) |

## 追加精度：絶対総費用

| ε | PNG | SVG | PDF |
|---:|---|---|---|
| 0.01 | [PNG](absolute_cost_eps_0p01.png) | [SVG](absolute_cost_eps_0p01.svg) | [PDF](absolute_cost_eps_0p01.pdf) |
| 0.001 | [PNG](absolute_cost_eps_0p001.png) | [SVG](absolute_cost_eps_0p001.svg) | [PDF](absolute_cost_eps_0p001.pdf) |

## 補助図：full PFを基準にした相対費用

分母は同じH-chain・同じεの**表示full PFのS2/S4のうち低い総費用**。
方式ごとに異なる分母を使わず、PRも同じ分母。1の水平線がその基準を示す。
この分母はB0/B0+Iを含む最良PFとは異なる。全補助図は同じlog縦軸範囲。

| ε | PNG | SVG | PDF |
|---:|---|---|---|
| 0.05 | [PNG](relative_cost_eps_0p05.png) | [SVG](relative_cost_eps_0p05.svg) | [PDF](relative_cost_eps_0p05.pdf) |
| 0.005 | [PNG](relative_cost_eps_0p005.png) | [SVG](relative_cost_eps_0p005.svg) | [PDF](relative_cost_eps_0p005.pdf) |
| 0.0005 | [PNG](relative_cost_eps_0p0005.png) | [SVG](relative_cost_eps_0p0005.svg) | [PDF](relative_cost_eps_0p0005.pdf) |
| 0.01 | [PNG](relative_cost_eps_0p01.png) | [SVG](relative_cost_eps_0p01.svg) | [PDF](relative_cost_eps_0p01.pdf) |
| 0.001 | [PNG](relative_cost_eps_0p001.png) | [SVG](relative_cost_eps_0p001.svg) | [PDF](relative_cost_eps_0p001.pdf) |

## 系列・条件・採用規則

7系列：PR、full PF S2/S4、B0 S2/S4、B0+I S2/S4。
PRは青丸、full PFは灰四角、B0は橙三角、B0+Iは緑菱形。PFのS2は実線・塗り、S4は破線・白抜き。
各Hの点に小さい固定横offsetを付けて重なりを避けた。offsetに物理的意味はない。

直鎖H2/H4/H6/H8、隣接1 Å、STO-3G、tol-only 1e-8 DF（実rank 3/10/19/28）、
有限時間複素信号T=0.8、P=0、通常signed bias、経験的allowance倍率1、保存α_axis=0.025。
縦軸はcosine/sine両軸の保存解析shotを含むfull Hadamard-wrapper compiled RZ総費用。
状態準備を含まない。εはこの有限時間複素信号の要求精度であり、基底エネルギー精度や全RPEの精度ではない。
PRは元のA探索n=8の採用点。H2/H4のPFはD70/D190の保存会計、H6/H8のPFは保存済み
ALL_SAVED458直接単価置換政策（登録136/184候補）の保存総費用・保存順位を用いる。
この単価政策には旧profileの実測軸の保持があり、runtime/profileの完全同一性、初回compileのglobal final gate、
S4 composition仮説などの元の留保を継承する。独立確認Pの標本は図に含めない。

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
