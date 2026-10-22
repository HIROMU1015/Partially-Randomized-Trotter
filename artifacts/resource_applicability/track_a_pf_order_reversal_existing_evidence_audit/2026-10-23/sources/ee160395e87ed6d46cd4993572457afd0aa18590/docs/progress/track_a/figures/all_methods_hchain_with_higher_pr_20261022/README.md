# 高次PRを含む保存済み比較図

作成：2026-10-22 JST。**52図・156出力（各PNG/SVG/PDF）**。
通常signedのA/B/C各10図に加え、B/Cの非相殺診断各10図、D監査図2図を保存した。
対象精度は全図群で0.05 / 0.01 / 0.005 / 0.001 / 0.0005。

## 最初に確認する区別

**Aは旧全方式と新outer32を別パネルに分けた拡張overview**。
旧強化S2/S4 PFとのidentity結合は固定報告で未実施と明記されているため、1軸8系列の直接比較や
高次PR/旧PF比は作らない。左は公開commit 5d536985の7系列・元140枠の値をそのまま表示。
右はouter32 PR S2と、Yoshida4/m5_bestの保存formula bestのうち低い**高次だけ**の値。
high bestにS2を含めない。絶対図は左右のy範囲を共通にして参考併置する。
相対図の分母は左がbest full PF（S2/S4）、右が同じouter32のPR S2で、左右の比を直接比較しない。
図タイトル・軸・caption・auditでこの区別を保持する。H8右パネルは未保存。

**Bは同じouter32系列内のPR内部比較**。
H2/H4/H6についてPR S2 / PR Yoshida4 / PR m5_bestを表示。各formulaの4q内の保存採用点を転記し、
新しい候補順位を作らない。相対費用は同じ系・精度・bias modeのouter32 PR S2を分母にする。
通常signedとNONCANCELLATIONを別図に保存する。

**Cは同じnative系列に結合済みのH4/H6・m5決定論部分集合との比較**。
PR S2 / PR high best / full PF m5 / B0 m5 / B0+I m5を表示する。
このm5はS4の一つであり、旧全方式図のPF S4と同一候補域・同一費用政策の系列ではない。
通常signedの相対図の分母は保存されたm5部分集合内の最良費用。全強化PFの最良とは呼ばない。
非相殺ではB0/B0+IのPF/discard成分が未保存なので点を描かず、full PF m5だけとの診断図とする。
その分母はfull PF m5であり、欠測候補を除外した値をm5部分集合全体の最良として代用していない。
H2は同じidentityで結合した決定論signal集合がなく、H8はouter32未保存のためCのx軸に含めない。
同一系列のS2決定論対照も未保存・未結合であり、旧PFを代入しない。

既存の2次PR全方式図10枚と、その生成packetは保全した。旧図を誤りとして置換するものではない。

## スライド検討用の主3精度

| ε | A：保存域を分けた絶対費用 | B：PR内部の相対費用 | C：m5部分集合の相対費用 |
|---:|---|---|---|
| 0.05 | [PNG](A_usual_signed_absolute_eps_0p05.png) / [SVG](A_usual_signed_absolute_eps_0p05.svg) / [PDF](A_usual_signed_absolute_eps_0p05.pdf) | [PNG](B_usual_signed_relative_eps_0p05.png) / [SVG](B_usual_signed_relative_eps_0p05.svg) / [PDF](B_usual_signed_relative_eps_0p05.pdf) | [PNG](C_usual_signed_relative_eps_0p05.png) / [SVG](C_usual_signed_relative_eps_0p05.svg) / [PDF](C_usual_signed_relative_eps_0p05.pdf) |
| 0.005 | [PNG](A_usual_signed_absolute_eps_0p005.png) / [SVG](A_usual_signed_absolute_eps_0p005.svg) / [PDF](A_usual_signed_absolute_eps_0p005.pdf) | [PNG](B_usual_signed_relative_eps_0p005.png) / [SVG](B_usual_signed_relative_eps_0p005.svg) / [PDF](B_usual_signed_relative_eps_0p005.pdf) | [PNG](C_usual_signed_relative_eps_0p005.png) / [SVG](C_usual_signed_relative_eps_0p005.svg) / [PDF](C_usual_signed_relative_eps_0p005.pdf) |
| 0.0005 | [PNG](A_usual_signed_absolute_eps_0p0005.png) / [SVG](A_usual_signed_absolute_eps_0p0005.svg) / [PDF](A_usual_signed_absolute_eps_0p0005.pdf) | [PNG](B_usual_signed_relative_eps_0p0005.png) / [SVG](B_usual_signed_relative_eps_0p0005.svg) / [PDF](B_usual_signed_relative_eps_0p0005.pdf) | [PNG](C_usual_signed_relative_eps_0p0005.png) / [SVG](C_usual_signed_relative_eps_0p0005.svg) / [PDF](C_usual_signed_relative_eps_0p0005.pdf) |

## 全50費用図

### A / USUAL_SIGNED

| ε | 絶対総費用 | 相対費用 |
|---:|---|---|
| 0.05 | [PNG](A_usual_signed_absolute_eps_0p05.png) / [SVG](A_usual_signed_absolute_eps_0p05.svg) / [PDF](A_usual_signed_absolute_eps_0p05.pdf) | [PNG](A_usual_signed_relative_eps_0p05.png) / [SVG](A_usual_signed_relative_eps_0p05.svg) / [PDF](A_usual_signed_relative_eps_0p05.pdf) |
| 0.01 | [PNG](A_usual_signed_absolute_eps_0p01.png) / [SVG](A_usual_signed_absolute_eps_0p01.svg) / [PDF](A_usual_signed_absolute_eps_0p01.pdf) | [PNG](A_usual_signed_relative_eps_0p01.png) / [SVG](A_usual_signed_relative_eps_0p01.svg) / [PDF](A_usual_signed_relative_eps_0p01.pdf) |
| 0.005 | [PNG](A_usual_signed_absolute_eps_0p005.png) / [SVG](A_usual_signed_absolute_eps_0p005.svg) / [PDF](A_usual_signed_absolute_eps_0p005.pdf) | [PNG](A_usual_signed_relative_eps_0p005.png) / [SVG](A_usual_signed_relative_eps_0p005.svg) / [PDF](A_usual_signed_relative_eps_0p005.pdf) |
| 0.001 | [PNG](A_usual_signed_absolute_eps_0p001.png) / [SVG](A_usual_signed_absolute_eps_0p001.svg) / [PDF](A_usual_signed_absolute_eps_0p001.pdf) | [PNG](A_usual_signed_relative_eps_0p001.png) / [SVG](A_usual_signed_relative_eps_0p001.svg) / [PDF](A_usual_signed_relative_eps_0p001.pdf) |
| 0.0005 | [PNG](A_usual_signed_absolute_eps_0p0005.png) / [SVG](A_usual_signed_absolute_eps_0p0005.svg) / [PDF](A_usual_signed_absolute_eps_0p0005.pdf) | [PNG](A_usual_signed_relative_eps_0p0005.png) / [SVG](A_usual_signed_relative_eps_0p0005.svg) / [PDF](A_usual_signed_relative_eps_0p0005.pdf) |

### B / USUAL_SIGNED

| ε | 絶対総費用 | 相対費用 |
|---:|---|---|
| 0.05 | [PNG](B_usual_signed_absolute_eps_0p05.png) / [SVG](B_usual_signed_absolute_eps_0p05.svg) / [PDF](B_usual_signed_absolute_eps_0p05.pdf) | [PNG](B_usual_signed_relative_eps_0p05.png) / [SVG](B_usual_signed_relative_eps_0p05.svg) / [PDF](B_usual_signed_relative_eps_0p05.pdf) |
| 0.01 | [PNG](B_usual_signed_absolute_eps_0p01.png) / [SVG](B_usual_signed_absolute_eps_0p01.svg) / [PDF](B_usual_signed_absolute_eps_0p01.pdf) | [PNG](B_usual_signed_relative_eps_0p01.png) / [SVG](B_usual_signed_relative_eps_0p01.svg) / [PDF](B_usual_signed_relative_eps_0p01.pdf) |
| 0.005 | [PNG](B_usual_signed_absolute_eps_0p005.png) / [SVG](B_usual_signed_absolute_eps_0p005.svg) / [PDF](B_usual_signed_absolute_eps_0p005.pdf) | [PNG](B_usual_signed_relative_eps_0p005.png) / [SVG](B_usual_signed_relative_eps_0p005.svg) / [PDF](B_usual_signed_relative_eps_0p005.pdf) |
| 0.001 | [PNG](B_usual_signed_absolute_eps_0p001.png) / [SVG](B_usual_signed_absolute_eps_0p001.svg) / [PDF](B_usual_signed_absolute_eps_0p001.pdf) | [PNG](B_usual_signed_relative_eps_0p001.png) / [SVG](B_usual_signed_relative_eps_0p001.svg) / [PDF](B_usual_signed_relative_eps_0p001.pdf) |
| 0.0005 | [PNG](B_usual_signed_absolute_eps_0p0005.png) / [SVG](B_usual_signed_absolute_eps_0p0005.svg) / [PDF](B_usual_signed_absolute_eps_0p0005.pdf) | [PNG](B_usual_signed_relative_eps_0p0005.png) / [SVG](B_usual_signed_relative_eps_0p0005.svg) / [PDF](B_usual_signed_relative_eps_0p0005.pdf) |

### B / NONCANCELLATION

| ε | 絶対総費用 | 相対費用 |
|---:|---|---|
| 0.05 | [PNG](B_noncancellation_absolute_eps_0p05.png) / [SVG](B_noncancellation_absolute_eps_0p05.svg) / [PDF](B_noncancellation_absolute_eps_0p05.pdf) | [PNG](B_noncancellation_relative_eps_0p05.png) / [SVG](B_noncancellation_relative_eps_0p05.svg) / [PDF](B_noncancellation_relative_eps_0p05.pdf) |
| 0.01 | [PNG](B_noncancellation_absolute_eps_0p01.png) / [SVG](B_noncancellation_absolute_eps_0p01.svg) / [PDF](B_noncancellation_absolute_eps_0p01.pdf) | [PNG](B_noncancellation_relative_eps_0p01.png) / [SVG](B_noncancellation_relative_eps_0p01.svg) / [PDF](B_noncancellation_relative_eps_0p01.pdf) |
| 0.005 | [PNG](B_noncancellation_absolute_eps_0p005.png) / [SVG](B_noncancellation_absolute_eps_0p005.svg) / [PDF](B_noncancellation_absolute_eps_0p005.pdf) | [PNG](B_noncancellation_relative_eps_0p005.png) / [SVG](B_noncancellation_relative_eps_0p005.svg) / [PDF](B_noncancellation_relative_eps_0p005.pdf) |
| 0.001 | [PNG](B_noncancellation_absolute_eps_0p001.png) / [SVG](B_noncancellation_absolute_eps_0p001.svg) / [PDF](B_noncancellation_absolute_eps_0p001.pdf) | [PNG](B_noncancellation_relative_eps_0p001.png) / [SVG](B_noncancellation_relative_eps_0p001.svg) / [PDF](B_noncancellation_relative_eps_0p001.pdf) |
| 0.0005 | [PNG](B_noncancellation_absolute_eps_0p0005.png) / [SVG](B_noncancellation_absolute_eps_0p0005.svg) / [PDF](B_noncancellation_absolute_eps_0p0005.pdf) | [PNG](B_noncancellation_relative_eps_0p0005.png) / [SVG](B_noncancellation_relative_eps_0p0005.svg) / [PDF](B_noncancellation_relative_eps_0p0005.pdf) |

### C / USUAL_SIGNED

| ε | 絶対総費用 | 相対費用 |
|---:|---|---|
| 0.05 | [PNG](C_usual_signed_absolute_eps_0p05.png) / [SVG](C_usual_signed_absolute_eps_0p05.svg) / [PDF](C_usual_signed_absolute_eps_0p05.pdf) | [PNG](C_usual_signed_relative_eps_0p05.png) / [SVG](C_usual_signed_relative_eps_0p05.svg) / [PDF](C_usual_signed_relative_eps_0p05.pdf) |
| 0.01 | [PNG](C_usual_signed_absolute_eps_0p01.png) / [SVG](C_usual_signed_absolute_eps_0p01.svg) / [PDF](C_usual_signed_absolute_eps_0p01.pdf) | [PNG](C_usual_signed_relative_eps_0p01.png) / [SVG](C_usual_signed_relative_eps_0p01.svg) / [PDF](C_usual_signed_relative_eps_0p01.pdf) |
| 0.005 | [PNG](C_usual_signed_absolute_eps_0p005.png) / [SVG](C_usual_signed_absolute_eps_0p005.svg) / [PDF](C_usual_signed_absolute_eps_0p005.pdf) | [PNG](C_usual_signed_relative_eps_0p005.png) / [SVG](C_usual_signed_relative_eps_0p005.svg) / [PDF](C_usual_signed_relative_eps_0p005.pdf) |
| 0.001 | [PNG](C_usual_signed_absolute_eps_0p001.png) / [SVG](C_usual_signed_absolute_eps_0p001.svg) / [PDF](C_usual_signed_absolute_eps_0p001.pdf) | [PNG](C_usual_signed_relative_eps_0p001.png) / [SVG](C_usual_signed_relative_eps_0p001.svg) / [PDF](C_usual_signed_relative_eps_0p001.pdf) |
| 0.0005 | [PNG](C_usual_signed_absolute_eps_0p0005.png) / [SVG](C_usual_signed_absolute_eps_0p0005.svg) / [PDF](C_usual_signed_absolute_eps_0p0005.pdf) | [PNG](C_usual_signed_relative_eps_0p0005.png) / [SVG](C_usual_signed_relative_eps_0p0005.svg) / [PDF](C_usual_signed_relative_eps_0p0005.pdf) |

### C / NONCANCELLATION

| ε | 絶対総費用 | 相対費用 |
|---:|---|---|
| 0.05 | [PNG](C_noncancellation_absolute_eps_0p05.png) / [SVG](C_noncancellation_absolute_eps_0p05.svg) / [PDF](C_noncancellation_absolute_eps_0p05.pdf) | [PNG](C_noncancellation_relative_eps_0p05.png) / [SVG](C_noncancellation_relative_eps_0p05.svg) / [PDF](C_noncancellation_relative_eps_0p05.pdf) |
| 0.01 | [PNG](C_noncancellation_absolute_eps_0p01.png) / [SVG](C_noncancellation_absolute_eps_0p01.svg) / [PDF](C_noncancellation_absolute_eps_0p01.pdf) | [PNG](C_noncancellation_relative_eps_0p01.png) / [SVG](C_noncancellation_relative_eps_0p01.svg) / [PDF](C_noncancellation_relative_eps_0p01.pdf) |
| 0.005 | [PNG](C_noncancellation_absolute_eps_0p005.png) / [SVG](C_noncancellation_absolute_eps_0p005.svg) / [PDF](C_noncancellation_absolute_eps_0p005.pdf) | [PNG](C_noncancellation_relative_eps_0p005.png) / [SVG](C_noncancellation_relative_eps_0p005.svg) / [PDF](C_noncancellation_relative_eps_0p005.pdf) |
| 0.001 | [PNG](C_noncancellation_absolute_eps_0p001.png) / [SVG](C_noncancellation_absolute_eps_0p001.svg) / [PDF](C_noncancellation_absolute_eps_0p001.pdf) | [PNG](C_noncancellation_relative_eps_0p001.png) / [SVG](C_noncancellation_relative_eps_0p001.svg) / [PDF](C_noncancellation_relative_eps_0p001.pdf) |
| 0.0005 | [PNG](C_noncancellation_absolute_eps_0p0005.png) / [SVG](C_noncancellation_absolute_eps_0p0005.svg) / [PDF](C_noncancellation_absolute_eps_0p0005.pdf) | [PNG](C_noncancellation_relative_eps_0p0005.png) / [SVG](C_noncancellation_relative_eps_0p0005.svg) / [PDF](C_noncancellation_relative_eps_0p0005.pdf) |

## D：カバレッジ・確認状態

| 図 | 出力 |
|---|---|
| 旧方式と高次PRを直接比較できる範囲 | [PNG](D_coverage_legacy_vs_higher.png) / [SVG](D_coverage_legacy_vs_higher.svg) / [PDF](D_coverage_legacy_vs_higher.pdf) |
| outer32の独立確認状況 | [PNG](D_higher_pr_confirmation_coverage.png) / [SVG](D_higher_pr_confirmation_coverage.svg) / [PDF](D_higher_pr_confirmation_coverage.pdf) |

[coverage_audit.json](coverage_audit.json) / [coverage_matrix.csv](coverage_matrix.csv) /
[coverage_audit.md](coverage_audit.md)に、方式・系・精度・mode・scope別600枠を保存。
plotted / saved-but-ineligible / missing / not comparableと、独立確認状態を別columnで記録する。
Aの高次点は旧PFとnot comparableだが、別パネルではplotted。状態の軸を混同しない。

## 条件と保存値の採用

直鎖H2/H4/H6、隣接1 Å、STO-3G、tol-only1e-8の保存DF rank3/10/19、T=.8/P=0。
新outer32 PRはprefix1/3/5、K4、q=1/2/4/8、δ=.8/.4/.2/.1、各outerの全piece合計32 short steps。
S2・3段Yoshida4・固定binary64 m5_bestを区別する。通常signed・経験的allowance倍率1・α_axis=.025の保存会計を使用。
costは保存解析shot込みのfull cosine/sine Hadamard-wrapper compiled RZ総費用。εは有限時間複素信号の要求精度で、energy精度ではない。
状態準備、実量子shot、noise、FT T count、全RPE/QPE round総費用は含まない。
新探索のnは候補ごとに33〜255で異なる。pilot・旧n8・確認cohortとpoolしない。
独立確認は固定commit c7c1d822で準備段階。提案n_Cを取得済み標本数へ読み替えない。

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
