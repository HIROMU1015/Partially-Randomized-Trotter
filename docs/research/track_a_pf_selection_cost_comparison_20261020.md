# qモデル単価と直接実測単価によるPF選択・費用順位の比較

2026-10-20。基点`f63065fb636504b78d295f351ffc339370b9a532`。
ユーザーが`ALL_SAVED458`の入力準備済み70件（scope重複を除く30条件）について、保存shot込み総費用再会計・PF再選択・PR/PF比・順位比較を明示承認した。
実行SOURCEは`19f2c5a9409db69fd8bd6500eb4228fefa00145c`。

**70件中、最良PF候補は2件で変わった。qとPF次数の変更、PR/PFの優劣逆転はともに0件だった。**
最良PFの総RZ費用は30件で変わり、候補全体の順位には変化があった。これは保存された過去の混在政策と直接実測置換の条件付き比較であり、モデル固有誤差とprofile/runtime差を分離した検証ではない。

## 実行範囲と条件

linear H6/H8、隣接1 Å、STO-3G、tol-only DF19/28、T=.8、P=0、symmetric_directional。状態準備を除くfull Hadamard wrapperのcompiled RZに、保存cosine/sine shotを掛けた総RZを用いる。登録済みB0/B0I/full PF、S2/S4、prefix、q=1/2/4/8、bias適格性、allowance倍率1/10、候補集合を保持した。

| 系列・bias | scope | 実行件数 |
| --- | --- | ---: |
| H6 signed | old32 / fine S2 96 / S2+truncated S4 136 | 各10、計30 |
| H8 signed | old32 / strengthened184 | 各10、計20 |
| H8 empirical noncancellation | old32 / strengthened184 | 各10、計20 |

各10件はε=.05/.01/.005/.001/.0005×倍率1/10。70はscope重複を含む。独立した70実験や70分子条件とは数えず、system/bias/ε/factorの一意条件は30である。

基準側は元の**実測とqモデル推定が混在する保存政策**。全候補を純粋モデルで再予測した比較ではない。H6の基準snapshot SOURCEは2261a3ad、H8はsix-axis置換後のaa984d98。PR対照はH6確認union8/H8確認union6の各n32で、保存された選択・費用・SE・cohortをそのまま保持した。探索n8や論文主表の独立P n48/n104を混合していない。

## 基準再現とone-shot実行

人工13件PASS後に本番SOURCEをcommitし、新しい契約・grantへ70 condition ID、入力SHA、SOURCE8 file hash、Python/算術stdlib hash、候補集合と禁止callをsealした。新しいone-shot marker/ledger/終了証拠を用い、旧grant/marker/STOPは変更していない。

同じ候補/endpointの費用をscope間で共有し、PF候補条件5,040行を一度処理した。適格PF2,594件について、

`G = N_cosine × saved_RZ_cosine + N_sine × saved_RZ_sine`

をFractionで計算し、保存総費用と完全一致した。70件すべてで元の最良PF・exact ties、保存PR対照/費用/SE、元の比または削減率を照合した。**この基準gateが完全にPASSした後だけ**直接価格への置換を開始した。保存値修正・retry・shot再計算は0。

モデル単価450軸＋過去の公開実測8軸を置換し、元から実測だった102軸を保持した。全560単価が少なくとも一つの承認適格行で使用された。保存単価がない80 slotは不適格行でnullのままで、0補完や追加モデル評価をしていない。

候補費用は有理数、選択は費用→候補ID辞書順、exact tiesは全候補IDを別保存。PR/PF比が1未満ならPR_LOWER、1超ならPF_LOWER、等しければTIEとした。保存PR費用・SEを再計算せず、新しいPR候補も選択していない。

## 最良PF・q・総費用・PR/PFの変化

| 項目 | 70件中の変更件数 |
| --- | ---: |
| 最良PF候補 | 2 |
| prefix | 2 |
| q | 0 |
| PF次数 | 0 |
| 最良PFの総RZ費用 | 30 |
| PR/PF優劣の方向 | 0 |
| 適格PFの辞書順tie込み全順序 | 36 |

最良費用の変化率`100×(置換後の最小費用/基準側の最小費用−1)`は−0.221699%～+0.327685%。勝者が異なる2件では異なる候補の最小費用同士を比べている。

PR/PFの方向は、基準側・置換側ともPR低費用60件、PF低費用10件、同率0件。これは重複scopeを含む保存n32の点比較の内訳であり、PR母平均の優位確率ではない。

### 最良PFが変わった2件

両方ともH8、empirical noncancellation、ε=.0005、strengthened184。倍率1と10で、**B0I p16/q4/S4 → B0I p15/q4/S4**に変わった。q=4、S4、PF低費用という方向は維持した。

| 倍率 | 基準PF総RZ | 置換後PF総RZ | 最小費用変化 | PR/PF比の変化 |
| --- | ---: | ---: | ---: | --- |
| 1 | `770296671316160/7` | `110353805016120` | +0.283003% | 1.290910077 → 1.287267071 |
| 10 | `770305822651840/7` | `110355077430428` | +0.282968% | 1.290921013 → 1.287278427 |

倍率1では、p15の費用110,353,805,016,120は元から実測で不変だった。p16はモデル費用`770296671316160/7`から、直接実測置換後110,495,335,086,560へ変わり、p15を上回った。小さい単価差でも近い候補の順位は変わり得るという保存値の例になった。モデル差と環境差の原因分離はしていない。

### H6 ε=.005の指定候補

fine S2 / S2+S4の両scopeで、最良候補は`B0_p11_q4_2nd`のまま。倍率1では総費用`782937954816/5`→`156742257152`、PR/PF比0.650347442→0.649705708。倍率10でも候補・q・PF次数は同じで、PR/PF比0.650347769→0.649706035。いずれもPR低費用の方向を維持した。旧論文主表や保存会計は上書きしていない。

## 候補全体の順位と同率

| 系列・bias・scope | 最良候補変更 | 最良費用変更 | 順序変更条件 | 厳密な候補pair逆転 | tie関係変化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| H6 signed old32 | 0 | 0 | 0 | 0 | 0 |
| H6 signed fine S2 96 | 0 | 10 | 8 | 36 | 0 |
| H6 signed S2+S4 136 | 0 | 10 | 10 | 50 | 8 |
| H8 signed old32 | 0 | 0 | 0 | 0 | 0 |
| H8 signed strengthened184 | 0 | 2 | 8 | 28 | 0 |
| H8 noncancellation old32 | 0 | 0 | 0 | 0 | 0 |
| H8 noncancellation strengthened184 | 2 | 8 | 10 | 32 | 0 |
| 計 | 2 | 30 | 36 | 146 | 8 |

全139,834候補pair関係を確認し、厳密な大小の逆転146組、tieへの出入り8組を保存した。scopeと倍率をまたぐ重複があり、146個の独立現象とは解釈しない。tie関係8組はH6 S2+S4、ε=.05の倍率1/10に各4組。最良PF自体に新しい同率最小値は発生せず、最良tie候補集合のIDが変わった2件は上記H8の勝者変更による。

## 成果物・図・検証

[結果packet](../../artifacts/resource_applicability/track_a_pf_selection_cost_comparison/2026-10-20/README.md)にCSV/JSON、[全70件の表](../../artifacts/resource_applicability/track_a_pf_selection_cost_comparison/2026-10-20/conditions70.md)、2図×PNG/SVG/PDFを保存した。

- `candidate_totals`：5,040候補条件行。保存shot、基準再現費用、置換費用、出典locator/結果SHAへの結合。
- `selection_comparison`：70件の最良候補/q/prefix/PF/総費用、固定PR費用/SE/cohort、比、変更flag。
- `candidate_ranks` / `rank_relation_changes`：全適格候補の順序/tie rank、大小逆転とtie変化の明細。
- `used_price_evidence`：実際に使用した450＋8＋102単価と継承条件。
- `excluded_unresolved30`：未解決30件をそのまま保存。数値解析は実行していない。

`PF_cost_change`は全70件の選択最小費用の散布図と変化率。`PR_PF_ratios`は7つのsystem/bias/scope groupに全70件を表示し、基準側と置換側、倍率1/10、比1の境界を示す。図座標と英語captionは`figure_data.json`に保存した。線は保存条件の点の接続で、fitではない。

数値監査では全2,594適格費用を別の整数cross-productとDecimal80桁で検算した。整数比の比較器で70scopeの最小値・exact ties・順位を再確認し、139,834 pairの関係と保存差分表を照合した。PR費用/SE/cohortの完全保持、未解決30件の未実行、全入力/SOURCE/runtime SHA、全図への70件の一意な割当、6出力形式のheader/SHAがPASSした。新13人工検査もPASS。著者であるprimary agentの検査であり、独立した詳細科学レビューではない。

| 主結果 | SHA256 |
| --- | --- |
| selection_comparison.json | `13d96d8a0688896e8cbdcdb70dd8a2cea494a4e89f785d1f6b5ff5d25e8cf2aa` |
| selection_comparison.csv | `fd5eb243788c6be6108a6ca0ee6e20e4b08a0aee80a55a6413c0c883d2aa526d` |
| candidate_totals.json | `c0cc55eb66f883b2b0eea5d83e3081bb62fe5350d6e2709df6c38e2007d336b3` |
| rank_relation_changes.json | `da27378e125f3855a736a6aa84ed4257359697ca1ae1f269a73175e8f665d4b7` |

科学workerはnative exit0、起動親がwait4して回収した。parent/child、stdout/stderr SHA、CPU/RSSを`native_exit_evidence.json`に保存。数値stage wall3.419秒、起動/前後gate等を含むchild wall3.860秒、child CPU3.890秒、peak RSS117,944,320 bytes（約112.5 MiB）。現在科学worker0。CPU/RAM/workerの新たな恣意的上限、共有環境/OS/他ユーザーの変更は行っていない。

## 継承する限界と終了

H6のprofile差、H8等のmodel/runtime同一性未証明、S4合成仮説、初回572軸のrun全体最終gate欠測を保持する。使用新450軸のうち428軸が初回SOURCEに由来する。旧profile待ち48軸も元実測のflagを保全し、新規取得・同一性認定をしていない。結果をモデル単独の誤差や厳密な費用優位保証にしない。

対象は上記の公開過去会計snapshotであり、最新Dの非公開原本や論文主表の独立確認Pそのものを再評価していない。70scopeでPR/PF逆転0という結果を未比較条件へ一般化しない。単価置換で一部候補順位が変わったことと、保存PR/PF優劣の方向が変わらなかったことを区別する。

H6 noncancellationの未解決30件は未実行。新compile/sampling/モデル予測/refit/shot再計算/PR統計再計算/候補追加はすべて0。旧SOURCE/raw/grant/ledger/marker/STOP、論文主表/Figure 2を保全。詳細科学レビュー・最終evidence freezeは未開始。

新しい成果物の公開は未承認で専用branchへのlocal commitのみ。次担当**ユーザー**が今回の公開範囲を判断する。未解決30件や追加解析を進めたい場合は、欠測を解消する方法と科学実行範囲の別承認が必要。**Codex**は新たな明示承認の範囲以外へ進まず、**mandatory STOP**。
