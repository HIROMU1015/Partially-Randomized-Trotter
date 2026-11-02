# 現行最良S2 PR/PF：選択確定・compiled系列図・理論Gの入力不足

**現行理論Gの取得は0/20条件。理論／compiledの2系列比較および図Bは未完成。**
今回の実行承認に基づき、H4・ε=.001のS2 PF選択を確定し、40人工テスト・17保存理論G回帰照合・20compiled費用比の表示を実行した。
新しい理論計数を実行するための元preparation・Tail法則・shot原表を、この環境では取得できなかった。
ユーザーからも「そのパスはこの環境にはない」と確認されたため、値を推定せず、取得できた部分を保存した。

## 保存した図

|ε|保存した図Aのcompiled系列のみ|
|---|---|
|0.05|[PNG](A_compiled_only_eps_0p05.png) / [SVG](A_compiled_only_eps_0p05.svg) / [PDF](A_compiled_only_eps_0p05.pdf)|
|0.01|[PNG](A_compiled_only_eps_0p01.png) / [SVG](A_compiled_only_eps_0p01.svg) / [PDF](A_compiled_only_eps_0p01.pdf)|
|0.005|[PNG](A_compiled_only_eps_0p005.png) / [SVG](A_compiled_only_eps_0p005.svg) / [PDF](A_compiled_only_eps_0p005.pdf)|
|0.001|[PNG](A_compiled_only_eps_0p001.png) / [SVG](A_compiled_only_eps_0p001.svg) / [PDF](A_compiled_only_eps_0p001.pdf)|
|0.0005|[PNG](A_compiled_only_eps_0p0005.png) / [SVG](A_compiled_only_eps_0p0005.svg) / [PDF](A_compiled_only_eps_0p0005.pdf)|

図内タイトルは**Compiled only**、系列はCompiledのみ。要求された図Aの片系列に当たる補助出力であり、
Theory/Compiled比較が完成した5図とは数えない。図Bの削減率は分母の理論Gがないので作成していない。
全5図×PNG/SVG/PDFの15出力。旧H4 DF12の参考図や既存スライドを変更していない。

## 固定条件と候補

現行H2/H4/H6/H8、linear chain・隣接1 Å・STO-3G・tol-only1e-8 DF rank3/10/19/28、
T=.8、P=0、通常signed bias、numerical allowance倍率1、cosine/sine両軸のfull Hadamard wrapper総RZ費用。
PR外側PFはS2で、PRは従来H-chain図の探索A n8を固定。高次PR・outer32別subdivision・旧確認n32を混入しない。
controlはsymmetric-directional、scalarはoriginal_plus_tail_once、reference_rotated=false。
保存compilerはQiskit1.3.0、rz/sx/x/cx、opt1、seed17、topologyなし。
各候補のL_D・q・δ=.8/q・r・K、SOURCE・Hamiltonian/state/snapshot/compiler hashは[figure_data.json](figure_data.json)に転記した。
固定時間信号精度の会計で、長RPE全round総費用やFT Toffoli資源ではない。PF係数Cのfit/delta窓を新規評価していない。

**H4・ε=.001の最良S2 PF：B0_p7_q8_2nd、compiled G=1,666,873,406,028。**
保存適格S2の21候補をFull/B0/B0+Iを含めて比較し、exact G最小・同点はID順という既存規則を適用。
保存shotはcosine=22,035,273、sine=18,163,285。新しいshot計算・回路計算なし。
他19条件は前回固定した選択を維持し、理論側で候補を選び直していない。
[候補選択監査](candidate_selection_audit.json)・[20条件表](coverage_matrix.md)・[CSV](coverage_matrix.csv)。

## 理論Gを取得できない理由

既存native counterと、NATIVE_MIXED / explicit_rz_sx_x_cx_v1の数学的定義は確認できた。
元counterのbytesは固定コピーで保全し、技術adapterと人工検証を追加した。
ただし、現行物理系の順序付きGaussian基底操作、対角係数、完全なPR Tail法則・元scalarの結合情報を、
保存hashだけから復元することはできない。記録上の元保存先`/home/AbeHiromu/projects/...`はこの環境にない。
公開H6/H8新探索A CSVのREADMEもshot原表は公開していない旨を明記している。

H4 legacy DF12、H6旧p10/p15、H6/H8旧n8/n32からの代用はしていない。
同じcandidate IDやcompiler hashだけでは、元の回路metadataが一致したことにはならない。
PFのcompiled単価モデル、ALL_SAVED458価格、compiled有限MC平均をnative理論単価の代用品にしない。
新しいHamiltonian/DFや基底を生成して埋めることも、今回の承認外。
[不足入力27候補の一覧](required_inputs.md)・[counter監査](theory_counter_audit.json)・[coverage監査](coverage_audit.json)。

## 実行・検証の範囲

新execution namespace：`artifacts/resource_applicability/track_a_current_s2_theory_compiled/2026-10-22/run01`。
新SOURCE `e7838a8c1a816ba39b4a2e9e4a9488f56cd746a5` をローカルcommitで固定し、新plan・grant・one-shot marker・ledgerの下で実行。
使用済みgrantの再利用、既存STOPの解除、科学retryは0。

|実行内容|件数・状態|
|---|---|
|H4 ε=.001のS2 PF選択|1条件、既存適格21候補、完了|
|元native counter人工検証|27 tests PASS|
|新adapter人工検証|13 tests PASS|
|保存理論単価×保存shotによる総費用回帰|旧固定S2の17条件、exact Fraction一致|
|新しい現行理論counter評価／理論G|0、元入力未取得|
|現行compiled比表示|20条件|
|図Aのcompiled系列だけの補助出力|5図、15形式出力|
|図Aの2系列完成／図B|0／0|
|sampling・signal・compile・Hamiltonian/DF再生成・独立確認|全て0|

旧固定17条件の回帰は保存単価と保存shotから保存Gを再現するチェック。
元preparationから理論counterを実再評価した検証や、現行20条件の理論G取得と称さない。
PRの理論法則期待値とcompiled有限MC平均は異なる量であり、全差をcompiler最適化効果と呼ばない。
検証PASSは技術処理の範囲で、科学的結論や現行理論モデル適用の証明ではない。

## 図から読めることと未判定点

compiledではε=.05のH6がPR/PF≈1.768でPF低費用、残り19点はPR/PF<1。
この数値は登録集合・探索A n8・保存compiled価格政策に条件付きの点費用で、母平均・独立確認の結論ではない。
H6/H8のPF価格はALL_SAVED458の既存条件付き値で、profile/runtime同一性未証明などの留保を継承する。
理論側がないため、仮説A「理論段階からPFが競争力を持ったか」と仮説B「PFの理論→compiled低下が大きかったか」は、
今回の現行比較ではどちらも判定できない。

## ファイル・再現・公開状態

[figure_data JSON](figure_data.json) / [CSV](figure_data.csv)・[input sources](input_sources.json)・
[caption](captions_ja.md)・[notes](figure_notes_ja.md)・[verification](verification.json)・
[display check](display_check.json)・[provenance](provenance.json)・[STOP](STOP.json)。
原科学出力は新namespaceのoutput/へ保存し、図はそこから読み出す。原SOURCE/raw/result/grant/ledger/marker/STOP、
旧図・スライド・論文本文は保全。結果発見用の索引と現状文書だけに追記する。

```bash
python build_figures.py --check
python verify_packet.py --check
```

SOURCEのrun commandは消費済みone-shotであり、再実行しない。作図の再生成は科学runnerを呼ばない。
branch `codex/pr-pf-s2-theory-compiled-main-20261022`、ローカルcommitまで。push・公開は今回未承認なので実施しない。
科学的詳細レビュー・新stage・最終evidence freezeには進まない。

次段は、ユーザーが元保存入力を保持する環境・取得先を指定し、Codexが同じhash・候補・cohortのmetadataを照合して不足する理論Gを評価する段階。
今回その入力を取得できないため、自動再生成・別cohort代用・追加探索はせず**mandatory STOP**。
