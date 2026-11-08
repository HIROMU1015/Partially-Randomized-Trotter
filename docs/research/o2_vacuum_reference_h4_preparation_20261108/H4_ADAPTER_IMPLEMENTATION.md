# 実H4向け真空参照adapter：技術準備

2026-11-08。実H4への接続コードと人工検証を準備した。保存H4のpath抽出・分類・build・compile・Born・sampling・shotは一切実行していない。既存0edc0fa5の14 PASS／128人工比較を再利用し、そのsuiteは再実行しなかった。

## 実装と再利用

| 新SOURCE | 役割・再利用 |
|---|---|
| vacuum_h4_adapter.py | SavedPRが旧SavedEngineのmetadata復元部分だけを独立化。restore_operations、DFBasisRegistry、保存block/componentの型、selected_circuitとphase関数をそのまま利用。state読込みなし |
| vacuum_h4_numeric.py | 旧interval/certify_gateを再利用し、CSWAP/CX/X/SX/Z/SWAPの正確な小行列を追加。全入力operatorと測定POVMの丸め込み誤差評価 |
| vacuum_h4_execution.py | 既存五ファイルloader・SOURCE検査方式・phaseを保つcompiled_wrapper/cacheを再利用。全方式同じlevel2。価格count/depthは既存rte_compiled_cost helper |
| run_vacuum_h4_limited.py | 明示的な次回科学実行専用の薄いCLI。importでは起動しない。今回未実行 |
| run_vacuum_h4_preparation_artificial.py | 新adapter差分だけの人工runner、SOURCE snapshot／call／失敗／STOPを保存 |

旧constructorをそのまま呼ぶとN2 stateを要求するため、metadata復元だけを分離した。ダミーstateで旧constructorを満たす方法は使わない。旧科学SOURCE・engine・prototype・既存compiler/profileは無変更。SavedPRはprobability／qualify／uniform_inventoryを拒否する。

## 入出力・identity

入力はN1のSOURCE.json、CONFIG.json、LAW.json、POLICY.json、PREPARATION.jsonの既存schema。実行時だけ既存m_n0_metadata_execution.load_fixed_inputsで五Git blobとモデルscopeを検査する。Mの群分け関数や旧科学runnerは呼ばない。

SavedPRはcallerのcontent hashと照合してrecordsをowned copyへ復元し、policy declaration／definition hash／SOURCE／tail／preparation／block順序／basis metadata・matrix hash／lawとcatalogの対応を検査する。未知gate・不正control・非有限角・不正wireはSTOP。descriptorからの実matrix復元は実H4の場合は次回科学callに含まれる。

将来の出力は固定path identity／exact質量／L/R descriptor／split／phase ledger、compiled QPYとlayout、24結果行、局所gate証明、全call journal、SOURCE／INPUTS／STOP。未完了attemptをUNKNOWNで残し、既存STOPと出力衝突ではロード前に拒否する。approval certificate・grantは新設しない。

## Gate順序とsplit

既存selected_circuitのcontrol1 bodyを正本とする。決定論forward block、RTE application、決定論reverse blockをsemantic unitと定義する。総unit数は B=2D+(order+1)。Rは先頭floor(B/2) unit、Lは残り、同距離なら早い境界。gate数やcompiled費用を見てsplitを選び直さない。

deterministic blockの後端と、既存RTE builderの中央application出力直後をsubclassのobserverで記録する。gateを挿入せず、共有basisのfusion/cancellationは元builderに任せる。最後のapplicationには閉じるbasisと蓄積event phaseを含める。途中でGaussian basisのframeが開いていても、各gateが真空固有stateを保つのでlane条件は成立する。逆列とforward列を勝手に隣接消去しない。

traced bodyと元selected wrapper内のgate名・hex角・wire・global phaseが完全一致することを各path／axisで確認する。r=q=1のため複数step境界の再融合は今回対象外。selected fullのsupport_run_le_1規則は維持する。

## Phaseとregister

control1 branchを読み、crz/crzz/czから元のRZ/RZZ/Zを復元し、basis gateは既存列のまま保持する。ancilla Pは順序付きscalar termとしてbinary64の正確なFractionへ保存する。laneのQiskit global_phaseを加算しないため、mod 2piや繰返しfloat和による新誤差を避ける。

scalar-free L0/R0を二laneへ配置し、補正は

\[
\Phi=\ell_0+r_0+\sum_j\phi_j
\]

である。つまり元U=exp(i sum phi_j)L0 R0を保持する。L0†のgate inverseは既存Qiskitを使用し、RZ/RZZの負角とXXPlusYYのtheta符号反転を保持する。RZ/RZZの真空位相は−theta/2、XXPlusYY/Zは0。元Hの真空energyは使わない。

基本wireはtarget0..7、reference8..15、signal16。人工小幅では同形の任意permutationを確認した。reference全wireの初期reset、signal H、二層各n CSWAP、L0†/R0、P(Phi)、X:H／Y:Sdg→H、signal bit0へのmeasure。0→+1、1→−1。postselectionなし、state準備なし、fanoutなし。

## 人工検証

新18件PASS、保存schemaを模した人工2-qubitの2 path×X/Y×3方式×2 topology=24行PASS。n=8のdescriptorについて17 wires／16 CSWAP／8 reset／signal位置だけを検査し、大きなstatevectorの列挙は行っていない。前回14件／128行を作り直していない。

新テストはstate-free復元、既存builderとgate順序一致、noncommuting split、exact scalar ledger、X/Y、reference復帰、wire permutation、identity改変STOP、未知gate、丸め上界のFraction照合、相対位相negative、reset/readout、level2 profile、出力衝突、例外journal/STOPを含む。人工Hamiltonian/arrayは既存EngineConnectionTestsのfixture生成だけを再利用し、旧testsのsuiteは起動しなかった。

開発時の相対import・人工fixture不足field・Fraction文字列変換の不備を修正した。level2の測定最適化でunmeasured isometryが変わる例は、正しい評価対象である測定POVMの一様差へ接続した。失敗SOURCE・ログ・STOPも保全した。許容差を緩めて救済していない。

実保存H4でのgate alphabet・path identity・数値上界・compile資源は未取得。対応不能なgateが実入力で判明すれば、条件を変更せずSTOPする。
