# m17の決定論的周辺平均照合：実装準備

2026-11-03 JST。**runner・人工検証・固定実行計画の準備完了。H4科学実行は未実施。**
今回の方針への同意に基づき、[前段の選定計画](../rpe_mean_tolerance_validation_selection_20261103/NEXT_STAGE_DECISION.md)を
実行可能な別SOURCEへ具体化した。保存値の新しい残余・平均・確率を計算していない。

親commit：`be0757472ffa6bed60ecbeef4df27f1985046045`。
branch：`codex/rpe-m17-mean-validation-preparation-20261103`。

## 実装と再利用

[run_m17_mean_validation.py](run_m17_mean_validation.py)は、Partialのm17/q131072/r32/K2だけを対象とする。
linear H4/隣接1.00 Å/STO-3G/保存legacy DF12、8 qubit、4電子sector70、Bのphysical state、L_D3、δ=.02。
新たなsampler・RPE decoder・DF engine・compilerは作っていない。

| 部分 | 再利用・差分 |
|---|---|
| H/state/source照合 | A修正版SOURCE d0b1da1b04c266a1f7abf8e43966ca6e289a8695のbyte確認を再利用 |
| Sector basisとF/R | 同SOURCEのrestricted_basis、numeric_half_operatorsをそのまま再利用。δ=.02固定なので既存halfに適合 |
| DF preparation/config | carrier4591b8d75a2a6a55c74a51225f0c61648023c8eeの既存実装。coefficient_atol=1e-12を明示し、configへr32/K2を渡す |
| Probability側の平均 | samplerと同じcomponent確率再正規化とorder_probabilitiesを用いる短いliteral-law集約を追加 |
| Coefficient側の参照 | tail抽出の保持係数から平均Aを別に構成し、既存finite_rte_operator_momentsを使用 |
| 計算量削減 | 両Aをbasisごとのdiagonalに集約。共通basisは一度だけ変換・cacheし、r/qはmatrix_powerで反復二乗 |
| 保存scalar | 固定1行2軸のN/α/epsilonから統計半幅・残余を求める処理を追加。Decimal80桁、実入力への適用は将来の--runのみ |
| 記録 | 途中段階と終了結果をRESULT.jsonに保存。完了時に7個の小行列をoperators.npzへ保存する |

旧Aの`prepare_context()`はr1固定なので呼ばない。必要な入力照合・basis/half helperだけを再利用する。
元のfinite-RTE validator全体もH/eigensolveを行うため呼ばない。
新しいliteral-law集約が必要なのは、保存meanと同じTaylor関数だけを再呼出しする検証にしないためである。
前段の小さい人工fixture用の式を、実行時の明確なinterfaceへ移した。
一方、二経路はDF metadata・basis・state・F/Rを共有し、その共通部分の誤りを独立に排除する証明にはならない。

## 平均・判定・数値誤差の扱い

Probability側ではA_prob=sum p_i P_iから、orderごとに

    M_law = sum_n pi_n (-1)^(n/2)
                (cos(theta_n) I - i sin(theta_n) A_prob) A_prob^n

を作る。Coefficient側は既存paired-Taylorのattenuated tailを使う。
それぞれ`<psi | (R M_tail F)^q | psi>`を一つずつ求め、両軸はcomplex値のRe/Imから読む。
Probability側のM_tailはM_law^rであり、normalizationをさらに掛けない。

保存μとの二つの差と、二経路間の差について、各complex絶対残差が1e-10以下なら
`NUMERICALLY_CONSISTENT`、超過なら`INCONSISTENT`という**数値再現の診断**を出す。
座標残差、AおよびtailのFrobenius差も記録するが、後二つは診断を通すための追加条件にしない。

RPEへの接続では、軸ごとの残余`epsilon-sqrt(2 log(2/alpha)/N)`と、
そこから観測したlaw対保存μの座標残差を引いた値を記録する。
これは「数値不確かさを置くなら残される余地」であり、真の誤差の上界ではない。
**u_num_boundはnull、RPE_tolerance_certifiedはfalseを維持する。**
根拠のないu_num=1e-10や、二経路の一致を数値誤差上界として採用しない。
科学statusはUNDETERMINEDとし、局所の数値接続という別の診断結果を報告できる形にした。
必要な上界の評価をせずに全18round成功や厳密半径へ昇格させない。

## 人工検証

[test_m17_mean_validation.py](test_m17_mean_validation.py)の13 tests PASS、約0.040秒。
検証記録は[VALIDATION.json](VALIDATION.json)。主実行者による技術テストであり、独立した科学レビューではない。

- 正負の係数・異なる非可換basis、probability重みとcoefficient重みの分離、basis変換の共有。
- 小さい全supportとliteral-law平均の一致、order2と負tau、r内側/q外側の作用順、normalization一回。
- 人工scalarでの残余計算、非有限/不正N・α、非正の残余、数値診断と科学認定の分離。
- default CLIが科学処理やNPZ decodeに入らないこと。実保存行は文字列の転記のみ。
- mock contextでcoefficient_atol=1e-12とr32/K2を確認し、照合失敗時の実値を記録すること。
- 人工218成分・2x2行列でcontext全体のschemaと、共有basisの一度の変換を確認すること。
- 人工中断のRESULT保全・fresh outputの上書き拒否、人工完了時の2平均・7行列の保存。

人工配列は2x2、数値入力は人工scalarのみ。H4 contextはmockし、RNG draw・sampler・量子shotを使っていない。
実H4の数値identity、r32での最終平均、残余の符号や値、1e-10内の再現性は未検証である。

## 性能準備と保全

主要な反復をevent列挙ではなくmatrix powerに置き換え、basisの重複変換を避けた。
人工70x70 matrix powerの12回の小測定では、BLAS1/4/32 threadの観測時間は
約.0213/.00677/.0692秒だった。実H4の性能比較ではなく、4 threadを初期構成とする根拠に限る。
CPU/RAM/GPUの人工上限は設けず、将来の実行時に空き資源と共有負荷を再確認する。
runnerは環境のthread数を強制変更せず、実行コマンドのjob内環境とRESULTに構成を残す。

旧V1/V2/V3/R1、B、A run01/run02、前段設計、library、raw、manifestを変更していない。
新しい科学artifact directoryも作成していない。関係する文書索引だけへ入口を追加する。
M06-F、R2/R3、run02のSTOP、別研究H8を保全する。

次は[固定実行計画](SCIENTIFIC_EXECUTION_PLAN.md)のSOURCEを公開・照合してユーザーへ返す。
この準備の完了から科学実行や重要レビューへ自動移行しない。
