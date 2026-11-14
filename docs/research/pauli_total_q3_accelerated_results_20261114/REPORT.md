# 保存Pauli LD1/LD2のZ1＋T1/Q3直接モデルMSE診断

承認されたH4/1.00 Å/STO-3G/DF10・Pauli LD1/LD2、各1 control vectorのみ。
固定SOURCE・条件は[INPUTS](INPUTS.json)と[EXECUTION_SCOPE](EXECUTION_SCOPE.md)。新しい探索・MC・価格/Gなし。

| 構成 | status | モデルMSE下界 Ha² | モデルMSE上界 Ha² | 採用方法 |
|---|---|---:|---:|---|
| LD1 | CONDITIONAL_OPT_IN_TOTAL_MODEL_BOUND_SAVED | 4.7246214689726834613461E-7 | 4.7246214689726850477135E-7 | MIN_OF_COMPLETED_VALID_UPPER_BOUNDS:total_state_dp |
| LD2 | CONDITIONAL_OPT_IN_TOTAL_MODEL_BOUND_SAVED | 4.7241534608509197182154E-7 | 4.7241534608509211675289E-7 | MIN_OF_COMPLETED_VALID_UPPER_BOUNDS:total_state_dp |

## LD1

- canonical：`9fa3c47dbdef3b7d08d55cc02c9451cbb400ebbcf807b817db1abb72c5cd946c`。元/新binding：`ea10de7ce5b6a4923bc62719230d551c1eeeecca071264b980b84a75eb549726`。
- bias²区間：['1.0998117425E-8', '1.0998117425E-8'] Ha²。
- variance区間：['4.6146402947E-7', '4.6146402947E-7'] Ha²。
- bias²＋Var整合：INTERVAL_CONSISTENCY_ONLY_DEPENDENT_TERMS。partial区間の端点和を等号で同定しない。
- lemma：TOTAL_ESTIMATOR_FIRST_UNSAFE_LEMMA_BOUND。各first-unsafeの主値guard/capはSUMMARYのlemmaに保存。
- total DP：PARTIAL_TOTAL_STATE_DP_WITH_STATE_RESIDUAL_BOUND。遷移組数 94048712。
- 追跡massは `1 − 9.4156722884616923805329E-24`、残余を捨てていない。
- 解決終端の追跡loss 4.7246214690E-7、残余loss上界 1.5863674231E-22 Ha²。
- any-zero確率 9.9980794237E-8。各roundのzero保持/tie argminのtracked確率はDP.roundsに保存。
- total decoderはzero/tieにもenergyを返す。旧部分decoderの未解決確率診断は別fieldであり、total推定量の未解決eventとして扱わない。
- zero/tieのeventに独立のloss寄与を割り当てて合算する分解は固定kernelにない。tracked event確率・旧部分decoder unresolved boundsとtotal lossを分けて報告する。
- 性能：{"wall_seconds": 234.3204120148439, "cpu_seconds": 221.405227899, "peak_worker_RSS_KiB": 830364, "pid": 2553368, "child_cpu_seconds": 198.006757, "process_tree_cpu_seconds": 419.411984899, "max_single_child_RSS_KiB": 641848, "RSS_scope": "parent peak in peak_worker_RSS_KiB; child peak is maximum individual child, not sum", "sampled_peak_process_tree_RSS_KiB": 5447388}。

## LD2

- canonical：`285bf4812e07a519f266ed88b83293bc4675493c5a9cc03985a6312363c6b8f0`。元/新binding：`f4165e534ad8423fcd32aec689b1a9a7110fe6b59e747fe9d69d82e016d22690`。
- bias²区間：['1.0935533648E-8', '1.0935533648E-8'] Ha²。
- variance区間：['4.6147981244E-7', '4.6147981244E-7'] Ha²。
- bias²＋Var整合：INTERVAL_CONSISTENCY_ONLY_DEPENDENT_TERMS。partial区間の端点和を等号で同定しない。
- lemma：TOTAL_ESTIMATOR_FIRST_UNSAFE_LEMMA_BOUND。各first-unsafeの主値guard/capはSUMMARYのlemmaに保存。
- total DP：PARTIAL_TOTAL_STATE_DP_WITH_STATE_RESIDUAL_BOUND。遷移組数 94048712。
- 追跡massは `1 − 8.3564913537088712865390E-24`、残余を捨てていない。
- 解決終端の追跡loss 4.7241534609E-7、残余loss上界 1.4493135122E-22 Ha²。
- any-zero確率 9.9737950118E-8。各roundのzero保持/tie argminのtracked確率はDP.roundsに保存。
- total decoderはzero/tieにもenergyを返す。旧部分decoderの未解決確率診断は別fieldであり、total推定量の未解決eventとして扱わない。
- zero/tieのeventに独立のloss寄与を割り当てて合算する分解は固定kernelにない。tracked event確率・旧部分decoder unresolved boundsとtotal lossを分けて報告する。
- 性能：{"wall_seconds": 231.5863759170752, "cpu_seconds": 218.515432195, "peak_worker_RSS_KiB": 825176, "pid": 2553369, "child_cpu_seconds": 201.174844, "process_tree_cpu_seconds": 419.690276195, "max_single_child_RSS_KiB": 633664, "RSS_scope": "parent peak in peak_worker_RSS_KiB; child peak is maximum individual child, not sum", "sampled_peak_process_tree_RSS_KiB": 5432984}。

## 資格・限界とSTOP

保存solver energy −2.1663874486347634 Haを条件付きproxyとし、保存ideal finite-K raw・IID・EXACT_AFFINEの下で評価したモデル診断だけである。
元packetのground未認証・実sampler law移送未認証・数値certificateなしを保持。binary64 transition allowanceは未認証engineering allowance。
元Minkowski契約は参照として保全し、PF/finite-Kの位相ずれを含むraw直接riskへAのδEを追加していない。
目標energy精度に対する追加filterを行っていない。control vectorはnominal/十分N/最適Nではない。sufficient_N/global_RMSE_bound/G/G_qualified/winnerはnull。

[SUMMARY](SUMMARY.json)にexact有理数と診断、[run01](run01/)に元record・部分証拠・資源観測を保存。
次：ユーザーが結果に基づく研究方針と追加の有限範囲を判断し、Codex Bは別途明示許可後だけ進める。新しいshot探索・価格/G・正式採択へ自動進行しない。
**mandatory STOP**。

## 高速版の実行来歴

ユーザーが旧run01の停止・部分成果保全と、高速版での2件の再実行を明示許可した。旧runは未完了として別branchに保全しており、完了したことにしていない。

元停止runのcoarse/lemmaと、両構成の全非SOURCE fieldが一致。新backendはexact整数集約・process並列化・遅延約分。

各構成8 worker、2構成並列。CPUはparent＋child合計を別fieldに記録し、RSSはparent peak・最大単独child・2秒サンプリングのprocess-tree合計を区別する。

## zero/tieの確率診断

total推定量の未解決確率は0（各count履歴にenergyを定義）。以下はzero保持・tie argminを採用したtracked massのround別確率で、残余に入った履歴のeventを0扱いするものではない。tracked確率は全law確率の下端であり、追加寄与は当該round前の残余mass以下。全roundのany-zeroは二項周辺分布からのexact値で別途保存。

| m | LD1 zero保持tracked | LD1 tie採用tracked | LD2 zero保持tracked | LD2 tie採用tracked |
|---|---:|---:|---:|---:|
| 0 | 1.136779E-33 | 0.000000E+6 | 1.483513E-37 | 0.000000E+6 |
| 1 | 5.886284E-24 | 9.828776E-27 | 9.837588E-25 | 1.467959E-27 |
| 2 | 5.608595E-18 | 9.272903E-22 | 3.057845E-18 | 4.400747E-22 |
| 3 | 7.353254E-29 | 3.860811E-18 | 2.545064E-29 | 2.140869E-18 |
| 4 | 0.000000E+6 | 0.000000E+6 | 0.000000E+6 | 0.000000E+6 |
| 5 | 0.000000E+6 | 0.000000E+6 | 0.000000E+6 | 0.000000E+6 |
| 6 | 5.594067E-19 | 9.649458E-15 | 5.635477E-19 | 9.849201E-15 |
| 7 | 8.178245E-13 | 2.874209E-15 | 8.120132E-13 | 2.852732E-15 |
| 8 | 2.963382E-12 | 6.773361E-13 | 2.966280E-12 | 6.735913E-13 |
| 9 | 0.000000E+6 | 0.000000E+6 | 0.000000E+6 | 0.000000E+6 |
| 10 | 0.000000E+6 | 0.000000E+6 | 0.000000E+6 | 0.000000E+6 |
| 11 | 0.000000E+6 | 0.000000E+6 | 0.000000E+6 | 0.000000E+6 |
| 12 | 9.997701E-8 | 7.010514E-7 | 9.973417E-8 | 7.010288E-7 |

LD1：旧部分decoderのzero-free tie確率範囲は [7.010521E-7, 7.010521E-7]。元decoderのままなら未解決となる確率の診断であり、新total推定量の未解決確率ではない。

LD2：旧部分decoderのzero-free tie確率範囲は [7.010295E-7, 7.010295E-7]。元decoderのままなら未解決となる確率の診断であり、新total推定量の未解決確率ではない。

RSSのprocess-tree合計は共有pageを重複して数えるresident-sizeの和であり、独立した物理メモリ使用量の保証ではない。サンプル間隔2秒の観測最大。

全13roundの伝播・mass gate・残余会計、MSE上下界、bias²/Var区間の整合確認PASS。旧runのcoarse/lemma 4件もSOURCE以外は全一致。実行gateのUNKNOWN・エラーは0だが、ground/target/state/law移送/数値certificate等の科学的未資格は残る。

## allowanceとlemmaの適用域

LD1：principal guard成立first-unsafe round [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]。保守的L_Dを維持したround [0, 1]。transition excess観測最大 8.049116928532385e-16 rad。宣言allowance `1e-12 rad/transition`は未認証のまま。

LD2：principal guard成立first-unsafe round [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]。保守的L_Dを維持したround [0, 1]。transition excess観測最大 8.049116928532385e-16 rad。宣言allowance `1e-12 rad/transition`は未認証のまま。

表示の丸めでMSE上下界を同一値とみなさないため、表は多桁表示。exact有理数はSUMMARY/rawを正本とする。
