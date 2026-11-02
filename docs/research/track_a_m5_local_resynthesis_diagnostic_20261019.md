# m5 compiler：Euler位相生成・角度簡約の局所人工診断

2026-10-19。**角度簡約が小誤差を生み、外側の位相加算を正確にしても残る経路を、固定runtimeの人工例で再現した。H6でこの経路が実際に発生したかは未検証で、H6の原因確定・修正完了ではない。** 利用者の「ではその検証を行って」に基づく局所検証は完了し、local commit後mandatory STOP。実H6の新compile・回路作用・費用受理は0、pushは行わない。

SOURCE：`057d954b16924b796ab2780727bfddc663a7635d`。branch：`track-a-m5-local-resynthesis-diagnostic-20261019`。[直前のH6結果](track_a_h6_m5_phase_batch_run01_results_20261019.md)のraw基準超過STOPは変更しない。

資料入口：[summary](../../artifacts/resource_applicability/track_a_m5_local_resynthesis_diagnostic/2026-10-19/summary.json)、[全人工出力](../../artifacts/resource_applicability/track_a_m5_local_resynthesis_diagnostic/2026-10-19/diagnostic.json)、[SOURCE/runtime結合](../../artifacts/resource_applicability/track_a_m5_local_resynthesis_diagnostic/2026-10-19/source_binding.json)、[構造検証](../../artifacts/resource_applicability/track_a_m5_local_resynthesis_diagnostic/2026-10-19/structure_validation.json)。

## 1. 対象と分離方法

分子入力を使わない1qubitの人工回路・行列だけを使用した。H4/H6/H8、PF係数、geometry、DF、状態、T、shot、費用ランキングの新評価ではない。H6に接続する背景条件は前報のH6直鎖1 Å/STO-3G/tol-only1e-8・保存rank19/full m5_best p19/q2/T=.8/P=0/cosineだが、そのデータを今回の入力にはしていない。

固定Python3.12.3、Qiskit1.3.0、NumPy1.26.4、mpmath1.3.0、SymEngine0.13.0を再利用し、共有環境を変更していない。Qiskit公式1.3.0の保存SOURCE11fileをSHA照合した。release commitは`896d1c8b72888a52a46081a6d1f4de8c56db8aea`であり、これはinstalled Rust binaryのbuild証明ではない。

比較を次の四つに分離した。

1. **Euler抽出**：同じbinary64行列に対して、installed `params_zyz`の出力と、同じ式を80桁で計算した出力を、global phaseを含む行列で比較する。非厳密unitaryとなる入力丸めも別欄へ残す。
2. **内部scalar算術**：`params_u1x`のscalar変換と、ZSXX生成中のphase加減算を調べる。後者は元の角度・分岐・signed binary64増分を保ったまま、増分の和だけを有理数で計算した人工対照と比較する。
3. **角度簡約**：同一人工入力を既定`True/None`、`True/atol=0`、`simplify=False`で生成し比較する。これらは診断用の呼出しであり、production compiler条件の変更ではない。
4. **gate行列・積**：人工3パターン×16/256/4096 gateで、binary64 gate行列の丸め、明示的binary64積、高精度積、実際の最適化passの出力を別々に比較する。明示的積の式はRust SOURCEに合わせたが、compiled Rustとのbit同一性は主張しない。

生成部は一般角、zero、小角、π/2・π・wrap端点・axis-flip付近の45入力×3設定＝135件。各入力のbinary64 hexを保存している。Euler抽出も同じ45入力で検査した。有限の狙いを定めたprobeであり、全入力の誤差上界・全格子網羅ではない。

参照行列は数式をmpmathで評価し、入力floatを正確なbinary64値として読み取る。45参照と9個の積について80/120桁の内部整合を確認した。高精度数値参照は厳密certificateではない。全比較はraw行列または指定basis状態の差で、phase alignment・事後正規化を使わない。

## 2. 小角度簡約の独立した反例

角度`ε=2^-40=9.094947017729282e-13`のRZを用いた。ZSXXへ直接渡す人工入力は`theta=0, phi=0, lambda=ε, phase=-ε/2`であり、数式上の対象はRZ(ε)となる。

| 生成条件 | raw Frobenius差 | 意味 |
| --- | ---: | --- |
| 既定簡約 | 9.094947017729282e-13 | 出力gateなし、scalarだけ残る |
| 位相増分の和だけを正確にした人工対照 | 9.094947017729282e-13 | 加算誤差が0でも簡約の差は残る |
| atol=0 | 2.872051323469032e-16 | 同じ入力の診断用対照 |
| simplify=False | 2.872051323469032e-16 | 同じ入力の診断用対照 |

同じ行列に対するEuler抽出誤差は1.4622625510057203e-25であり、この例の約9.1e-13をEuler抽出が生んだものとはしない。固定SOURCEではZSXXのRZが`abs(angle)>atol`の場合にのみ出力され、その場合にだけ`phase += angle/2`が行われる。既定の局所簡約が意図した近似を、長い回路全体のraw基準と区別する必要がある。

さらに、barrierで局所runを分け、各runに同じRZを置いて**実際のaccelerated最適化pass＋前報の正確な外側phase batch**を実行した。

| 独立局所run数 | 出力RZ数 | raw差：basis状態1 |
| ---: | ---: | ---: |
| 1 | 0 | 9.097396311327578e-13 |
| 16 | 0 | 1.455216015772668e-11 |
| 256 | 0 | 2.3283088858322944e-10 |
| 4096 | 0 | 3.725290543391274e-9 |

この人工反復は、外側の位相加算だけを正確にしても、局所近似の整合した積み重なりを消せない例である。直接生成との差には、DAGへnumeric phaseを戻す際のwrap等も含まれる。barrierや反復数を実H6から取得しておらず、H6の誤差量・発生頻度・状態依存性の説明には使わない。上表のbasis状態はH6保存状態ではない。

## 3. 位相生成・丸めで確認した範囲

135生成結果について、診断用のSOURCE mirrorのgate角度とphaseがinstalled kernelのbit列に一致した。その上で、元のphase増分を全て保持した有理数和との差を記録した。**人工の和だけの対照であり、実回路への補正ではない。**

| 局所処理 | 今回の観測値 |
| --- | ---: |
| 45入力：binary64 Euler再構成 vs 同じ入力行列のraw差・最大 | 3.607101124546594e-16 |
| 45入力：U1X scalar変換の丸め・最大絶対値 | 1.1102230246251565e-16 rad |
| 135生成：内部phase加減算と同じ増分の正確な和との差・最大絶対値 | 4.718447854656915e-16 rad |
| 135生成のうち既定45件：raw差・最大 | 9.097951826421707e-13 |
| 簡約を無効にした45件：raw差・最大 | 8.015221223404592e-16 |

`mod_2pi`の`(angle+π) mod 2π-π`は、人工角`2^-55`を0へ丸めた。`simplify=False`でもこのwrap自体は実行されるため、簡約を無効にすれば全丸めが解消するとは言わない。また、内部scalar加減算の正確化だけでは、角度計算・π定数・Euler計算の全誤差は消えない。

9個の積probeでは、明示的binary64積と同じgate行列の高精度積との差は最大約1.0013e-13、gate行列自体の丸めを積み重ねた差は最大約1.0019e-13だった。一方、実際の正確な外側phase batch付きpassの出力と、数学的gate列の参照との差は最大約7.5786e-15だった。これらは異なる参照との比較であり、加算して誤差予算にはしない。今回の有限例だけで、実H6のmatrix積・Euler・scalar算術を原因候補から除外しない。

## 4. 実行・検査・保全

19:24:34 JSTに人工runnerを起動し、約2.55秒でnative exit0、launcherと全workerを回収した。生成・Euler・積・整合反復の4独立単位を4processで並列化し、高精度gate行列をcache、積の参照検査には反復blockの冪を利用した。共有設定変更・他processへのsignal・ホストPSIによるkillは0。

最終版の新人工テスト12件と、再利用phase batch/契約の回帰13件がPASSした。自身による検査であり、別AIの独立レビューではない。SOURCE mirrorのbit一致、原raw比較、簡約とscalarの分離、積み重なり、80/120桁整合、誤ったgateの拒否、既存出力の上書き拒否、既定CLIのscope表示を検査した。

保護対象2,598fileを前後SHA照合し、旧SOURCE/runtime・実結果・grant/ledger/marker・STOPを保全した。旧58受理軸、未受理H6一軸、m5_best主比較のAD03保留を変更していない。今回の実分子QPY/NPZ復元・新sampling・分子compile/作用・signal・shot/費用再会計・fitは全て0。新有効grantの発行・旧grant再消費も0。

再実行用の人工専用CLI（固定runtimeを使用）:

```bash
python scripts/m5_local_resynthesis_diagnostic.py --run-artificial --output NEW_OUTPUT.json
python -m unittest discover -s tests -p test_m5_local_resynthesis_diagnostic.py -v
```

`--run-artificial`なしではscope表示だけを行う。既存出力への上書きは拒否する。全出力、SOURCE結合、実行終了、自己検査、構造検査、保全要約、inventoryをbundleに保存した。

## 5. 結論と次の担当

今回、**Euler抽出の丸め、内部scalar算術、角度簡約を人工例で分離できた。角度簡約は、前報の位相蓄積修正だけでは消えない誤差源として再現した。** H6の残差の原因、production修正の十分性、H6の全状態等価性・費用受理は確定していない。

**次のユーザー判断**：保存H6の局所runに、この種の簡約が存在するかを照合する限定解析へ進むか判断する。

**Codex（次の承認がある場合）**：保存回路に対する局所簡約の発生箇所・回数・raw局所差を調べる最小契約を具体化する。実H6の全compileを修正なしに反復せず、実H6再解析・新compiler-policyの採否・全回路再検証は別の来歴と承認で扱う。既定の簡約無効化や基準緩和を、今回の検証結果だけで自動適用しない。

local commit後**MANDATORY STOP**。公開push・新H6再計算へ進まない。
