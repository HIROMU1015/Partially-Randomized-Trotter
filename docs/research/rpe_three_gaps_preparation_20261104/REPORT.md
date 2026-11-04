# RPE D後続：残存3課題の技術準備

2026-11-04 JST / LOCAL。**SOURCE調査、証明条件、最小実装、人工28件、限定実行計画の固定まで完了。**
実H4の確率上界・固有値/reference・decoder全域に対する新しい数値評価は0件。
新SOURCEの科学実行は公開後の別途明示許可待ちであり、今回開始していない。

対象はlinear H4／隣接1.00 Å／STO-3G／保存legacy DF12、Partial L_D3/δ=.02、
保存physical state ray・4電子sector70、m0..17の保存q/r/K/N/α/β/ε/E_ref。
旧結果89cb94fe4db1d39d574877c3bc2db79a6d7a5f9f、旧D SOURCE6942e5176d2dec115ac3c271e48f84377abda915を保全した。
旧Dのliteral energy上界0.000138679200360120 Ha、保存Σα=0.050000000000000004626684、
総合UNKNOWN/STOPは前回報告からの引用であり、今回は再計算していない。

| 課題 | 準備状況 | 保証可能範囲 | 未保証事項 | 次の科学実行 |
|---|---|---|---|---|
| αの95%保証 | direct Hoeffding上界・exact上端和のSOURCEと人工検証を準備 | 同じN/ε/U_catによる全36軸G、数学的decoderへの条件付き成功率 | 実H4のΣp_upper≤.05は未評価。IID/実測定bridgeは条件 | 別承認後にA：160 bit・36軸各1 pass＋上端和 |
| Float64 decoder | 3方式を調査。旧strict条件再利用、小さい人工prefixの全履歴checkerを準備 | 指定した人工有限域・scalar呼出し経路のみ | H4全域の一様a/c/d、とくに全距離評価の数値誤差・coverageが未解決 | H4実行入口なし。primitive/全距離の静的証明を先に固定する必要あり |
| E₀ reference | exact保存係数のsector構成＋interval LDL*＋Rayleigh SOURCEと人工検証を準備 | 保存DF係数をexactとする4電子sector最小固有値のbracket・η_ref | 実H4のexact Hermitian性・正pivot・η_refは未評価。upstream構築/分子/実state preparationは対象外 | 別承認後にC：256 bit・1構成/1正定値検証/1Rayleigh/1 energy移送 |

## 成果物・再利用

- [証明条件・SOURCE調査・保証範囲](PROOF_AND_SCOPE.md)
- [A/Cを独立指定できる限定科学計画](EXECUTION_PLAN.md)
- [固定17入力/SOURCEのcommit/path/blob/SHA256](fixed_inputs.json)
- [α adapter](alpha_adapter.py)、[reference adapter](reference_adapter.py)、[人工有限prefix checker](decoder_finite.py)
- [defaultでbyte/literal照合だけを行うrunner](run_three_gaps.py)、[人工テスト](test_three_gaps.py)、[技術検証記録](VALIDATION.json)

既存Arbのexact rational・端点・adjoint、Dの幾何/branch/energy helper、
V2の純関数AST loader、既存decoder、DFの整数excitation符号・順序を再利用した。
DFHamiltonian/finite-RTE/RPE4の関連testも調べた。既存testには実分子生成・eigh/compileへ入るものがあるため、
そのmodule全体のpytestは起動せず、必要な人工恒等式と既存純関数を今回のtestへ再利用した。
旧Dの24人工＋V2の10人工PASSは既存証拠として参照し、今回の新規実行数には含めない。

新規部分が必要だった理由：旧α和は直接tailを評価せず、既存float eigsh/eigh/residualは最小性の保証付き証明を与えず、
既存decoderの少数sanityも全域の証明ではないためである。これを補う小さい数値包含関数に限定した。
既存SOURCE・decoderを改変せず、approval-proof、grant/認証/監査基盤は追加していない。

## 課題ごとの到達点

Aはt=ε−U_cat>0と2 exp(−Nt²/2)を外向き評価し、上端をexact rationalで足す。
保存αの再配分を行う処理ではない。目標はexact1/20、等号は可、許容差はない。
U_catのmean bridgeを一度だけ使い、phase予算へ二重加算しない。
新しい確率和が目標を満たすという見込みを科学的結果としては記録しない。

BはNumPy1.26.4のangle→arctan2経路をlocal SOURCEと公式資料で確認した。
公開の関数説明とruntime versionだけから一様ulp上界は得られず、本調査で十分なprimitive契約は未取得。
有限測定格子はangle誤差の全域確認に使えるが、全18roundの履歴と全candidate/距離の直積を単純列挙する案は採用しない。
等間隔rootを使って列挙を避けるには全computed distanceの誤差/順序保存が必要で、そこを省略しない。
人工checkerは2round・全36履歴を包含oracleで確認するだけで、H4全域の代用ではない。
既存の停止済みfloat guard系列や無制限のadapter拡張へ進んでいない。

Cの保存snapshotはDF係数であり、70×70行列の保存byteではないことをheaderとSOURCEで確認した。
保存E_refの由来は同snapshotをmaterializeしたsector行列へのnp.linalg.eigh。Bの保存fingerprintも一致する。
今度は保存係数をexactに扱うHermitian targetをAcbで構成し、H−LI≻0の下界と保存rayのRayleigh上界を組み合わせる。
下界が最小固有値を識別するため、stateの小residual・近似gap・ground-state fidelityを仮定しなくてよい。
非Hermitian係数を黙ってsymmetrizeせずUNKNOWNにする。trial offset1e−8 Haは検証parameterであってη_refの仮定ではない。
η_ref取得後は旧u_Eへ一度だけ加え、実装decoder未保証の上界はnullのままにする。

## 人工検証・今回未実行の範囲

新規28件PASS。主な確認は指数の独立高精度包含、t正値・境界・UNKNOWN、上端和のexact比較、
float underflowとの区別、励起stateの残差ゼロ反例、正定値pivotのstrict境界、複素Hermitian・退化基底、
非正規化ray、fermion符号、mean変換差、有限prefix全域・誤branch/tie/origin、
A未解決でも独立Cが進む人工統合、defaultが科学関数へ入らないこと。
人工checker初版ではantipodal誤rootでArg区間が広がりoracle一意性を確認できず、
checkerだけをcos距離順序の厳密包含比較へ修正した。既存decoderや実H4条件は変更していない。

```bash
cd docs/research/rpe_three_gaps_preparation_20261104
<research-python> -m unittest -v test_three_gaps
```

Python3.11.0rc1、NumPy1.26.4、SciPy1.14.1、python-flint0.9.0／FLINT3.6.0の既存環境を使用した。
人工統合の8qubit/70次元例も全てzero/constant係数から作った人工データで、保存H4 arrayは読み込んでいない。
実H4のNPZはbyte/hashとZIP/NPY headerだけを確認し、係数・stateのnumeric decode、eigenvalue、
probability tail、成功格子列挙、decoder走査はいずれも0。
sampling/RNG replay、回路build/compile、費用評価も0。新しい科学result directoryは作成していない。
人工PASSは技術的挙動の証拠であり、immutable CI、外部再現、独立科学レビューや科学的GOではない。

## 公開・次の担当・STOP

正式共通運用指示2026-10-25 §8.1・§8.4・§10、repository指示に従う。
専用の独立clone/branch `codex/rpe-three-gaps-preparation-20261104`だけを変更し、
Codex C、他branch/worktree、SERVER1/2、旧SOURCE/raw/grant/ledger/marker/STOPには触れない。
研究上の新結果はないため旧概要・科学manifest・過去研究ノートは変更せず、今回のpackageと関連索引だけを公開する。
公開SHAと公開本文はpush後に照合し、最終チャットに記載する。

次は**ユーザー**：公開SOURCEと計画を確認し、A/Cのどれを実行許可するか指定する。
**Codex（その別許可後）**：許可されたA/Cだけを固定精度・1 passで評価し、結果とUNKNOWNを保存・公開後STOP。
Bを進める場合は、**ユーザーの別指示後にCodex**がprimitive/全距離の静的証明の範囲を限定して準備する。
今回の準備から新科学計算・重要レビューへ自動進行しない。

`RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`、
`U_virtual_IID_upper=null`、`U_exec_upper=null`、旧UNKNOWN/STOPを維持。
**本準備はcommit/push・remote SHA/公開成果物照合後mandatory STOP。**
