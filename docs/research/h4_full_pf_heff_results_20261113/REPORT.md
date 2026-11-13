# H4 Full PF：H_eff固有値シフトの初回実数値診断

2026-11-13 JST。[固定計画](../h4_full_pf_heff_preparation_20261113/SCIENCE_PLAN.md)／[raw結果](run01/result.json)／[全固有系・残差](run01/EIGENSYSTEM.json)／[B HANDOFF](HANDOFF.md)。

**承認されたFull PF S2・Δ=0.1の限定計算を一回完了し、全数値gateを通過した。** 指定principal分枝上の最低固有値シフトは **+0.00013371206319456874 Ha**。元full DF-Hの最低固有状態と最大の射影overlapを持つeffective groupも最低groupだった。これは当該点でH_eff診断を使用できる数値的証拠であり、RPEの厳密bias上界・全RMSE・十分Nの認定ではない。

## 固定条件とSOURCE

Linear H4、隣接1.00 Å、STO-3G、保存tol-only 1e−8 DF10、weighted-Hermitian completion、sector(2,2)/36次元。Full PFは全one-body＋DF10、prefix10、S2、signed weights=[1]、Δ=.1、元fragment順。h=.1/M12/P0、H/state/snapshot・target・prior・decoder・compiler/profileを変更していない。Full PFなのでR/K/κはnull。

| SOURCE | 対象 |
|---|---|
| 準備 `9af1ac8ed8adcaef059f439ce690b22b7a320de0` | PLAN、数値条件、既存H_eff評価器 |
| 科学実行 `3a26b9ba70f7c5022a7e07356fe1df8b620545e8` | `scripts/run_h4_full_pf_heff.py`と人工テストをcommitで固定して実行 |
| C `86f37d25ffd5eee3daf7f25ce001b090b1f2cf33` | 保存H4 loader、`Context.pf_step`。Cのworktreeを変更せずprivate namespaceで読込 |
| PF kernel `dd24dd329949c255d5e37b90ff8993fa8df7159b` | 既存direct-expm、行列累乗・norm/unitarity gate |
| canonical `c5466946ab0ef9b50511210ca1e58fb7147bb0f0` | 元C Full PF configurationと全13round metadata、保存solver proxyのcarrier |

configuration IDは`f2d58f320ed0ebbeff8e6d4c0fbd5de14d6338b493c92015c966935201fd93ef`。H/state/snapshotのhash・二時点のcanonical/physical IDは[実行scope](run01/SCOPE.json)と[B packet](B_HANDOFF.json)に保持した。準備PLANのC numerical alias `5cc7c07...`とC86の数値moduleが同じであることも照合した。

比較targetは**元保存full DF-Hの同sector**。DF前の分子Hamiltonianの厳密groundへの誤差は含まない。元constantは2.29310124732 Ha、stepにはexp(−iΔc)を一回、直接PFのS^qにはexp(−iTc)を一回含めた。tail identityの抽出・追加やenergy shiftは行っていない。H_effのenergyには既にscalarが含まれる。

## スペクトル・保存state・分枝

| 保存された診断値 | 値 |
|---|---:|
| 同sector元full DF-H matrix minimum | −2.1663874486347643 Ha |
| 保存solver proxy（変更なし） | −2.1663874486347634 Ha |
| principal H_eff minimum | −2.1662537365715697 Ha |
| E_eff,min − E_H,min | +0.00013371206319456874 Ha |
| E_eff,min − 保存proxy | +0.00013371206319368056 Ha |
| proxy − matrix minimum | 8.881784197001252e−16 Ha |
| 保存stateの元最低group weight | 1.0000000000000004 |
| 保存stateのeffective最低group weight | 0.999999924308577 |
| 元最低group ↔ effective最低groupの射影trace overlap | 0.9999999243085774 |

両最低groupは指定resolutionでmultiplicity1。元最低groupからの最大trace overlapのindexは`[0]`。最小値を無条件に対応させたのではなく、保存された全basis overlapを別に観測した。`matched_effective_group=null`、`ground_identification_qualified=false`は保持した。元weightが1を約4e−16超える丸め値もclamp/再正規化せず保存している。

principal/selected cut clearanceは2.9249672799326363 rad、branch整数は全36成分で0。Δ‖H_ref‖は0.2166387448634764でπ未満だった。ただし、これだけでPF logの物理的branch追跡やRPEのalias/branch正解を証明しない。分枝statusは`CONVENTION_ONLY_ALIAS_EVIDENCE_NOT_SUPPLIED`、`physical_alias_qualified=false`のままである。reference energyやdecoder priorをlog分枝選択に用いていない。

## 二時点signal照合

| m / T / qPF | 直接PF signal | 全H_effスペクトル再構成 | 絶対差 |
|---|---|---|---:|
| 0 / .1 / 1 | 0.9766283359451701 + 0.21493508784294038 i | 0.9766283359451704 + 0.21493508784294268 i | 2.3276653796347474e−15 |
| 12 / 409.6 / 4096 | 0.20101518904501564 + 0.9795880100619949 i | 0.20101518903662935 + 0.9795880100642788 i | 8.691709382164289e−12 |

全state weightsを使ったΣ_j w_j exp(−iE_eff,j T)と、既存direct PFのS^q state actionを比較した。単一固有状態によるsignal置換はしていない。両点とも固定絶対差1e−9 gate通過。全13roundの共通step条件は準備metadataで保持しているが、今回の直接signal評価はこの二点だけである。

## 数値gate・技術検証

| 数値診断 | 観測値 | 固定gate |
|---|---:|---:|
| scalar込みUのunitarity Fro欠損 | 6.0106401898819976e−15 | 1e−9 |
| U固有系residual Fro | 1.5369889518424796e−14 | 1e−9 |
| exp(−iΔH_eff)再構成 Fro誤差 | 1.115825932878703e−14 | 1e−9 |
| H_eff Hermiticity Fro欠損 | 8.040246397037904e−16 Ha | 1e−9 Ha |
| 元H固有系residual Fro | 1.2955714962740859e−14 Ha | 1e−9 Ha |
| 元H固有basis Gram Fro欠損 | 1.7873013348275928e−13 | 1e−9 |
| m12 power unitarity Fro欠損 | 2.1971122789023622e−11 | 1e−9 |
| m12 propagated state norm誤差 | 5.505595979116151e−13 | 1e−9 |

全残差は[EIGENSYSTEM](run01/EIGENSYSTEM.json)、[round0](run01/round_0.json)、[round12](run01/round_12.json)。cut guard1e−8 rad、energy cluster1e−8 Ha、入力state norm1e−12も変更していない。これらは浮動小数点診断でrigorous enclosureではない。

新callerの4人工テストはPASS（[記録](TESTS.txt)）。非可換2×2、Cの凍結builder、scalar、二時点再構成、非unitary時の保存/STOP、fresh outputを確認した。評価器自体は準備版の19人工テストを再利用し、今回再実行していない。NumPy1.26.4/SciPy1.14.1、実科学caller経過2.8117885950487107秒。CPU/RAM/worker上限や共有環境設定を追加していない。

実行はcontext1、PF step1、既存effective evaluator1（Schur1＋元H Hermitian対角化1）、直接mean2＋全spectral mean2。**新reference0、旧保存mean/reference再解析0、normalizer0、retry0、N/MC/価格/G/幅探索0。** 入力snapshot/receipt/contextの3hashは実行後も全て一致（[記録](POSTRUN_INPUT_CHECKS.json)）。元SOURCE/旧結果/KEEP/STOPに変更なし。

## 結論の範囲とSTOP

この1点では、主経路として採用したH_effの数値構成・元最低状態とのoverlap診断・二時点signal再構成が成立した。保存stateはeffective最低groupを強く支持しているが、他spectral成分を捨てたり、取得したδEを全RPEの厳密bias上界に昇格したりしない。

`rpe_bias_ha/sufficient_N/global_RMSE_bound/G=null`、state ground同定・physical alias・数値誤差の厳密認証は未資格。Bが実PF raw signal由来のriskを評価するなら、同じPF効果をδEとして再加算しない。独立した誤差予算へ移す場合は状態支持・分枝・残余成分・decoder/lossを含む契約が別途必要である。

次は**ユーザー**がBへの接続又は追加の有限検証範囲を指定し、**B**が既存risk経路との会計を確認する。新しい幅探索・Truncated/PR exact-tailへの拡張は自動実行しない。成果公開・remote照合後に**mandatory STOP**。
