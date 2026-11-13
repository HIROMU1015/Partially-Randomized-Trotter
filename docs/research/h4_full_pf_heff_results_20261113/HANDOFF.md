# A → B/C：Full PF S2・Δ=.1のH_eff結果

[REPORT](REPORT.md)／[機械可読packet](B_HANDOFF.json)／[全固有系](run01/EIGENSYSTEM.json)。科学SOURCE=`3a26b9ba70f7c5022a7e07356fe1df8b620545e8`、固定準備=`9af1ac8ed8adcaef059f439ce690b22b7a320de0`。保存H4 DF10/sector(2,2)/Full PF S2/prefix10/Δ.1/h.1/M12/P0の一構成だけ。

## 取得済み

- 元full DF-H matrix minimum：−2.1663874486347643 Ha。
- principal H_eff minimum：−2.1662537365715697 Ha。
- 両minimumのシフト：+0.00013371206319456874 Ha。
- 元最低groupに最大overlapを持つeffective groupは0、射影trace overlap0.9999999243085774。保存stateの同effective group weight0.999999924308577。
- m0/m12の直接PFと全スペクトル再構成の差は2.3276653796347474e−15／8.691709382164289e−12、全数値gate通過。
- step1、Schur1、元H Hermitian対角化1、直接mean2、spectral mean2。新reference/旧signal再解析/normalizer/N/MC/C/G/幅探索0。

canonical configuration ID=`f2d58f320ed0ebbeff8e6d4c0fbd5de14d6338b493c92015c966935201fd93ef`。physical IDはm0=`5669f142d693eee92583`、m12=`4841a53bcaf02927b34a`。元Cの全13round scheduleとidentityを変えず、今回取得した二点だけに結果を紐付けた。全13roundの新signalやN packetを生成した意味ではない。

## API・数値の意味

既存`pf_effective_hamiltonian.evaluate`の出力を[EIGENSYSTEM.json](run01/EIGENSYSTEM.json)へそのまま保存した。Bは`H_eff/spectrum/reference_comparison/diagnostics`と元SOURCEを参照できる。original/effective双方の全energy・eigenvector・保存state weights、basis間overlap、energy groups、選択分枝整数を保持する。新しいN/RMSE schemaやdecoderは実装していない。

`matched_effective_group`はnullのまま。最大overlapの観測と対応先の科学的採択を区別する。保存proxyは−2.1663874486347634 Haで、元Hの新matrix minimumに置換していない。元weightの1+4e−16という丸め値も修正していない。

H_effには元constantをstep時間で一回含め、直接PFは元規則どおりS^qへtotal-T scalarを一回適用した。したがってBがspectrumからsignalを使う際にglobal scalarを再加算しない。Full PFはB=1、raw=corrected。Truncatedのtail identity、PR head/exact-tail/finite-Kへこのshiftを移植しない。

## 資格とriskの会計

方法は`ADOPTED_MAIN_DETERMINISTIC_PF_ERROR_ROUTE`、数値statusは`EFFECTIVE_HAMILTONIAN_DIAGNOSTIC_AVAILABLE`。principal cut余裕は十分だったが、physical alias・厳密ground同定・数値誤差の厳密認証はfalse。δEは指定branchのspectral診断であり、`rpe_bias_ha/sufficient_N/global_RMSE_bound/G`はnull。

実PF rawを直接decoder riskへ渡すとPF効果はその信号に含まれる。同じδEを更に誤差項として加算しない。QPE統計モデルからPF errorを分離してMinkowski予算へ組み込む場合は、比較する理想state/energy/branchと他spectral成分、decoder/alias/未解決lossの条件を先に明示する。二時点一致だけではこの資格を付けない。

**B**：ユーザーが指定した既存risk経路への接続を検討する。**ユーザー**：追加幅・Truncated/PRへの有限検証が必要なら対象とscopeを別途決定・承認する。今回の公開・remote照合後Aはmandatory STOPし、追加科学計算を開始しない。
