# 研究C：tail-P4 PR性能・条件付きnative32の実行準備

2026-10-28 JST。起点`0ad1b8ef06e979707b75d272333b525327af75b4`。今回のチャット依頼は技術準備から条件付きStage B・公開までの一括科学実行許可である。追加承認・承認証明・新grantを要求せず、以下を科学開始前に固定した。

## 対象と既存実装の再利用

保存直鎖H4・隣接1Å・STO-3G、Nα=Nβ2/sector36/system8/full512、tail-P4 CDF8/元順LD2、元g補正κ/nuclear scalar一度。Partial ordered S2 δ=h=.1/s1/r128/K0/m0..12、own-DF ground oracle・状態準備除外。CA=.00159Ha/DFreserveCA/8、βtotalπ/6/βstatπ/12、failure.01/axis.01/26を維持する。

| 必要な処理 | 再利用と今回の追加 |
|---|---|
| 保存CDF/構造 | 前段21-callの最終J/U・DF予算・support。新fit/DF誤差の再計算なし |
| 第三候補 | B1の下位general-K、spin-adapted one-body basis、NativeBlock/NativeModel/splitを使用。新`tail_p4` roleと固有frame/atom/seed identityだけ追加。旧`prepare_arm`のpaired gateは無変更 |
| 独立reference | SectorCARのbase+8 raw因子、B1のcandidate_factor/component_check、ground/alias、reference_kernels/reference_round、feasibilityを使用。新J/UのPF/RTE/groundは新計算 |
| Shots/energy | 既存stage_cの3 lane・qualification/assess_round/energyとM4 shots_summaryを使用。証拠laneの未知値はnull |
| 費用 | M4 affine_accountingを再利用。4保存対照×2仮定laneのbreak-evenだけ薄い集計を追加。既存helperの`old_M3_*`というfield名の値は今回は**保存M4 r128**を指すと明示 |
| Native | M4-nativeの段階構成を第三候補用に限定継承。B1 sampler/build_kernel、B2 immutable CAR cache・law/raw/compiled/adjoint/X/Y/calibration、StrongBuilder、P0、QPY、metricsを無変更再利用 |
| 並列・STOP | 既存Calls/resource_plan、SharedQiskitCallsのbuild/compile/actual-inverse-build mutex。新管理基盤なし |

追加SOURCEは`c_df_structural_pr.py`、`c_df_structural_pr_native.py`、専用runner、人工testの4件。既存SOURCEは変更していない。新PF engine/PR sampler/RPE solver/CAR/Gaussian compilerを再実装していない。

## 固定契約・call

[固定条件付き科学契約](research_c_df_structural_pr_contract_20261028.json) SHA256 `789df4e7373e80a598560fc9a8e009fe6810ecf30cee67a2721ba6db478d7de6`。
[SOURCE93](../../artifacts/research_c_df_structural_pr/2026-10-28/technical/SOURCE.json) SHA256 `324fdf9c09fada29b37916ab69569b0389f8beaa575d608bb10900a63c2fa505`。
保存input25件・旧保護1054件をhash照合。新J/U SHA256は`ec0dfe3d53e8209b343b294a103ab89405ac757d3e26164ae591344d4423b985`。旧Proposedに偽装せず、旧PF/ground/価格を新candidateへ移送しない。

| 段階 | 上限logical units | 内容 |
|---|---:|---|
| Stage A reference | 63 | 保存証拠/input/JU/support/model、sector因子9の独立照合、新ground/alias、新law、新kernel、13 exact/PF-RTE/classification、要約 |
| Stage A accuracy | 48 | 3 qualification＋39 round accuracy＋3 energy＋3 shots要約 |
| Stage A price/GO | 12 | 3 affine＋8対照閾値＋1登録GO |
| 条件付きStage B | 325 | 全256 CAR因子9/sector/law、basis9/relative4、shell6読込/3検査、32 sample/build/reference/raw/compile/compiled/adjoint/XY/metrics、校正/総資源 |
| 最終要約 | 1 | B未実施でも結果・mandatory STOP |
| 最大合計 | **449** | 新fit/optimizer/retry/resume/実量子shots=0 |

Callごとの内容は契約に列挙した。Nativeのadjoint1単位はraw/compiledの2 actual inversesとその照合を含む。全32実行時QPY128件・reference32件。各roundを巨大長回路へbuildしない。新candidateの同J/U決定論・通常CDFγ0のmatching-r128評価は今回含めない。

## 固定したGOとSTOP

SOURCE/input/JU/general-K/law correctness、new own-DFと保存physical ground/alias、全13roundのPF+補正RTE≤sin(π/12)とpositive attenuation、主`ideal_exact_native`のshots/energy適格、現行M4-r128へのRZbreak-evenがfinite/positive、科学開始前に固定したcohort/compiler/allowance、の全条件を満たす場合だけ、**追加承認なしでStage Bへ進む**。新kernel価格を旧平均で代用せず、旧平均より低い閾値であることは停止理由にしない。

不適格なら全13roundを保全しBをNOT_RUNとする。Correctness FAILでは依存処理を停止し、開始済みthreadを回収・partial/STOPを保全する。Retry/resume/fallback/別namespace再試行なし。Native価格の有利不利による32標本の適応停止・追加はしない。

Sampling master2026102806/cohort`tail_P4_s1_r128_native_fixed32_v1`、PCG64、候補JU hash＋SOURCE digestをseedへ結合する。旧M4と非paired、poolなし。各kernelは128 fresh signed K0 events。P0/Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1/qubits_initially_zero=false、MPFR160 declared phase、relative head/同basis tail/no cross-head-tailを共通化する。Matrix1e-10/complexsignal2e-10/rtol0・phasefitなし。

## 人工・回帰と実行環境

人工のdet±/一般対称J/負係数/P4-tailから実full512 raw/compiled/reference・actual adjoint/X/Yを確認。旧paired gate不変、144 atom・newrole/seed、new13round/ground/law/RPE/energy、break-even正負/等値、全449-callのmock dispatch/QPY128/fixed32、部分FAIL/STOP/fresh拒否を確認した。

[ログ](../../artifacts/research_c_df_structural_pr/2026-10-28/technical/)：初回12 PASS（136.11秒、full512実人工回路を含む）、最終回帰39 PASS/2 deselected（8.28秒）。11件の重複を除き**40 unique PASS**。除外の一方は既に初回PASSした新full512 test、他方は同内容の旧M4 full512 testで、二重実行を避けた。実H4科学入力はこの技術testに使用していない。

Task-local venv/cache、既存packageをread-only参照。32 affinity CPU、利用可能RAM約52GiB、GPU T1000 4GiB、低負荷を確認。Sector13roundとnative32を既存plannerで並列化し、複数worker時はBLAS1 thread/workerでoversubscriptionを避ける。CPU/RAM/worker/wallの人工capは追加しない。過去M4-native32のwall約920秒は規模の参考値で、新候補への実測speedupを主張しない。共有環境・他jobは変更しない。

## 科学的な解釈の範囲

保存二候補・固定PR/native政策の比較であり、supportとprojected startの純粋因果分離ではない。2仮定laneは条件付きshots・有限cohort資源の点比較。Evidence lane、全law synthesis/implemented-law error、cost support、母平均優位、actual CA・実量子shotsは未資格/nullのまま。校正SEは同じ32平均の再利用を反映し`W×SE_kernel`、sqrt13で縮めない。Observed min/maxはsupportではない。

SOURCE/契約をcommit・push・remote照合してからfresh run01で一度実行する。結果を公開照合後mandatory STOP。今回の一括許可を別support/γ/rank/LD/s/r/K/分子/距離、新fit、全law保証、重要GPT詳細レビューへ拡張しない。
