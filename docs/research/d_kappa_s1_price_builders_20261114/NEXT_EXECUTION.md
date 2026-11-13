# 次に必要な科学実行・compile（すべて未承認・未実行）

固定SOURCEは `04bead115233641233ae1fc409b2c4d89ebc2483`。各callerは実行時に、自分自身と使用moduleのbyteをHEAD・固定blobと照合する。このため、同じbyteを持つ子孫commitから実行してもよい（記録されるcommitはHEAD）。いずれの段階も別々に承認でき、自動では次へ進まない。

## S-1：Full-H κ2 signal（推奨・最初）

| 項目 | 内容 |
|---|---|
| 実行 | `python -B scripts/run_full_h_kappa2_signals.py --execute-approved-s1`（Python3.12/NumPy1.26.4/SciPy1.14.1。R448と同じvenv） |
| 対象 | configuration `54ac3eb38de5f90a77e9921100576c89f6c870f68384f4c7d9efe61b2c80c07e`、m0〜12、q=2^m、K2、κ2 |
| 入力 | INDEX `7a8bc7f1` のsnapshot/receipt/context（hash照合）、Full-H preparation（9b83転記）、保存reference pair 13組（`bba15083`） |
| 件数 | context 1、非scalar spectrum 1、finite normalizer 13、mean 13（raw/correctedの組）。新reference/scalar抽出/sampling/回路は0 |
| gate | 既存の数値gate一式（state norm 1e−12、Hermiticity 1e−9、reference pair 1e−10、closure 1e−9、Raw∈[−1,1]、b>1）に加え、0<logB≤(λT)²/s＋1 ULP感度、IDとq/R/sの一致 |
| 負荷 | event列を生成しない。R448実行（B0I＋Full-Hの26 meanで4.1 s）と同程度の見込み（未計測） |
| 出力 | `docs/research/d_full_h_kappa2_s1_results_20261114/run01/`（PLAN、round_m、RESULT、POST_GATES、B_SIGNAL_INPUT、SIGNALS.csv、COVERAGE、STOP） |
| STOP | SOURCEのbyte、configuration、入力hash、reference、gate、post-gateのいずれかが不一致。部分保存し、retryしない |

## P-1：κ2 m3の新ZZ配線（S-1とは独立）

| 項目 | 内容 |
|---|---|
| 実行 | `python -B scripts/run_full_h_kappa2_p1.py --execute-approved-p1`（親）→ 子 `--child`（元 `install()` が最初、provider c31ef424とbyte照合） |
| 対象 | `FULL_H_4de6130f0d3a66e5d695`（T=.8、q8/R71/K2、s568）、cohort `D_FULL_H_KAPPA2_NEW_ZZ_M3_WIRING_V1`、seed 8個（固定） |
| 件数 | context 1（元Full-H、prepare_full_h）、sample 1 path（8 request）、新ZZ compose 1、legacy replay compose 1（compileなし）、wrapper 2、**compile最大2**、statevector 7以下 |
| gate | index0の既存5〜7検査、新ZZ有効、`crzz`=0、parity PASS、one-bodyがsampleされたこと、scalarの一回会計と期待値の一致 |
| 出力 | public `docs/research/d_full_h_kappa2_p1_results_20261114/run01/`、private `/home/AbeHiromu/projects/d-full-h-kappa-20261114-private/P1-KAPPA2-M3/run01/`。PRICES/SIDECARS/IMPLEMENTATION_EVIDENCE（公開commitの確定後にcommitを記入） |
| 位置付け | 配線と実回路の技術証拠。law平均・CI・model点ではない |

## T3：Truncated PF p7の価格（Cの計画：最大18 wrapper）

| 項目 | 内容 |
|---|---|
| 実行 | `python -B scripts/run_truncated_p7_prices.py --execute-approved-t3` |
| 対象 | S2 16h/7（m4 q7、m5 q14、m6 q28）、Y4 16h/3（q3/6/12）、M5 16h（q1/2/4）、B0I p7。physical IDはPLAN_METADATAと一致 |
| 件数 | native_context 1（prefix7）、決定論cost_trial 9、**compile最大18**。samplingは0 |
| gate | index0の既存gate、保持block数・順序、B0I tail identityが保存prefix7 scalarとbit一致 |
| 役割 | m4/m5でCの係数を校正し、m6をholdoutとする（fit・holdout判定はCのscope）。B0とB0Iの同値確認はCの元計画のとおりで、本callerには含めない |

## Pauli LD1/m0 pilot（実行callerは未実装）

接続library、辞書・packetの束縛、seed、scalar（c＋s_R）は準備済み。実行前に次を実装し、人工loaderで確認する必要がある。

1. 子process（provider root）で `native_context('H4',(1,))` からhead blockを得る。
2. index0のfull-256独立oracle。A P0の `LD1_OPERATORS.npz:head.npy` を使い、Pauli行列とのbit順（保存mode-0-MSB ↔ Qiskit little endian）の整合を実行前gateにする。
3. C guardのPauli route拡張（C担当）。

規模はnormal 1 path、X/Y、compile最大2。

## 共通STOPと保全

- 不一致や失敗があれば部分保存してSTOPする。retry、別seed・round・幅・κへの切替、閾値緩和、片軸での代用はしない。UNKNOWNを0にしない。
- 一時的なPSI・CPU競合だけでは止めない。他担当や他ユーザーのprocess・設定には触れない。CPU・RAM・workerに人工上限は置かない（P-1とT3は少数compileなので逐次で十分）。
- 実行後もmandatory STOP。N、G、価格modelの採択、κ・K・groupingの採択へは自動で進まない。
