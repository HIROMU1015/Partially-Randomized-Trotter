# H6原稿・H4補完・H8入力事前検証：現在の資料入口

2026-10-13 JST。利用者が準備後開始と対象1・3を指示した範囲で、H4固定anchor補完とH8入力/stateの最初の事前検証を実行した。以下の「完了」はこの二scopeの完了で、H8 full comparisonや20compile feasibilityの完了ではない。

- [H6本文v0.2](../../../../docs/manuscripts/track_a_h6_resource_study_v0_2.md)、[H6資料/図3点/CSV/55metadata検査](../../track_a_h6_manuscript_h8_plan_v1/2026-10-13/README.md)。探索592/独立256/pilot8を分離し、約54–63%点RZ削減とtail/u/preparationの限界を記載。
- [H4実補完：16 signal/264group/532compile/256trajectory](../../track_a_h4_common_anchor_result_v1/2026-10-13/README.md)。SOURCE217c971d、exit0/全264reap/cleanup0。約17分48秒、8worker/hash再利用。点PR/PF比.507/.472/.453/.511、fixed anchor内のみ。
- [H8 raw→state source/準備](../../track_a_h8_saved_state_preflight_v1/2026-10-13/README.md)。最終SOURCE2ff3fc9f、26限定人工/mock＋実state完了。raw DF28/16modes/sector4900、入力/state snapshot SHAとnative proofを本directoryへ保存。NPZそのものはcommitしない。
- [受領/実消費/source/STOP/carry概要](result_receipt.json)、[H8 snapshot receipt](H8_snapshot_receipt.json)、[projection](H8_projection_receipt.json)、[DF receipt](H8_df_receipt.json)、[terminal](H8_state_terminal.json)。成功state matvec61、SCF1/DF1を再実行0。coordinator297793/CPU31、workerCPU30、exit0/cleanup0。
- [H4/H6共通条件と不足](../../../../docs/research/track_a_h4_h6_common_comparison_plan_v1.md)、[H8科学・高速化・予算案](../../../../docs/research/track_a_h8_resource_comparison_plan_v1.md)。actual rank28で相対prefix generator案は7/14/21になるが、H8科学契約は未採用。

## 条件・停止履歴・消費の保持

H4は既存Hamiltonian/state/rank12のbytesを維持。H8はSCF/raw生成1/1を保持し、Hermitian gate停止→保存rawのweighted projection→stateへ進め、SCF/DFを再生成しない。weighted primaryN16のeta≈3.6595e-12、decision9.9e-11 Hartreeを維持、工学的評価でinterval certificateではない。保存stateのenergyはDF Hamiltonianのsolver値、residual≈4.43e-15、ground-state証明や化学精度として引用しない。

停止したstate02の正確なmatvec数は終端counter未保存で不明、保存境界1を実消費合計としない。保守的予約10000を保持し、成功61を別に記録する。新26th testは全state portを人工operator/solverで実行し、実入力やtrajectory/compileを使わない。旧grantの再利用・消費返却・resetなし。H6carry1716/842＋予約20/10、H4 M1履歴12128/6208と別production予算も保全する。

共有環境・既存venv・他job/旧H4 run変更なし、GPU query/use0。CPU/AS/RSS/thread設定はown processだけ。失敗ごとに新SOURCE/grant/outputを使い、native STOP/cleanup evidenceを保存。実H8 cost/trajectory・full-wrapper transpileはまだ0。

## 次の研究判断と実行範囲

H8の科学比較契約（resource-mapかtransfer、prefix/q16、reference/数値allowance、tail diagnostic/estimator、H4 DF政策）をGPTと利用者で確定する。H8の16-mode circuit/control validation、signal/referenceと小規模8case/20compile resource feasibilityは未実装・未実行部分を残す。ここを既存12-mode成功や入力/stateの成功からPASSへ昇格しない。fullH8 grantは発行しない。

今後の実計算も、canonical hash/preparation再利用・group並列化を先に組み込み、heavy実測から安全なworker数を決める。ここまでのH4補完とH8入力scopeは終了し、現在新scopeのrunnerを継続していない。旧H4 productionは継続する。
