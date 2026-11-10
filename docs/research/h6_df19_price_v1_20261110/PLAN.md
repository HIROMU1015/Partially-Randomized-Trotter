# H6 DF19価格モデルV1：固定実行scope

2026-11-10 JST。ユーザーは指定H6 V1の実データdescriptor・予測・残差/相対誤差・誤差要因分析を一括許可した。4c965ff69の設計と今回の追加指示を適用し、残差非0だけで他の保存回路の評価を省略しない。一律の%採用閾値や補正係数fitは導入しない。

固定対象はFull19/m5/q1/T.8を開発、q1/T1.6を留保。PR Aはp5/R32/K2/q1/T.8、Bはp7/R16/K2/q1/T1.6、それぞれ保存探索9を開発、独立確認33を留保する。元trial_filesは完了順なので保存path indexへ対応付け、参照集合を変えない。PF旧q2/T.8/δ.4はreceiptのH6/Full19/preparation/compiler/axisを確認できる場合のみ補助とし、固定δのq外挿検証にしない。

元H6 snapshot/receipt/completion/H/state/fragment順・signed weights/scalar、Qiskit1.3.0/NumPy1.26.4、rz/sx/x/cx/opt1/seed17、元provider c31ef424、symmetric_directional/full Hadamard/measurement/P=0を保持する。stateはロードせず、保存12×12係数から元providerの局所対角化とGaussian関数だけを再利用して価格用basis/diagonal descriptorを取得する。新SCF/DF/Hamiltonian/state生成・ground solveではない。入力原本やbindingに不足・実質的不一致があれば再生成せずSTOP。

H4 V1の局所ゲート展開・literal basis対消去・wire-local RZ融合核を、同一providerの演算に適合する部分だけ再利用し、wire数は13/control12へ変更する。H4係数・誤差率は使わない。H6側は11 signed pieceの半時間・R32/R16配分・prefix5/7とFull19を元SOURCEに沿って明示する。native proxyとcompiled価格を区別し、全compilerの再実装はしない。予測入力に実価格やcohort名を入れない。

科学SOURCEはコード/runnerをcommitして固定する。開発・留保は同じSOURCEのまま一括実行し、開発後のrule変更なし。公開済み集計値とH4不一致結果は既知なので完全blindとは呼ばない。各path/X-YのC/Ĉ/ΔC/絶対誤差/相対誤差、cohort別平均・最大値・整数一致を保存する。探索と確認、X/Yを独立標本としてpoolしない。

既存NativeCounter、元局所関数とH4の人工testを再利用し、H6差分を加えた8人工test PASS。実QuantumCircuit constructor/transpileを禁止する検査下で実行する。descriptorとpiece表をparentで共有し、独立pathのpure countingを利用可能CPU affinityとjob数に応じたprocess並列で行う。数値workerは新しい回路・sampling・BLAS重複計算を行わず、共有設定や他processを変更しない。

新fit/bootstrap/sampling/実build/compile/q2・4新価格取得/signal/N/G/全13round/再最適化/詳細科学レビューは0。近似的有用性、有限完全一致、全13round provider資格を区別し、長q/全law/rare tail/covarianceを未資格のまま残す。旧SOURCE・入力・raw・価格・cache・grant・ledger・marker・KEEP・STOPと他担当worktreeを保全する。新しいapproval証明機構は作らない。結果公開・remote SHA/成果物照合後mandatory STOP。
