# 固定request Bの実行範囲

2026-11-03 JST。前段の[具体的計画](../rpe_18round_measurement_connection_20261103/MINIMAL_SCIENTIFIC_VALIDATION_PLAN.md)
を提示した後、ユーザーが次の作業への進行を許可した。同じB範囲を実装・人工検証・SOURCE固定・
1回の科学実行・記録まで進める。承認を証明するためのartifactは作らない。

linear H4/1.00Å/STO-3G/legacy DF12、8qubit、N=4/dimension70。
L_D3/12、δ=.02、保存m0/q1とm3/q8。Partial r1/K2、Full r0/K0。
N/α/β/εを変更せず、理想平均照合のmetadataとして保持する。
Partialは全order0／先頭だけorder2の固定profile、Fullはdeterministic。
6request、12axis wrapper、physical ground state solve1、必要な準備と理論作用だけを実行する。
sampling、RNG、compile、価格評価、旧V1/V2/V3/R1の再実行は0。

SOURCEは[run_fixed_request_b.py](run_fixed_request_b.py)と人工testsを事前commitで固定する。
旧16主要moduleはcarrier4591b8d75a2a6a55c74a51225f0c61648023c8eeとのbyte一致を確認する。
これは全推移的依存のclean再現性保証ではない。H snapshot/H hashは前段計画と同一。
policyのtraining fingerprintは旧artifactの`training_selection/training_fingerprint`
`9dcdc4a28866dc223afb0ea212df645a3ed4a28e5c6dad6ddfc0ec1fcb28ba9e`を使う。
旧artifact全体のcontent fingerprintとは別の値である。

理論側はuncontrolled deterministic halvesとsigned componentをsectorで構成し、
元_apply_event_to_stateを再利用する。元_trajectory_signalは出力vectorを捨てるため、
同じ積順序のvector-returning companionを最小追加した。人工非可換toyで元scalarと照合する。
この参照はcontrolled回路の出力から信号を逆算しない。ただし同じdeterministic builderを
half参照にも使うため、builder全体に共通するモデル誤りを独立に検出する検証ではない。

CPU32論理コア、利用可能RAM約51GiBを確認した。NVIDIA driverは利用不可。
512振幅のQiskit ideal actionを6独立requestのspawn processへ分割し、各jobの小行列処理は
BLAS1threadとする。小さいlocal gate演算でthreading overheadとoversubscriptionを避ける選択で、
科学的なCPU/RAM/wall-time上限は設けない。親のreference構築では必要な2componentのbasisをcacheし、
全tail componentの重複dense構築を避ける。GPU/共有環境設定、他ユーザーjobには触れない。
人工回路でserialとprocess経路を照合してから実H4へ適用する。

人工testsは11件PASS（Python3.11.0rc1 / NumPy1.26.4 / Qiskit1.3.0、unittest約0.239秒）。
主エージェント自身の技術確認であり、独立科学レビューではない。

既存matrix allowance1e-10でstate norm/residual/保存energy、component/half leakage、
同一physical stateへの両control branch action、Hadamard Re/Imを判定する。
controlled作用をglobal phase alignmentして合格にしない。
出力先はfresh directoryを1回だけ作成し、既存directoryでは再実行しない。
不一致・未判定でも結果と原因を保存してSTOPし、科学的retryや別profileへ進まない。

実行結果は新artifactへ保存し、旧SOURCE/raw/状態/価格/契約/STOPを書き換えない。
元Full診断のmissing fingerprint、V1のFull7点半径差、V2条件付き成功、R1費用未確定を保全する。
科学実行後の記録・公開照合を終えたらmandatory STOP。
A/C/D、18round sampling、費用再解析、独立詳細科学レビューは自動開始しない。
