# 2026-10-19 m5局所人工診断

利用者の「ではその検証を行って」に基づき、Euler抽出・U1X scalar・ZSXX内部phase・角度簡約を分離して検証した。新SOURCEは057d954b16924b796ab2780727bfddc663a7635d。前報SOURCE96cd7cd4の実H6結果はraw基準超過STOPのまま。

45人工入力×3設定、Euler45、積9、barrier分離反復4件を4processで実行し、約2.55秒/native exit0・全回収。小角2^-40のRZは既定簡約で省略され、scalarの和だけを正確にしても約9.1e-13差が残る。簡約無効の同じ人工入力は約2.9e-16。実際のpassでも4096反復のbasis状態1で約3.725e-9となり、外側の正確phase batchだけでは局所近似の蓄積を防げない人工反例が得られた。

これはH6で同じ簡約が起きた証拠ではなく、H6局所runでの発生は未検証。実H6の入力を使わず、原因確定・修正完了・費用受理を認定しない。既存の1e-9基準、旧STOP・58軸・AD03保留を維持する。生成mirror135件のbit一致、最終新人工12＋既存回帰13件PASS、2598file保全。実分子QPY/NPZ・compile/作用・sampling・signal・会計・有効grant・push0。

[結果・限界・次の担当](../track_a_m5_local_resynthesis_diagnostic_20261019.md)。次はユーザーがH6保存局所runへの限定照合へ進むか判断し、承認があればCodexが最小契約を具体化する。local commit後mandatory STOP。
