# PR-2 matched-accuracy M1-A execution authorization v1.1

作成日：2026-09-30
基準commit：`88bad4bd06db398c53c5e77dadb2e5b0bef23226`
status：`M1_A_RETRY_AUTHORIZED_AFTER_IMPLEMENTATION_GATE_FAILURE`

## 1. v1実行の停止記録

v1認可による初回実行は、result artifactを作る前に
`ValueError: finite Taylor cutoff does not meet truncation_tolerance.`で停止した。原因は、研究契約で
固定した`K={2,4}`を`make_rte_config`へ渡す一方、実装が暫定値`truncation_tolerance=1.0`を
使っていたため、残差が1を超える候補をAPIのself-consistency gateが拒否したことである。

停止時点でresult JSON、selector結果、compile job、circuit、trajectory、quantum shotは0である。
したがって結果を見た候補・閾値・selector変更ではない。

## 2. 許可する修正

各凍結candidateについて

`tau = lambda_R * delta / r`

を計算し、既存validation moduleと同じ規則で、固定cutoffの一step残差を受理する最小の正の
`truncation_tolerance`を構成する。残差が正なら`nextafter(residual, +inf)`、0なら最小正subnormalを
使用する。この値はRTEConfigの整合性fieldであり、candidateの`K`、signal polynomial、normalization、
bias、shot、selector thresholdを変更しない。

専用testは残差が1を超えるsynthetic `(tau,K)`でも固定cutoffをself-consistentに構成できることを確認する。

## 3. 再実行範囲

v1と同じdevelopment snapshot、208 base候補、最大4 boundary候補、単一process、compile-free M1-Aを
同じoutput pathへ一度だけ再実行できる。初回失敗でoutputは作成されていないことを実行前に確認する。

held-out path/file access、trajectory、circuit build、direct compile、quantum shot、S3は引き続き禁止する。
`SELECTION_LIMITED`なら停止し、clearでもM1-A artifactをcommitするまでM1-Bへ進まない。
