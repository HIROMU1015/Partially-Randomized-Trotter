# 承認済みFull PF・B0I p7 shortlistの48 mean補完

ユーザー添付指示（2026-11-15）の48 candidate-roundを新科学SOURCEで一回実行する。S3の44件と、旧Track Aに現行gate証拠のないS2L/m3の4件のみ。旧SOURCE/raw/staging/KEEP/STOPを変更しない。全scopeは[preflight/PLAN](preflight/PLAN.json)、既存重複照合は[DUPLICATE_CHECK](preflight/DUPLICATE_CHECK.json)を正本とする。

| 候補 | 新round | 保存再利用 | 新件数 |
|---|---|---|---:|
| Full PF S2・16h/7 | m2,m3,m5〜11 | m0,m1,m4,m12 | 9 |
| Full PF Y4・16h/3 | m3,m5〜11 | m0,m1,m2,m4,m12 | 8 |
| Full PF M5・16h | m5〜11 | m0〜4,m12 | 7 |
| B0I p7 S2・16h/7 | m2,m3,m5〜11 | m0,m1,m4,m12 | 9 |
| B0I p7 Y4・16h/3 | m3,m5〜11 | m0,m1,m2,m4,m12 | 8 |
| B0I p7 M5・16h | m5〜11 | m0〜4,m12 | 7 |

保存Linear H4/1.00Å/STO3G/tol-only1e−8 DF10/sector(2,2)36・同H/state/snapshot、C共有target、h=.1/M12/元prior/decoder/policy/P=0を維持。actualΔ/q/signed scheduleは既存canonical/CELL_PLAN_V2と保存CELL_SCOPEから厳密に照合する。Full S2とB0I S2のm0の2件は、ユーザー指定どおり同一物理roundの保存値として再利用し、50件へ拡張しない。

## 重複照合とSOURCE

A217eab647と前段c212cee85の710保存record auditを再利用。関連A/B/C/T2/A-signal/Dの公開HEAD・同family実行jobのdeltaを確認した。新しい受入可能mean0、同時重複job0（照合範囲内）。C855573064→0b4b2d94は候補対応表/metadataだけで新mean/数値codeなし。S2Lは既存gate証拠不足のままで、旧値を昇格せず今回directで4件取得する。未参照の別server非公開runを確認済みとは呼ばない。

科学開始前にcaller/helper・計画・人工testをcommitし、SOURCEを固定する。既存baseline providerは無変更で、数値式・gate・B intakeを変えない。追加のgrant/承認証明書は不要。

## 数値経路

既存 `I.plan→C.prepare→I.produce_signals→C.execute` を使う。保存30件はstagingの `stage_record→C.b_mean` と同じ束縛で `saved=records`へ渡す。raw/corrected/normalization/reference/旧数値gateは元値のまま。単一SOURCEのpacketへ束縛するが、元producer/元file/物理ID・元numerical SOURCEは `provider_binding.input_source`に保持する。古い科学結果を新しく計算したことにはしない。

新48件はFull PFのscan/follow-up、Truncated T2に保存された10個のstep operatorを一回ずつread-onlyで読み、同じstepを繰り返し構築しない。既存 `baseline_signal_numerics.evaluate` に渡す最小SavedStepContextだけ追加する。新しいexpm、Hermitian対角化/Schur、spectrum/powerの保存値再評価による幅最適化は行わない。PF stepをq乗する既存direct matrix-powerとstate内積が今回の新mean科学callである。

stateは固定save-only snapshot loaderで読み、hash/receipt/sector/lift/normを照合。Hの生成・SCF/DF/state solveや固有系再計算は0。Full PFはc、B0Iはcとp7 tail identityを元kernel規則で総Tに一回ずつ適用する。既存kernelの二つのphase因子は合算scalarと代数的に同じで、符号や適用回数を変えない。

参考pairはbba15083の保存13組、新reference0。既存step/power unitarity・norm1e−9、reference pair1e−10、nonfinite/raw range・scalar/law/SOURCEのgateを維持。決定論PFではraw=corrected、B=1/logB=0/attenuation=1で、X/Yは同じ複素meanの実部/虚部。96 science callsに二重計上しない。

## 並列化・STOP

自然な6候補を独立taskとしてThreadPoolで処理する。実CPU affinityと候補数でworkerを決め、36次元の共有state/10stepはread-only、各candidate/roundのoutputは独立。frozen関数importはthread開始前に一回初期化する。CPU/RAMの恣意的capや共有環境変更はない。

このprocess内で既存evaluateへのcallを薄く包み、承認済みphysical roundだけ一回投入し、数値FAIL/例外でglobal STOPを設定する。元evaluate算術は変更しない。失敗後の新規callは拒否し、開始済みcallは結果を保存して回収する。partial/error分類を残し、retry、backend切替、閾値緩和、候補差替え、missing値の補完は行わない。

## 成果と技術受入

6候補各13round packetを新runへ保存し、成功なら `require_complete` とB9c0e12d87の既存 `T.create/T.bind_signal` へ渡す。BはA所有の一時detached checkoutで無変更実行し、B/C/Dの作業treeは変更しない。intakeは技術受入だけでrisk/MSE/shot solverを呼ばない。

REPORT/HANDOFF、重複照合、保存30件SOURCE、新48件raw/gate、6packet、B受入結果とロードマップ影響を記録する。ロードマップ本体は統合担当のみが更新するため直接編集しない。commit/push・remote SHA/tree/主要file照合後mandatory STOP。MSE/N/価格/G/候補順位・追加科学計算へ進まない。
