# H4/H6 m5 決定論的費用run01：独立2契約で起動

2026-10-20。**ユーザーがH4 38軸、H6 54軸の実compileをそれぞれ独立契約で承認した。両契約はfresh gateを通過し、一回grantを個別に消費して起動した。H4/H6各3 worker・合計6 processで進行中。完了・全PASS・最終費用を確認した報告ではない。**

科学実行SOURCEは`635ba563766875d688ad6912499b7f90397d0cc5`、承認対象の準備commitは`cc5813a1756528069fb45a81abaea890fa5ff0ec`。科学adapter/契約を変更せず、2系の空きRAM二重計上を避ける起動補助をSOURCE`9e44ec883348bad024d18442bdb97ebb9946908f`で追加した。新人工5件PASS。branchは`track-a-h46-m5-cost-sampling-readiness-20261020`。

## 承認と範囲

ユーザーの承認原文：

> H4の38軸とH6の54軸を、それぞれ独立した契約で実際にコンパイルしてよい

[準備報告](track_a_h46_m5_cost_sampling_readiness_20261020.md)の同精度両軸適格性を使った取得案と、その経験的限界・資源admission政策を、この限定取得のために受け入れたものとして結合した。正式shot政策・PRのn_E・主比較採用AD03を採択したことにはしない。旧DRAFT・SOURCE・raw・STOPは保持した。

| 契約 | 新規task | 新規compile上限 | sampling | 実行worker | plan SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| H4_DET | 19 pair | 38軸 | 0 | 3 | `a7471b20d20341ce57b2002e7ba0f867bc305672a88bf48b79cbbae7cb538f64` |
| H6_DET | 27 pair | 54軸 | 0 | 3 | `729ff96cb4205be7af1d1f8f7b6511477d1d6faa0a510a0ed328cf9ab1a20112` |

grant/approval/config/output/ledger/marker/外部native-exit witnessは系ごとに分離し、各2323ファイルを封印した。保存full q2の計4軸は再compileしない。保存12軸全体も保持し、取得対象外の8保存軸を新領域の費用として補わない。

対象はm5の保存q1/2/4/8格子における、同精度で両軸適格な決定論候補。線形1 Å/STO-3G、保存tol-only1e-8・自動rank H4=10/H6=19、T=.8/P=0。full Hadamard wrapper、symmetric-directional control・scalar once、元Qiskit1.3.0/rz,sx,x,cx/opt1/seed17を保持する。m5係数は`PF_M5_BEST_20D64C2_BINARY64_V1`のまま。新SCF/DF/state/reference、signal平均評価、PR sampling、fit、shot再計算、費用会計・ランキングは今回の実行範囲外。別PF788軸研究の承認・結果は流用しない。

各pairはcosine検査後にsine compileへ進む。compiled状態差はCT01の1e-7未満、他native/oracle/control/Hadamard検査は1e-9未満、有限値を要求し位相合わせを行わない。異常ならそのstageのみSTOPし、自動retry・再fit・追加compileをしない。別stageの正常jobは停止しない。成功/失敗とも結果SHA・外部exit・worker回収を保存しmandatory STOPする。

## 資源と保全

起動前に既知3137ファイルのhash不変を確認した。他worktreeのHEAD/statusや資料更新を凍結していない。両系のSOURCE/input/native/compiler/runtimeは承認configのsealで再照合した。

初期RAM proxyは準備報告どおりH4 25,412,015,444 bytes、H6 92,394,930,176 bytes/worker。自分の新childだけに、RAM予約から導いたCPU集合H4={1,3,6}、H6={8,15,16,23}を割り当てた。OS・既存cgroup・共有ライブラリ・他プロセスは変更していない。CPU配置は今回の実行時admissionであり恒常的なCPU/worker上限ではない。

最大CPU slotでのRAM proxy予約合計445,815,767,036 bytesを、両系起動前の実効空きRAMと照合した。scratchは保存H6 q2 QPY byte数からの保守的な容量計画proxyとも照合した。どちらも数学的な資源上界ではない。各native runnerのfresh gateでは実効空きRAM約460.8 GBとscratch約474.8 GBを観測し、H4/H6各3 workerをadmitした。内側BLAS/compilerは1 thread。入力/prefix/fingerprint cacheとpaired evolution再利用を保持する。host PSIや監視遅延だけで正常workerを終了させない。

## 初期進捗と監査の境界

ASOF: 2026-10-20T01:45:44.747104+09:00。H4の先行2 taskがcompile_cosineへ進み、もう1 taskはwrapper_cosine。H6の3 taskはnative_build。6 scientific workerはCPU時間の進行を確認し、初期STOPは0。受理費用・完了pair・最終native exitはこのsnapshotではまだ0/未確定である。

追加の安定進行確認（2026-10-20T01:48:52.140920+09:00）：H4は3 pair・7軸のreceiptが揃い、後続構成へ進行した。H6は0 pair・2軸のreceiptが揃い、cosine/sineの処理を継続中。両系のSTOPは0。これらは部分完了であり、全stage PASSは未確認である。

[起動時bundle](../../artifacts/resource_applicability/track_a_h46_m5_det_cost_run01_start/2026-10-20/)には独立plan/config/grantのSHA、fresh gate、許可範囲、初期進捗、保全receiptを公開projectionとして置く。有効grant本体・入力array・QPY・rawはprivateに残す。これは開始snapshotであり、終了時の最終evidence freezeではない。

private入口：`/home/AbeHiromu/projects/h46-m5-cost-sampling-readiness/20261020/private/authorization_DET_run01/`。各系の`run01/task_completion.jsonl`、`output/<task>/progress.jsonl`、`completion_witness.json`と外部`external_exit_witness_PRIVATE.json`を読む。封印済みsource/grantを編集せず、同じgrantで再起動しない。

## 次の担当

- **Codex**：進行と終了証拠を読取り、全pairの単価・検査・call台帳・SOURCE/入力保全・native exit/worker回収を監査する。実行結果を保存・報告後mandatory STOP。
- **ユーザー**：PR標本設計・正式shot政策・後続の会計/選択を判断する。今回の2契約からこれらへ自動移行しない。
