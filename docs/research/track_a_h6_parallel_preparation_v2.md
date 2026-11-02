# H6 次回探索：4 workerへの変更 v2

2026-10-12 JST。利用者の「4ワーカーでよい」によりworker数を4へ変更した。
[一つの資料入口](../../artifacts/resource_applicability/track_a_h6_parallel_preparation_v2/2026-10-12/README.md)に新SOURCE/seal、資源配分、38限定tests、残るlaunch条件をまとめる。
SOURCE `df2976f05c97c8c777f9a6264e83a53b637fef65`。旧[2 worker準備](track_a_h6_parallel_preparation_v1.md)と証拠は保持した。

linear H6/1Å/STO-3G/tol-only rank19/sector400/保存state/T0.8、prefix5/10/15/full19、92候補/4精度の科学契約を維持。
科学SOURCE224/旧runtime233・compiler/seed/trajectory/full-wrapper/費用定義に変更なし。
内部thread1のgroup processを4つ同時処理。単一groupのcompilerをmulti-thread化する変更ではない。
古典wallの短縮率は未測定で、量子回路資源の改善とは扱わない。

Worker CPU20/21/24/25、coordinator22/23。AS32GiB/workerとcoordinator soft2/hard32GiBを維持。
合計RSS40GiB・RAM64GiBを保持し、RSS9GiB×4＋coordinator2＋margin2=40GiB。
9GiBはsampled STOP閾値。前回peak7.929GiBは一groupの実測であり、残142の最大メモリを確定しない。
Host/cgroup PSI/OOM/5秒・出力/monitor/journalの従来guardを保持する。

38 local metadata/mock/最大4 tiny-process検査PASS。旧2 worker grantを拒否し、4 slotsの重複稼働と全process終了を確認。
科学array読込/人工transpile/新science launch/approved grant0。独立review・immutable CIは実施していない。
Native Phase A完了は旧proofを再使用。450/592 group・904 wrappersを保持し、固定142探索のtask/seed/static budgetは同一。
新v2 output/control namespaceで旧資料を保存し、SOURCE/REVIEWを分けて固定。

4 worker指定は採用済み。技術seal=true、approved=false/runtime_authorization=false/allowed_cpus=[]。
142探索の一回launchは未実行・別指示待ち。新exact grant・remote SHA・fresh resource/input/source/profile/carry gateを必要とする。
完了/STOP後mandatory STOP、retry/resume・input再生成・selection/独立32確認/H8へ進まない。
User push方針を維持する。既存H4・共有環境・既存venvへ変更なし。

## 2026-10-12 実行承認追記

利用者が固定4 workerによる残142 groupの計算を明示指示。新exact grant/source/profile/input/carryを照合し、fresh resource gate合格。起動を認可し、完了/STOP後mandatory STOP。Remoteの新branchは未公開だが、利用者push方針と提示済みlocal SOURCEでの最新起動指示に従う。旧draft/旧grantは保持。実際のgrant/command/検査は資料入口のlaunch_authorization_v1へ固定した。

## 2026-10-12 実行結果：再現性gate STOP

4 workerを一度起動したが、index450の旧partial-event digest再現不一致で停止。4 worker/coordinatorの全PID終了・reap/cleanupを確認。新受理0、残142。Source/hash/version一致でも保存basis/component metadataに差があり、環境間のnative numerical/basis再現性は未解決。一致検査を緩和せず、grant/lock消費済みを保持して自動retryしない。詳細・原証拠・未記録counterは資料入口末尾に記録した。
