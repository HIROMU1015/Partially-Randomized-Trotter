# H6 旧DF準備値の固定によるnative再現性修正

[一つの資料入口](../../artifacts/resource_applicability/track_a_h6_frozen_basis_preparation_v1/2026-10-12/README.md)。
現在は旧環境からの基底・係数archive受領待ち。実H6一致検証、447互換、再seal、本計算再実行は未完了。
SOURCE b53271f59609f9b633f1771a85c7ca7181fdba8a。旧science224/server基盤233・既存venv/profile/input/compilerを保つ。

利用者の「停止しない状態まで進めて」に対し、旧450のevent不一致原因を調査した。
Version/source一致だけでは旧native数値出力を再現できない。CPU別OpenBLAS演算経路の限定比較も旧hashを再現しなかった。
OPENBLAS_CORETYPEの意味は[公式資料](https://www.openmathlib.org/OpenBLAS/docs/runtime_variables/)で確認した。process限定診断だけに使用し、本番環境・共有設定は変更していない。
完全な旧基底/係数準備はGitHub原結果に保存されておらず、部分event metadataの再合成では不十分。
旧環境が使用可能という利用者回答から、同環境でSOURCE固定exporterを使用する経路を準備した。

Float hex・typed JSON・dedup nodeによるcodecと、旧portの準備値だけを差し替えるscoped adapterを実装。
旧Hamiltonian、保存state、DF rank19、prefix5/10/15/full19、92候補・4精度・T0.8・normalization/shot/seed/trajectory/compiler/full-wrapper/cost定義を維持する。
旧prepared hashes7件・450元seed/events・local gate matrix SHAを必須検査し、一致gateを緩和しない。
B0はtruncated Hamiltonianとして正しくbindingする。input/SCF/DF/stateの生成経路を追加しない。

完了したserver447は、archiveと一致するfull-wrapper fingerprintを確認した場合だけ旧費用系列へ採用する。
非一致時は旧feasibility証拠を保持し、同条件447の再測定が必要。既存費用を黙って混ぜない。

限定回帰中にprocess退出時のRSS消失とwaitable退出の競合を確認し、20ms以内のWEXITED|WNOWAIT再確認を追加した。
Definitive退出のみ通常回収する。生存中欠測STOP、ownership/reap、非退出error、最初のSTOP、5秒資源監視を維持。
69 artificial/metadata/mock/tiny-process検査PASS。独立agent review/CI/実H6旧archive一致・production successは未実施。
CPU4worker/AS32GiB/RSS9GiB/aggregate40GiB/RAM64GiB、元host/cgroup pressure/OOM policy、wall cap nullを維持する。

新namespace/one-shot lock、240 source closureとSOURCE/profile/input/archive/receipt/carry bindingを用意した。
Archive未受領なのでexecution_plan_sealed=false、approved=false、runtime_authorization=false、allowed_cpus=[]、absolute command=null。
旧grantを使わず、追加科学campaign/H8/selection/独立32確認へ進まない。
旧失敗消費を返却/resetせず、export/server一致診断のtrajectory2/occurrence16もcarryへ含める。
今回agent pushなし、環境変更・GPU・H4変更・新科学runner起動なし。

旧2026-10-12 four-worker STOP資料と原証拠は変更・削除せず保持した。
