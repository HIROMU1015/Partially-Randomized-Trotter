# H4 geometry 入力生成専用plan・認可草案固定

2026-10-06 JST。状態は `H4_INPUT_GENERATION_AUTHORIZATION_DRAFT_FROZEN_AWAITING_REVIEW`。
今回認可されたのはsource-bound plan、result-prior authorization草案、未承認review、zero-science検査、commit/non-force pushだけ。
**保存reviewはapproved=false。CPU許可とmemory条件が未解決であり、実行準備完了ではない。**
入力生成・SCF/DF/state、signal/sampling/build/compile、production runner/worker、GPU、H6、Track Bは実行しない。

入口は[新bundle](../../artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06/README.md)と
[最終実行前レビュー依頼](../../artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06/FINAL_REVIEW_REQUEST_v1.md)。
sourceを変更せず、起点review `88461f3930b9fef511739f91edae88231c33a3f5`から独立branch/worktreeへ認可準備資料だけを追加する。
actual science SOURCE_COMMIT `6a121725ce751affd2d3d131a84944728e6b2343`、契約base `b662dbd72e49fa713a25c716f323843e547e973b`は不変。

## production wireと別資料

凍結済み `gates.structural_gate / authorize / checkout_gate` と `identity.fingerprint` をそのまま使用する。
production JSONには独自fieldを追加しない。status、資源説明、CPU候補、未解決事項、検査・公開scopeは別JSON/文書に保存する。

- planは `stage=input_generation / binding=SOURCE_BOUND / requested_workers=6`。
  actual SOURCE_COMMIT、17 Python source＋namespace parent2件のhash全集合、new source audit SHA-256、
  actual science checkout、artifact anchor、fixed run/output、契約fingerprint、218 templates、compiler/environmentを固定。
  6距離順を保持し、inputs/generation_freeze_digestはnull。未生成hash placeholderやtrajectory seedを作らない。
- authorization草案は入力生成だけ、one_shot/result_prior=true、plan fingerprintへ結合。
  明示CPU許可が未確定なので `allowed_cpus=[]`。観測値を許可に変換しない。
  空リストは既存schemaの構造検査には適合するが、意味論authorizeのCPU条件を満たさない未完成草案である。
- stage reviewは `approved=false`、plan fingerprintとauthorization digestへ結合する。
  現在の保存資料は既存gateで拒否される。利用者の最終承認を自作・代行しない。
  CPUリスト等を後で変更するとauthorization digestが変わるため、更新資料とreviewの再binding・別レビューが必要。

science checkoutは
`/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-parallel-source-20261006`。
認可準備worktreeとは異なる。planのsource_rootはscience checkoutを指し、helperもそこからgatesをimportする。
本番artifact anchorは `/home/AbeHiromu/projects/partially-randomized-trotter`、fixed runは
`track-a-h4-geometry-v2-20261006-run01`。本番output/registryをresolve/stat/作成していない。

## 資源・未解決条件

生成対象はH4 linear/STO-3G、neutral singlet、4 spatial/8 system qubits、DF requested/returned fragments12、
0.70/0.80/0.90/1.10/1.40/1.60 Å。RHF/MO/DF order、有意縮退/prefix tie STOP、sector36 ground stateと数値gateは凍結sourceどおり。
救済・再試行・代替入力は認可しない。6独立入力をfreezeしたらSTOPし、signal/compileへ継続しない。
218 templatesは既存checkout gateの契約照合用に機械転記しただけで、今回signal/costを評価していない。

requested workers6は6独立taskに対応する候補。actual wは将来の既存admissionで決まり、今回はnull。
契約最大12、driver/worker AS各8GiBとRSS別監視、headroom16GiB、required_available=8+8w+16GiB。
w6には72GiB、w1にも32GiBが必要。generation＋mapの累積wall72h、fixed run総output10GiB、内部thread/process各1を維持する。
failure/interruption時STOP、retry/resume/worker補充/払い戻しなし。signal/compileのworker設定は今回未決定。

準備processのCPU観測は0–255、レビュー用候補は0–5。これらは使用許可でも空きCPU情報でもない。
既存sourceの `process_cpus <= allowed_cpus` を満たす必要があり、候補0–5を仮に承認しただけでは現在の広いprocess CPU集合が条件を満たさない。
明示CPU許可と、それだけを露出する別レビュー済みlaunch contextが必要。今回priority/affinity/cgroup設定は変更していない。

準備時のhost MemAvailableはmetadataとして保存したが、既存 `resources.observe_memory()` は上位cgroup確認中の
`/sys/fs/cgroup/memory.max` 欠落で失敗した。実効availableとactual admissionは未取得。
同sourceを現在のcontextからlaunchしても入力生成前のresource gateで停止する条件である。
rootを飛ばす・unlimitedとみなす・guardを緩和する等の修正は今回行わない。
own cgroup membership、memory関連mount、controller metadataだけを別監査へ記録した。
このresource観測経路とlaunch contextのレビューが必要で、source修正が必要と判断されれば別のsource修正・再固定へ戻る。
今回はidentity/JSON gateが成立する未承認草案として保存し、resource gateを通過済みとはしない。
将来launchでは5秒以内のfresh observation、ancestor limits、CPU許可、pressure/OOM等を改めて確認する必要がある。

## zero-science検査と保存範囲

認可準備用の独立helper、guarded runner、専用test fileのみを追加し、production sourceはこれらに依存しない。
17 source＋parent2件をscience/準備checkout/actual SOURCE_COMMITで照合し、不変を確認した。
契約manifest37、旧bundle、旧247 source・保存6 JSON、依存45のversion/RECORD、installed critical source11、compiler metadataを読み取り照合した。

新規57 gate tests PASS、fail/error/skip0。実review falseの拒否、構造/FP binding、source/audit/root/distance/stage/
permission/one-shot/CPU等の改変拒否、pure synthetic admission・CPU subset predicateを検査した。
合格authorize＋metadata-only checkoutはメモリ内コピーのapproved=trueと架空CPU[0]で模擬しただけ。
コピーは承認済みreviewやauthorizationとして保存せず、実CPU許可の根拠にも使わない。
private science/input/output/worker境界をmock禁止し、実launch/OwnedRun/OwnedPoolは呼ばない。
科学依存import、scientific pathアクセス、seed・transpile・GPU・process操作をguardする。

準備・検査は各1attemptで成功、開発中検査失敗0。memory観測失敗は未解決資源条件として別記し、検査成功から除外して隠していない。
追加transpile0、旧系列28/64を維持。111-test suite・全repository tests・benchmarkを再実行していない。
分子アクセス/生成、科学処理、実seed、科学build/compile、GPU、本番runner/worker起動、共有環境・他job変更0。
authorization草案1件を保存したが、有効なexecution authorizationは0、最終review/明示launchは未実施。
local zero-science evidenceであり、実SCF/物理gate/資源成立/CI/独立外部再現ではない。
軽量資料・独立検査・索引追記だけをcommitし、旧科学結果・原稿・Track Bやsnapshot/runtime/cache/registryは対象外。
