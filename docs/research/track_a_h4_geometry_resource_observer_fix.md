# H4 resource observer修正・source再固定・未承認草案更新

2026-10-06 JST。`H4_INPUT_GENERATION_RESOURCE_FIX_FROZEN_AWAITING_REVIEW`。
今回認可されたのはobserver最小修正、zero-science検査、実環境のread-only観測、source commit固定、
別commitでの入力生成認可草案再binding、non-force pushだけ。科学計算・本番起動は未認可。
CPU許可を代行せず、保存 `allowed_cpus=[] / approved=false` を保持する。実行準備完了とはしない。

起点 `5245a29ca26cad7421640410934907647459b822` はfetch後のremote branchと一致し、
旧認可manifest29 member＋自身の30 file setと全blob/hashを修正前に照合した。
旧science source `6a121725ce751affd2d3d131a84944728e6b2343`、旧並列review
`88461f3930b9fef511739f91edae88231c33a3f5`、契約base `b662dbd72e49fa713a25c716f323843e547e973b`の系譜も照合した。
独立branch `track-a-h4-geometry-resource-observer-fix-20261006` を使用する。
旧worktree/bundle、root memory欠落の旧観測・旧plan/review履歴を上書きしない。

入口は[source固定資料](../../artifacts/resource_applicability/track_a_h4_geometry_resource_observer_fix/2026-10-06/README.md)と
[更新した入力生成草案v2](../../artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06-v2/README.md)。

## 原因・root判定・最小修正

旧observerは所属cgroupから全祖先をたどり、真のhierarchy rootにもmemory.max/current/eventsを必須としていた。
Linux cgroup v2ではこれらは非root interfaceであり、rootの欠落は正常である。
根拠は[Linux v6.8公式memory interface](https://www.kernel.org/doc/html/v6.8/admin-guide/cgroup-v2.html#memory-interface-files)と
[root interface規約](https://www.kernel.org/doc/html/v6.8/admin-guide/cgroup-v2.html#conventions)。
ただしmountinfoのrootが `/` という情報だけでは、private namespaceに隠れた上位制限を排除できない。

新observerは次の根拠をすべて要求する。

1. `/proc/self/ns/cgroup` がLinux初期namespaceの `cgroup:[4026531835]`。
   Linux v6.8 [proc_ns.h](https://github.com/torvalds/linux/blob/v6.8/include/linux/proc_ns.h)の
   `PROC_CGROUP_INIT_INO=0xEFFFFFFB` と、[公式namespace説明](https://www.kernel.org/doc/html/v6.8/admin-guide/cgroup-v2.html#the-root-and-views)に基づく。
   private/不明namespaceは推測せずSTOP。非初期namespaceを受け入れる拡張は行っていない。
2. 対応するmemory hierarchy mountが一意で、mountinfoのfilesystem rootが `/`。
   所属pathとmountの絶対/canonical pathを確認し、dotdot/escapeを拒否する。
3. cgroup mount内を覆う別mountがない。途中の制限/interfaceが隠れるshadow mountはSTOP。
4. 所属cgroup、mountinfo、namespaceが観測前後で一致する。観測中の移動/可視性変更はSTOP。

root識別はfile欠落を根拠にせず、上記のnamespace/mount/所属情報から決める。
v2真のrootではmemory controllerの存在を読み、rootのmemory制御interfaceとroot memory.pressureは要求しない。
global/root pressureはhost `/proc/pressure/memory` で監視し、全非root祖先のmemory.pressureも最大値へ含める。
全非rootではmax/current/events/pressureを引き続き必須とする。
missing/permission error、非有限/不正値、OOM field欠測/重複等はSTOPし、ENOENTを一律に無視しない。
v1は従来のrootを含むlimit/usage/failcnt検査を保持する。

実効availableはhost MemAvailableと、確認した全有限memory.max-currentの最小。
全非root制限を調べた上で全てunlimitedならhost値が最小になるが、欠測時のhost-only fallbackではない。
memory観測の修正と必要なparse/root helperは `resources.py` 内へ限定した。
production `gates.py` の変更は新source audit pathだけ。構造/認可/CPU/fingerprint gateは変更しない。
科学入力・signal・seed・回路・serializer・並列制御・compiler source15件はbyte-identical。
admission、AS/RSS、Monitor、wall/output等の非observer resource ASTも旧sourceと一致する。

## 実環境観測と限界

kernel `6.8.0-49-generic`、初期cgroup namespaceを確認した。
所属session scope、user slice、user.sliceの3非root祖先と真のrootを検査し、read-only observerが成功した。
準備観測時のhost/effective availableは約981.826 GiBで、全3祖先のmemory.maxはunlimited、OOMカウンタ0、PSI full avg10=0だった。
raw bytes、全path/current/maxと観測時刻はsource bundleの `live_observation_v1.json` に記録する。
process CPU観測は0–255だが利用許可ではない。live admission・AS設定・worker/runner起動は実行していない。

この観測は予約・launch許可・将来のfresh admission成立・RSS上界の証明ではない。
将来launchで5秒以内の観測、CPU許可、全祖先制限、pressure/OOMと既存admissionを改めて検査する。
private namespace/hidden mount等で真のrootを証明できないcontextでは
`H4_INPUT_GENERATION_RESOURCE_OBSERVER_BLOCKED` としてSTOPする。

## 検査・source固定と草案更新順序

observer専用33 tests PASS、fail/error/skip0。fake /proc/cgroup/mount metadataだけで、
root正常欠落、有限/unlimited/複数祖先、非root欠測/permission/不正値、hidden root/shadow/namespace、
pressure/OOM、fresh admission境界、CPU許可空/CPU subsetを確認した。observer検査attemptは1、失敗0。
source固定後の別binding suiteで新旧source/plan/auth/review混在を拒否し、結果は新草案bundleへ保存する。
合格authorize/metadata-only checkoutの承認と架空CPUはメモリ内だけ。保存review/CPU許可へコピーしない。
production launch、OwnedRun/OwnedPool、分子/科学/output境界をmock禁止する。
111-suite、全repo tests、科学artifactを読むtests、benchmarkは実行しない。追加transpile0、旧累積28/64を維持する。

A commitはobserver/gate audit参照の2 source、独立helper/runner/test、実装資料/索引、検査・read-only監査を固定する。
actual NEW_SOURCE_COMMITの自己参照値をsourceへ埋め込まない。
B commitはそのSOURCE blobを照合した外部source freeze auditと、new source-bound plan/auth/review/manifest等だけを追加する。
BではAのsource/検査/文書を一切変更しない。

new science checkoutは
`/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-resource-observer-fix-20261006`。
planのsource_root/source_commit/source_hashes/source_audit_sha256をこの実checkoutとactual new SOURCEへ再bindingする。
artifact anchor、fixed run ID/outputは契約どおりで、本番output/registryをresolve/stat/作成しない。
旧v1は保存し、認可草案v2を新 `2026-10-06-v2/` に追加する。

## 不変条件・未解決事項・STOP

H4 linear/neutral singlet/STO-3G、6距離0.70/0.80/0.90/1.10/1.40/1.60 Å、4 spatial/8 system、DF fragments12。
SCF/DF/order/sector36 solver/gates/master-seed、218 templates、将来74,784 wrapper capは不変。
inputs/freeze digestは未生成null、入力生成6件freeze後STOP、signal/sampling/build/compile stageは未認可。
requested workers6/最大12、driver/worker AS8GiB/RSS別、headroom16GiB、8+8w+16GiB（w6で72GiB）、
累積wall72h/output10GiB、one-shot/no retry/resume/fixed run/rootを維持する。

CPU許可は未確定、allowed_cpus=[]。process_cpus <= allowed_cpus guardを緩和せず、affinity/priority/cgroupを変更しない。
明示CPU許可と適合する別launch context、独立最終review、利用者の明示launchが必要。
保存reviewはapproved=false、有効execution authorization0。実行準備完了ではない。
実SCF/DF/state/分子入力、live worker/production性能、障害・pressure時の実process停止、全campaign資源内完了は未検証。
分子アクセス/生成、科学処理、実seed/build/compile、追加transpile、GPU、共有環境/他job変更、本番起動0。
旧証拠・原稿・Track B・旧bundle/manifest履歴を保存し、公開後STOPする。
