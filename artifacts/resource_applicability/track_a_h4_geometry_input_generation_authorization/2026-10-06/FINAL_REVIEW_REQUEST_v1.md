# H4 geometry 入力生成専用認可草案・最終実行前レビュー依頼

状態：`H4_INPUT_GENERATION_AUTHORIZATION_DRAFT_FROZEN_AWAITING_REVIEW`。
**このbundleは実行未承認である。保存reviewはapproved=false、CPU許可とresource観測経路は未解決。**
認可準備資料の固定を、review承認や利用者の明示launchとみなさない。

## 固定対象

- Repository: `HIROMU1015/Partially-Randomized-Trotter`
- Branch: `track-a-h4-geometry-input-generation-authorization-20261006`
- 起点review commit: `88461f3930b9fef511739f91edae88231c33a3f5`
- science SOURCE_COMMIT: `6a121725ce751affd2d3d131a84944728e6b2343`（不変）
- 契約base: `b662dbd72e49fa713a25c716f323843e547e973b`
- 契約plan SHA-256: `18aa36a2776d38852657f154a88c381b3299fbc009007f20a2d95fcb865d9f7a`
- 契約plan fingerprint: `c76e8f1f6de5a2625affde38cc471b8214b299f343da2817aadbb3ebabc7d933`
- 契約manifest SHA-256: `14cc5d0cc4da2b82168a0640cf8ff70ddf382b79810842bab1d7fedfae029f70`
- science source audit SHA-256: `5c9a1997339fa0f1f5479c62b11b6e2ef2ee024ce5a584aa958cfa80c4addd5f`
- 並列source manifest SHA-256: `a9cb2b03a9ff505b5964c7740818b6b41df3f7176105baadd52108330a0ad53a`

認可草案commitはこの資料とmanifestを収録する次commit。実SHAはGit履歴と最終報告から確認する。
sourceへの自己参照SHAやproduction JSONへの独自fieldsは追加していない。

## 読む順序とbyte/hash・fingerprint

1. [認可準備の実装・停止条件](../../../../docs/research/track_a_h4_geometry_input_generation_authorization_draft.md)。
2. [入力生成専用plan](input_generation_plan_v1.json)、[authorization草案](authorization_draft_v1.json)、
   [未承認stage review](stage_review_v1.json)。既存gatesのproduction wireへ厳密に合わせた。
3. [資源/CPUレビュー](resource_cpu_review_v1.json)、[利用者のCPU未確定指定](cpu_permission_decision_v1.json)、
   [own cgroup metadata](resource_context_metadata_v1.json)。
4. [identity監査](identity_audit_v1.json)、[認可準備全体のaudit](authorization_preparation_audit_v1.json)。
5. [gate test結果](gate-tests-attempt-01.json)、[ログ](gate-tests-attempt-01.log)、[準備guard](preparation_guard_audit_v1.json)。
6. [hash/fingerprint一覧](identity_summary_v1.json)、[commit対象manifest](artifact_manifest_v1.json)、
   [公開scope監査](publication_scope_audit_v1.json)、[将来command・未実行](FUTURE_INPUT_GENERATION_COMMAND_v1.md)。

| 資料 | SHA-256（ファイルbytes） | domain fingerprint |
|---|---|---|
| plan | `ac679b6cae2a03afd47ca018f6e3d24da9c0b017d5111f0bb7d41d69fa09f90b` | `37c266e0574dc62d22fe366f68b35c39beb72cf8f84acb4cc73851eb5777bdbd` |
| authorization草案 | `7699146b44913a01a1564d01075609f5e4763f8f672fb4c0d094cacb3949cd83` | `8320aae1f5132cebb514af3cfc4c1d5f3d91073e4053332d23c7519cb8c47f15` |
| approved=false review | `62b369571e0a92074517b52060e4a00b7a5df9cdf57173e993dd6e491a1e1306` | `ab8e936bc0998860bce1add2aa180a8f3d7d38c8301788913a1048544b3ceb79` |
| identity audit | `976311217570a5cab295520779ac399790cacd1d5fc1184b0cfb62ee7fee8c11` | `a3d80ebffb9d8358def2d6ceb693fffd4f7764e63cd027f6e22b374d8f95526e` |
| preparation audit | `4751cf14e8d51851b815c5fb371fda9d86a73edb835c46db0e11f0b2fde1ace8` | `cc8005f49fe8c7aa48055e0046016a519d19d14d86500b5a1b42f843d117f9bf` |

domainはidentity_summaryの各entryに記載する。productionのplan/auth/reviewは凍結 `identity.fingerprint` の規則で計算する。
manifestは自己参照を避けて自分以外を一覧化し、そのbytes SHA-256と
`h4-input-generation-artifact-manifest-v1` domain fingerprintを外部の最終報告で示す。

## 最終レビューで解決する事項

1. **CPU許可は未確定。** 利用者は「CPU許可は未確定として草案を固定」を指定した。
   保存allowed_cpusは空。観測process CPU0–255と候補0–5は許可ではない。
   明示許可が確定した後にauthorization/review digestの再bindingと別レビューが必要。
   `process_cpus <= allowed_cpus` を満たす別承認済みlaunch contextも必要である。
   CPU0–5を仮に許可しても現在process CPU0–255のままではguardを通らない。affinityやguardを変更していない。
2. **現在contextのmemory観測は未成立。** sourceの `resources.observe_memory()` は上位cgroupの
   `/sys/fs/cgroup/memory.max` 読み取りでENOENTとなった。host MemAvailableを有効availableの代用にはしない。
   既存sourceのままこのcontextでlaunchした場合、科学入力やoutput作成の前にresource観測でSTOPする。
   rootを飛ばす、値を推測する、sourceや共有cgroupを変更することは今回行っていない。
   observer挙動/実行contextのreviewが必要。source修正が必要と判断されればこの草案で進めず、別source修正・再固定へ戻る。
3. **準備時の観測はlaunch時のfresh admissionではない。** requested workers6、actual workersは未決定。
   5秒以内のCPU/memory/cgroup/pressure/OOM確認と既存admissionを将来launchで行う必要がある。
   AS driver/worker各8GiB、RSS別、headroom16GiB、8+8w+16GiB、累積wall72h、output10GiBを維持する。
4. **最終承認と明示launchは未実施。** actual stage reviewを代行しない。
   保存reviewはfalseで拒否される。CPU条件の解決やmetadata-only模擬成功だけでは入力生成を開始できない。

## gate検査と科学未実行

57 tests PASS、fail/error/skip0、準備と検査は各1attempt、検査失敗0。
実review falseの拒否、schema、fingerprint/binding、commit/source/audit/root/distance/stage/permission/
one-shot/CPU変異の拒否、installed sourceとcheckout byte改変のmock拒否、pure synthetic admission・CPU subset predicateを確認した。
合格authorize/checkout経路はメモリ内コピーのapproved=true＋架空CPU[0]だけ。コピーを保存していない。
real admission/OwnedRun/OwnedPool/execution.launch、production runner、private科学処理は呼ばない。
private science/output/worker境界をmock禁止し、guardカウンタとprotected attemptsはすべて0。

実行済みcommand（認可準備worktree内、将来の科学commandとは異なる）：

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
QISKIT_PARALLEL=false QISKIT_NUM_PROCS=1 \
/home/AbeHiromu/venvs/trotter-common/bin/python -B \
  scripts/resource_applicability/run_h4_input_generation_authorization_preparation.py --mode prepare \
  > artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06/preparation-attempt-01.log 2>&1

PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
QISKIT_PARALLEL=false QISKIT_NUM_PROCS=1 \
/home/AbeHiromu/venvs/trotter-common/bin/python -B \
  scripts/resource_applicability/run_h4_input_generation_authorization_preparation.py --mode tests \
  > artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06/gate-tests-attempt-01.log 2>&1
```

source17＋parent2、契約37、依存45、installed critical source11/compiler metadata、旧source247/保存6 JSON、
準備25・v1 26・v2 30・old source28・parallel11 bundleをbyte/hash照合して不変を確認した。
旧結果/status/validation manifest、原稿/図、Track B、snapshot/runtime/cache/registryも変更対象外。
mainや既存science worktreeの未commit変更を保持する。

追加transpile0、旧系列累積28/64を維持。旧111-suite・全repo tests・benchmarkは実行していない。
分子アクセス/生成、実SCF/DF/state、実signal/sampling/seed/build/compile、GPU、本番runner/worker起動、共有環境/他job変更0。
認可草案を1件保存したが、有効execution authorization0。実科学成立・live worker/資源条件・CI/外部再現は未検証。
生成対象はH4 linear/STO-3G、neutral singlet、4 spatial/8 system、DF fragments12、6距離だけ。
将来6入力をfreezeした時点でSTOPし、別input-bound signal/compile plan・認可・review・launchなしに継続しない。

この資料作成時点はpush未試行。non-force pushの成功とremote SHA一致は別の最終確認が必要である。
認証失敗時は共有環境・認証設定を変更せず未公開として停止する。
