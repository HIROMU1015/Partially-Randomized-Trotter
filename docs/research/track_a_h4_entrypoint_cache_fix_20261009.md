# H4 run03停止原因のstevedore disk cache修正

run03は利用者の17GiB/74805承認で一度起動し、最初の科学compile中のplugin managerのwrite guard拒否でFAIL_CLOSED_STOPした。
実行SOURCEは`6e68fd9bcc68e788db6f5d43eaa6a03866e53d3b`、認可artifactは`4e8d113533fd880840bb8f6570ae6a636e916615`、
STOP結果は`56250fd78849277827b9a1f36fafbbd5d1a7e2bc`。
driver2930049/observer2930256/worker12全14identityを2回ABSENT確認し、exit143。新予約1/完了wrapper0/signal0。
今回は`worker-log-first-stop.txt`に元worker例外を保存し、observer first STOPへreport/fsyncしてからown-run停止した。
[一度の実行・native停止証拠](track_a_h4_production_run03_20261009.md)を保持する。

## 必要な修正と限定検証

利用者の「実際に本計算をしていき問題が出たら解決していく」方針に沿う必要bugfix。
branch `track-a-h4-entrypoint-cache-fix-20261009`、新SOURCE `cd162e9305143d81c28908b9732f8eef12cd6b89`、closure46。
旧run03 worktree/source/output/control/one-shot/失敗証拠を保持する。
詳細：[軽量audit bundle](../../artifacts/resource_applicability/track_a_h4_entrypoint_cache_fix/2026-10-09/README.md)。

保存されたworker tracebackはQiskit `generate_preset_pass_manager`→`level_1_pass_manager`→`PassManagerStagePluginManager`でwrite guard拒否。
既存venvのstevedore `_cache.py`はplugin entry point metadataのcache missでdisk directoryをmkdirし、cache JSONをwriteする。
既定場所はprocessのXDG設定またはhome `.cache/python-entrypoints`。guardは予算外worker writeを拒否するため、
compiler最初のplugin registry探索でSTOPする。actual denied pathnameは旧messageに未記録なので、既定pathはinstalled codeからの推定。
同じworker guard下のmetadata-only cache missで同一mkdir拒否を再現した。
実証済みなのはQiskit plugin manager中のguard拒否。具体的なstevedore cache mkdir原因はinstalled sourceの直結経路と同guard再現に強く支持された推定で、
旧messageが1024Bに切られたためstevedore frame/event/targetは実traceから直接確認できない。元run02原因や実compile成功の証明とは扱わない。

library cache v2 profileは専用home directoryの`python-entrypoints/.disable`0B、単一link・byte SHA、stevedore version/source SHAを固定。
初回stevedore cache import前にown-process/childrenだけの`XDG_CACHE_HOME`を設定し、dependencyが対応するdisk cache停止を選ぶ。
既存venv/library source/ユーザー設定/HOME/他jobを変更しない。既存MPL cache routingとlegacy v1 profileを保持する。
disk cacheを使わず、worker内メモリcacheを保持する。write guardは新entrypoint cacheのmkdir/write/rename等を許可しない。
先行cache import/変更sentinel/extra file/symlink/version/source変更はfail-closed。
worker拒否messageにはeventとpathを512文字上限で加え、次の原因を判別しやすくした。

限定8件PASS、errors/failures0、wall 0.078717s、peak RSS 28,835,840B。
6 plugin群init/layout/routing/translation/optimization/schedulingで登録entrypointの順序・name/value/groupがimportlib.metadataと一致。
2回目のmemory cache取得も同一。修正後write attempts0、enabled disk pathでの期待mkdir STOPを1回記録。
通常1 process・数値内部thread1、AS2GiB/RSS256MiB/wall60s/output4MiB。実child0。
Qiskit/科学imports0、科学array/回路build/人工compile/transpile/実worker/affinity/GPU/追加本体起動0。
最初の8PASSも保持し、output合計測定を追加した最終sourceで8PASS。旧32/39回帰・synthetic28/64・benchmark128は再実行しない。
plugin metadataの一致を実科学compile成功や旧compiler出力完全同一性の証明にはしない。

## 次の実行のbindingと予算

run03後carryは22 consumed/reserved actual /12,956,511,264B /6004.111340102032s conservative upper、残actual74783。
壁時間upperはpost-stop監査待ちを含む。正確な終了時刻を補完せず、失敗費用とreservationを返却/resetしない。
承認済み17GiB/74805は維持。今回の一度の認可は消費済みで、新SOURCEの本計算は起動しない。
sourceのRUN_ID/CARRYは消費済みrun03/CARRY21のまま、新sourceによる次launchは未binding・未認可。
準備bindingはapproved/runtime false、allowed_cpus=[]、sealed=false、absolute command=null。
次にはrun03 native proof/carry22、新未使用run/output/control/one-shot、最終SOURCE/seed/profile/plan/auth/reviewとfresh gateの再結合が必要。
旧partial/random/cacheを混合しない。候補v2 policyはcompile options/environment/compiler fingerprintを変更しない。

新carry込みの全map charge boundは21,693,482,896B
（20.203630GiB）、actual22+74784=74806。
整数GiBの最小proposalは21GiB/74806、余裕855095408B。これは未承認提案であり、上限を変更していない。
private .disableはpayload0B・追加2directories/1inode、既存metadata余裕を使う。監視72h前払いreserve/他caps/periodは変更しない。
自動retry/次stage/入力再生成/GPU/共有環境・venv・他job変更なし。

## 固定・独立review・公開

source/tests4件をSOURCEへ、46旧/new blob/hash・v2 policy・8結果・carry/STOP参照・独立reviewを別REVIEWへ固定する。
NPZ・actual runtime/checkpoint/cache/test log・0B sentinel・credential・内部SSH・private utilityはcommitしない。
独立担当のreviewはbundleの`independent_entrypoint_cache_review_v1.json`を参照する。
起動したrun03のSOURCEと新cache fix SOURCEを区別する。本修正による追加本計算起動0。

独立review：`CACHE_POLICY_IMPLEMENTATION_PASS_NEXT_LAUNCH_UNBOUND_UNAUTHORIZED`、blocking実装所見なし。
SHA256 `f445a9aa3b6b71ab47e0fb31aa5699a55963d7dab278fa9bc64ab9fc52db0d49`。元compile成功・次runtime認可は未検証/未発行。

認証失敗時の手動push:

```bash
git -C /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-entrypoint-cache-fix-20261009 \
  -c maintenance.auto=false -c gc.auto=0 \
  push origin HEAD:refs/heads/track-a-h4-entrypoint-cache-fix-20261009
```
