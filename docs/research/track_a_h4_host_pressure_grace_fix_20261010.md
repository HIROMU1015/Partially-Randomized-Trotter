# H4 run08のhost PSI STOP：限定猶予の未承認修正案

run08は約1941.392秒でhost_memory_pressure STOP。compile完了2/予約6/signal1、旧partialを次attemptに再利用しない。
host full PSI avg10=1.26%、全visible nonroot cgroup PSI0、OOM増分なし、effective available約426.24GiB、worker最大AS12.222GiB/RSS11.852GiBで32GiB以内。
全6 owned identityを2回native ABSENT確認。元終了code/正確な終了時刻は未記録で今回の観測から補完しない。
停止後3回の受動観測ではhost/cgroup PSI0・available約466.65GiB。pressure発生者を自分/他jobのどちらとも特定できない。
旧監視実装は承認済み1%以上即STOPを正しく適用した。判定仕様を変えるため、先に限定inactive案を固定し、この一点だけ利用者の承認を求める。

## 限定変更案

host1%以上5%未満だけ、全nonroot PSI0・OOM基準一致・initial namespace/完全v2 hierarchy・effective available152.25GiB以上・観測間隔/freshness各5秒以内の場合に最大30秒の連続猶予を与える。
host<1%で連続warning timerをreset。host>=5%・host>=1%が30秒継続・nonroot非zero・OOM増分・欠測/不正/scope変化・5秒違反・空き余裕不足は即fail-closed STOP。
毎秒のresource observationを止めない。sleep/待機worker/共有設定変更は使わず、monotonic clockの経過時間だけで判定する。
first_failureを保持し、警告で32GiB role上限や所有確認・own-run限定cleanupを迂回しない。起動前は従来どおりhost<1%を求める。
5%と30秒は限定候補値であり、最適性や共有hostへの影響が検証済みとは扱わない。無条件にhost PSIを無視する仕様ではない。
従来host-positive条件のavailable120.25GiBを提案では152.25GiBへ強化、zero-host headroom16GiBと起動時admission152.25GiBを保持する。

## 実装と小さな検証

SOURCE `736bdc15ccfec6b2a715e1723850a481f0f2ff6b`、closure69。旧production67pathsは全byte不変、新inactive module/test2件だけ追加。
pressure_grace_proposal.pyは純粋な判定試算で、既存production pressure_policy/observer/launch_bindingへ接続していない。
approved=false/runtime_authorization=false/production_wiring_present=false、allowed_cpus=[]、未seal、absolute_launch_command=null。本計算再起動0。
14pure回帰PASS、fail/error/skip0、wall0.003146秒、peakRSS24,117,248B。通常1process/thread1、AS256MiB/RSS128MiB/wall30秒/output1MiB、実child/worker/observer/affinity/科学array/回路build/transpile0。
検査は1/5/30の境界・回復timer reset・間隔5秒・nonroot各scope・OOM/152.25余裕・不正/欠測/hierarchy・firstSTOP保持・旧production predicate1%即STOP不変を確認した。
run08の観測値0/.18/.32/.63/.88/1.26を人工scope/time fixtureに用い、synthetic回復を追加した。このfixtureは停止しなかった未来のpressureを復元した証明ではない。
旧53/11campaign・benchmark128・synthetic28/64を繰り返さない。今回は新2filesだけの差分reviewに絞る。

科学H4 linear/STO3G/DF12/6凍結入力/218templates・compiler options・既存private venv・CPU4[2,4,5,6]/driver16/observer18・worker32/driver8/observer256MiB/64MiB・17GiB/74805/72h・carry0不変。
共有host/cgroup/sysctl/他job/GPU/入力生成に変更なし。新work/log/tempはhomeだけ。NPZ/科学runtime/cache/credentialはcommitしない。
承認後はproduction接続・actual新SOURCE/seed/profile/authority/plan/auth/review bindingとfresh unused run/CPU/memory/pressure/OOM/fs/inode/quota/one-shotを軽量に固定し再実行する。
旧費用/証拠は保持し次attemptはcarry0。全map完走、旧compiler完全同一性、既知native stderr/formatter診断欠落2件の解消は未検証。
[資料・変更案・検査・停止監査](../../artifacts/resource_applicability/track_a_h4_host_pressure_grace_fix/2026-10-10/README.md)。

新2filesだけの独立差分reviewはPASS_INACTIVE_PROPOSAL_DELTA_ONLY、blocking指摘0。旧source/environment/input/proofの全面review・campaign再実行0。
review SHA256 `0e968b65c9093df1ebabf024b62fc72d52d6f55399ec291e420154f46dc912fa`。nonblocking検査scope注記はhost<1%非zeroでのtimer resetとexact5秒positive caseの単独test不足だけで、実装境界に不合格なし。
