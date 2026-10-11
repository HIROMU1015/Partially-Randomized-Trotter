# H6 次回142 group：4 worker準備 v2

利用者の「4ワーカーでよい」に従い、2 worker案から4 workerへ変更した。
SOURCE `df2976f05c97c8c777f9a6264e83a53b637fef65`、branch `track-a-h6-four-worker-preparation-20261012`。
REVIEW SHAはこの資料を固定したcommitの `git rev-parse HEAD` で確認する。利用者がpushする方針を維持する。

[設計](../../../../docs/research/track_a_h6_parallel_preparation_v2.md)、
[技術seal済みmanifest](sealed_v1/manifest_candidate_v1.json)、[未認可grant草案](sealed_v1/authorization_draft_v1.json)、
[変更理由・4 worker指定](scope_change_v2.json)、[binding監査](binding_audit_v2.json)、
[closure/blob/hash](source_closure_audit_v2.json)、[38限定test結果](test_result_v2.json)、
[Phase A完了監査](phase_a_completion_audit_v1.json)。

## 実装と資源配分

| 対象 | CPU | AS soft/hard | RSS停止閾値 |
|---|---|---|---|
| worker0 | 20 | 32/32GiB | 9GiB |
| worker1 | 21 | 32/32GiB | 9GiB |
| worker2 | 24 | 32/32GiB | 9GiB |
| worker3 | 25 | 32/32GiB | 9GiB |
| coordinator | 22,23 | 2/32GiB | 2GiB |

4 workerの指定を反映。合計RSS40GiB・job RAM予算64GiBを増額しない。
(40−coordinator2−余地2)/4=RSS9GiB/worker。AS32GiBは仮想address上限で、そのまま維持する。
RSS9GiBは1秒間隔/max5秒の観測STOP閾値で、RAM予約やkernel hard capではない。
前回B1 q8の実測peak RSS7.929GiBに約1.07GiBの余裕があるが、残142 groupの最大RSSは未測定。
9GiBを超えるgroupはSTOPする。4 workerで全groupが完走する保証はない。
より大きいRSSが必要なら、実測STOP証拠に基づきworker数/メモリ配分を再検討し、上限を勝手に増額しない。

CPU20/21/24/25は別physical core。H4の現worker2/4/5/6・driver16・observer18と重ならない。
割当はown processのaffinityのみ。CPUを他ユーザーから予約・排除する設定は行わない。
内部thread1・Qiskit num_processes1、起動nice10、既存host/cgroup memory/PSI/OOM/headroomを維持する案。
Coordinator硬AS32GiBを子へ継承し、各workerでsoft/hard32GiBを適用する。

## 科学条件・入力・carry

科学SOURCE `5a903aea2492c2d211c1e15e7737df0d50694c12` の224 pathsと旧control/runtime233 pathsは変更0。
既存group poolは任意completion orderを受理し、保存/集計はoriginal index order。
各group内のcost_task・cosine/sine順・seed・trajectory・full-wrapper・compiler・費用定義を維持する。
Source closure237。2 worker SOURCEとの差分はcontrol/preparerとexecution docstring、testのみ。
Schemaとrunner/module公開pathは既存v1を維持し、resource値・SOURCE digestで旧grantを拒否する。
新v2 preparation/output/one-shot lockを使い、旧2 worker資料とruntime namespaceを保持する。

既存H6専用venv/profile/33凍結input/5220旧rawを再照合し、既存venvへinstall/update/設定変更はしていない。
Profile SHA `4e32bb9ba57c38280562390100ebf916e9858218534504ead1519625a609bd75`。
Science plan/compiler hash・保存state・normalization/shot・92候補/4精度は不変。
旧compiler outputのbit完全同一性は未検証として保持。

Phase A447は正常完了/2 wrapper/worker・coordinator exit0/cleanup済み。新監査は元の完了proofを再使用する。
旧449＋447=450/592 group・904 wrapper。次の固定対象はindex450–591の142 group。
Carry compile904/control_probe292/trajectory434/occurrence1024/primitive1470/reference_matvec400。
追加static demand compile284/control_probe68/trajectory142/occurrence1136、旧2 worker案と同一。
既知合計compile1188≤1712/control_probe360≤1024/trajectory576≤832/occurrence2160≤6656。
過去未記録actualはunknownのまま保持し、0補完・budget resetはしない。

Output2GiB/worker単体32MiB/全worker log1MiB/journal32MiBを維持。
同時worker用128MiBとterminal余地をoutput admissionで予約する。
Phase A単体52 files/26569bytesは参考値。全142 groupのcache/tmp/log/block/inode使用量は未測定。
単体上限32MiB×142の最大和4.44GiBは全体2GiBより大きいため、容量完走は保証せず全体guardでSTOPする。
Phase/total wall capは旧nullを保持。ただし監視journal等の有限上限到達ではSTOP。
起動前filesystem10GiB/100000 inodes・quota/CPU/host216.25GiB/effective64GiB等のfresh gateが必要。
この準備ではfuture launch preflightをPASS扱いしない。

## 検証・承認・公開

38 local metadata/mock/tiny-process tests PASS（1.12秒）。
4つの最小processの同時稼働、全PID wait/reap・log/journal閉鎖、worker RSS9GiB/aggregate40GiB超過、
旧2 worker grant拒否、CPU/slot/予算整合、first STOPと非ESRCH cleanup error、既存seed/compiler/admission/bindingを検査。
Driver AS2GiB/thread1/wall120秒/file16MiB、tiny child最大4・各AS128MiB、全一時領域home内。
人工transpile0/科学array読込0/新科学runner0、新approved grant0。
限定差分確認で、独立review・immutable CI・実4 worker speedup検証ではない。

4 worker指定は反映済み。142探索のlaunch指示・一回grantはまだ発行していない。
Technical seal=true、approved=false/runtime_authorization=false/allowed_cpus=[]、未使用output/lockを維持。
Absolute launch commandは未確定/null。次回は新SOURCE/profile/input/carry、remote REVIEW、fresh resource gateを照合して一度実行する。
完了/最初のSTOP後mandatory STOP。自動retry/resume、入力再生成、selection/map・独立32確認・H8へ進まない。
共有環境/既存venv/旧H4/H6/旧2 worker worktreeは変更0。軽量SOURCE/test/review資料だけを固定した。

Pushは利用者側：

```bash
git -C /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-four-worker-preparation-20261012 \
  -c maintenance.auto=false -c gc.auto=0 \
  push origin HEAD:refs/heads/track-a-h6-four-worker-preparation-20261012
```
