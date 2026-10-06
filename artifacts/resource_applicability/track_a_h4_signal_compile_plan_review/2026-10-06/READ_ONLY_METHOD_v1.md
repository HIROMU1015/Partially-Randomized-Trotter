# Read-only metadata preparation method

この記録は科学runnerではない。今回実行したstdlib/固定gate metadata scriptを
[preparation_metadata_only_v1.py](preparation_metadata_only_v1.py)へ保存した。
scriptは/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-worker-bootstrap-run02-20261006からidentity/gates/resourcesをimportし、
generation-freeze.json・byte-budget.journalをread-onlyで確認した。NPZ/NPY/cacheはaudit hookが拒否し、
subprocessはgit showだけ許可、taskset/worker/exec/affinity操作は拒否した。
checkout_gateのQiskit importはsignature/plugins metadataだけで、transpileを呼ばない。
JSONサイズspecimenは値幅を測る仮値で、科学recordとして保存せずsignal関数やtrajectory seedを呼ばない。
観測logは[preparation_execution.log](preparation_execution.log)へ保存した。
これはローカル準備検査の証拠でありimmutable CIや本計算の検証結果ではない。

filesystemは既存artifacts上位directoryにos.statvfsを使い、f_bavail/f_favailを採用した。
既存installed headerのquotactl_fd syscall番号443、Q_GETFMT (user/group/project)だけを使用した。
project属性は既存directory fdへのFS_IOC_FSGETXATTRだけ。設定・sync・quotacheckを呼ばない。
既存read-only getterのsourceを[storage_probe_reference_v1.py](storage_probe_reference_v1.py)へそのまま保存した。
SHA-256：c2e2f412f8a6da08ecb5eac4791dcdf2948eaa3c47b1c9fd93588d468cdae547。
この参照sourceのmainは旧準備記録であり再実行しない。今回はrunpy.run_path(run_name='readonly_getter_only')
からquota_and_filesystemだけを呼んだ。元の/tmp/h4_storage_stage_review.pyも変更しない。

resource観測は固定observe_memoryでhost＋全可視非root limits、PSI、OOMを読む。
CPU topology/onlineは/sys、候補6 coreのsiblingsの3秒負荷sampleは/proc/stat、process許容集合は/proc/self/status。
広い準備process affinityからCPU使用許可を作らず、観測を記録するだけ。共有環境・他jobを変更しない。
1秒監視を72時間実行した検査ではなく、72時間の最大出版数を整数算術で見積もった。

再現はcommitのscript/JSON/source19/hash/metadata logを照合する。
既存bundleを上書きしてこのscriptを再実行しない。fresh観測が必要なら別の日時・別のcontrol証跡へ
read-only gettersだけを実行し、この草案と過去観測は保持する。
