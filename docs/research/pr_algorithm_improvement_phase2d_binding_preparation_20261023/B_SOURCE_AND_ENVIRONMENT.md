# Bのactive SOURCE・専用環境

## Active closure

元I/P0のSOURCEを変更しない。追加はB package三file（__init__/contract/engine）と専用runnerだけ。Bが実際に呼ぶ経路は以下の10fileへ限定し、bytes/SHA256/Git blobを[SOURCE_CLOSURE.json](SOURCE_CLOSURE.json)で固定する。

- B __init__.py / contract.py / engine.py、scripts/run_pr_dictionary_binding.py。
- I __init__.py / fixture.py / algebra.py / prototype.py / cases.py。
- 固定src/trotterlib/rte.py単独。

Iのfixture、S/P代数、canonicalize_arm、atom_record、catalog_atoms、bind_motif_labelsを再利用する。prototype.make_tail/configureをBから呼ぶと広いtrotterlib package initializerへ到達するため、新B tail_bridgeで同じRTEComponent fieldsを実際の固定rte.pyへ渡す。古い科学SOURCEは改変しない。

rte.pyは2978参照と同bytesで、top-level importはNumPyとstdlibだけ。Bでは固定fileをprivate名 `_pr_dictionary_binding_frozen_rte` へimportlibで一回loadする。元trotterlib moduleをsys.modulesへ偽装登録せず、class instanceをB/F1間でpickle転送しない。F1へ渡すのはJSON records。loadはB専用permission gate後、最初のCONTEXTS case内だけで行う。今回このloadを実行していない。

whole-moduleの保守的import graphでは、prototype/casesの**未使用関数**にtrotterlibやevent helperが見える。active closureは関数単位で、make_tail/configure/ordered_event/finite_events/motif_eventsはBで呼ばない。検査はその区別を保存する。

BではIのcircuits/reference/pricing/intervals/gates/authorization/execution、F1 runner、testを起動しない。2978 sequence builder、f29 support completion、strict DF wrapperとの差はIの限定portとして保全し、Bからnative Gaussian helperも呼ばない。recipe policyのSOURCE hashはmetadataとして参照するが、actual recipeを生成しない。

## 環境固定案と未解決

最小B環境案は**専用venv、Python3.11.9、NumPy1.26.4**。SOURCEが要求する>=3.11を満たすexact patchを提案として固定した。これは既に構築・検証された環境ではない。NumPyはRTE scalar metadata／型検査のためだけで、BLASによるmatrix計算を追加しない。

SciPy、Qiskit1.3.0、Qiskit Nature0.7.2、mpmath1.3.0、OpenFermion/PySCFはbasic B active pathでは不要。F1の広いimport closureには必要となり得るので、B最小envをF1 env lockの代用にしない。F1の候補pin自体もruntime検証済みと表示しない。

B-PREではstdlibで既存interpreter/config/site-package候補をread-only確認したが、調べた場所で適合するpackage metadataを取得できなかった。host全体にpackageが存在しないと断定せず、観測を[ENVIRONMENT_PLAN.json](ENVIRONMENT_PLAN.json)へ限定記録する。科学module import、venv作成、package install、共有設定変更はしていない。

実行承認前に、専用Python patch/実行binary、wheel filename/hashとplatform、NumPyとtransitive package RECORD files、stdlib/native extension、BLAS/shared libraries／libc／libmの必要closureをprivate receiptへ固定する。publicにはversionとreceipt digestだけを出し、内部絶対pathを出さない。実sourceの再利用と新B sourceを含むcommit／file closureもfreezeする。

future gateはPythonのisolated mode（`-I`）、専用venv、system-site-packages false、package version／file hash／native hashをscience import前に照合する。NumPy/RTEの動的APIが実際の5case内で使えなければ即STOPし、追加probeやretryをしない。これを独立科学validationのPASSと呼ばない。

現主状態は **B_ENVIRONMENT_UNRESOLVED**。Python/wheel/native identityが未確定のままBを承認可能と表示しない。環境準備を別途具体化し、数値処理なしのread-only保全を済ませた後にB科学実行を判断する。
