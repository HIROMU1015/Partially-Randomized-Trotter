# C1-5A：H4限定・3方式の探索的資源診断

ユーザーの2026-10-21指示で承認されたone-shot。固定基点 `af75af03c0b334c29b5a4311bd2b4578a999460c`、branch `track-b-c1-5a-h4-limited-resource-pilot-20261021`。

Ordinary / General Full / Closed P5を同じR1 finite Taylor targetで比較する。各方式1024 whole-path、各4 microstep、計3072 path / 12288 microstep。探索試行数であり十分Nではない。量子測定値・±1を生成せず、coherent-signal estimatorを完成したとは扱わない。

- [実行契約](execution_contract.json)：H4 1.00 Å/STO-3G/DF12、R1新partition（LD3）、8 system qubit/13 basis/324 label、T=4/5、q=1、r=4、m=5、sigma=+1、microstep時間1/5。H2048/K4096、eta=rho=tau=10^-12、角度K256/J64を固定。
- [SOURCE/入力/環境](source_manifest.json)：旧科学SOURCEとR1を変更せず、新しいdiagnostic entrypointを独立追加。SOURCE公開固定後の新authorization-only直接子から実行する。旧grantを再利用しない。
- [技術検証](technical_tests.json)：人工12件PASS。2-process保存・bit列再現、両zero理由、4 step会計、weight積、元degree位相、strict controlled人工operator/語順/scalar、failure/未完了を確認。H4科学runは未使用。
- [実装](diagnostic.py)・[one-shot runner](run.py)：既存samplerを実行し、全4 stepを先決定。zero pathの回路構築を回避し、古典費用は保持。phaseはraw Taylor degree、演算子はreversed reduced word→child回転。実πW=αをexact検査。

測る論理回路は**`TAIL_ONLY_PLUS_SCALAR`**。既存R1の保存deterministic halfは旧correctnessでも全接続されていないため、今回新しいHamiltonian/block再構成で埋めない。scalarのconstant+抽出identityに対するcontrol上P位相はT=4/5で一度計上する。outer S2全体、state準備/readout、native T/CX/1Q/depthはUNKNOWN。opaque controlled gateを含むraw論理instruction/depthでありnative費用ではない。

PRNGはseed20261021とmethod/trial/microstep/draw/blockをSHA256でkey化する。bit幅・実bit値・規則・key identityを保存し、並列順に依存しない。数学的fresh IIDを実証したものではない。

3 method coordinatorを並行実行し、各setupをread-only forkで共有する。各methodのworker数はCPU affinityと32 batchから同じ規則で選択。各workerのangle/event cacheはcoldから始め、同じ規則でmemo化する。既存samplerと`dyadic_index`は変更せず、P5のO(L²) group scanも残す。したがって最適selectorを持つ最強実装の普遍的CPU順位を示す実験ではない。FullはF3 preflight/guard、canonicalは未認証の既存local guardを持ち、この認証負担差も実装境界として記録する。

計時はinput/registry setup、normalizer/selector setup、sampling/検査/event Evidence保存、angle lookup/query、logical build/accounting、path Evidence、batch圧縮保存、最終検査を分離。CPU・wall・hierarchy CPU、RSS、保存bytesを記録する。forkの共有ページがあるのでRSSをprocess間で単純加算しない。setup/online/終端と未計上差分は区別する。環境・同時実行・cache条件に限定した観測値である。

[結果terminal](result/terminal.json)が完走を表す場合だけ、[比較JSON](comparison.json)・[CSV](comparison.csv)・[図](comparison.png)・[handoff](handoff.md)を参照する。rawは`result/<method>/batches/*.jsonl.gz`、exact event/angleはSHA256名の圧縮JSON。`Evidence`はhex整数/有理数を用い、gzipはmtime=0。gzipを展開するとF3 exact typed JSONまたはJSONLが得られる。

新SCF/DF/Hamiltonian生成、native backend compile/transpile/synthesis、shot/測定Monte Carlo、十分N/厳密資源winner、新非自明証明、他分子・precision探索は行わない。Qiskit logical Gate.controlが内部definition passを使うことはあるがnative backend/価格取得は呼ばない。

基本law/identity不一致や科学run中の失敗では原ログ/prefixを保全してSTOPし、SOURCE修正・再実行を行わない。完走後も公開HEAD/bytes/保全/clean確認後mandatory STOP。C1継続・縮小・終了の判断はユーザー・GPTが行い、後続の新scope・実行は別承認。
