# H4 DF10 Mq：同一δ・別qの独立検証

2026-11-10 JST。今回の利用者指示§6–8による科学実行。S1 carrier `cefc01622a3cd75dd16d30de7b9fb68274f3eddd`のMqを凍結する。bPF=606.75、bPR=176.88888888888894、各X/Y同値。新fit・特徴探索・MqB比較は行わない。

- Linear H4/1Å/STO-3G、保存tol-only1e−8 DF10/sector(2,2)、保存DF-H ground target/数値state条件付き/P=0。既存snapshot/H/state/fragment順/law/scalarを変更しない。
- PF B0/S2/p7：qPF32/T3.2/δ.1、決定的1 circuit×X/Y。PR S2/p1/R4/K2：qPF16/T1.6/δ.1、新33 trajectory×X/Y。元symmetric_directional・測定付きfull Hadamard wrapper。最大68新wrapper compile、top-up・q64以上なし。
- 元compiler：Qiskit1.3.0、rz/sx/x/cx、opt1/seed17、hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。新N/signal資格/N×C/G・全13round価格・Main Claim変更・詳細科学レビューなし。既存εE=.00159360010199040、αtotal=.05/方式、βtotal=.4、h=.1、13roundは不変。
- 事前資産調査：155529e2の既知6価格表に対象同δ・同qの価格なし。PR q32/T1.6/δ.05は対象外。元compiled_cacheは読取専用で、実際に組んだwrapperのcanonical numeric fingerprintとcompilerが一致するentryだけ再利用。missは自分専用cacheにcompileし、元cacheへ書かない。
- 元155529e2の`sample_requests/compose/cost_trial/native_context`、provider c31ef424のnative wrapper/compiler/fingerprint、V1のdescriptor/counterを再利用。V1 `predict`の最初の2つのq guardだけを今回scopeに差し替え、残る関数本体とAccountingは変更しない。旧qで人工一致を確認する。V1とS1ファイルは不変。
- 新cohort `H4_DF10_V2_INDEPENDENT_Q_20261110`。元trial_seedのSHA256(JSON([H46_LONG_TIME_20261025,cohort,candidate_id,index,outer,piece]))先頭64bitを使い、path0…32・outer0…15/piece0で固定。既存H4 registry/exclusionと元long-timeの既知cohort seed集合に対する重複0を要求。全履歴の完全被覆・PRNGの数学的独立性は主張しない。
- 同一runで全33 trajectoryを一回生成・private保存し、全68軸のM0/Mq予測をPREDICTIONS.jsonへ保存してから直接価格を取得。workerは生成済みrequestを受け取り、再samplingしない。測定価格で予測を変えない。
- 元index0の低control・高control・occupation-oracle・Hadamard readout・compiled statevectorの固定correctness検査を先行し、代表PR成功後に残32 pathを並列実行。これらはbuilder/compile検査で、新しいRPE signal/energy資格の評価ではない。元tol 1e−9、compiled-vector1e−7を維持する。
- process数は独立34 jobと実際のCPU affinityから選ぶ。Qiskit内compileは1 process、BLAS/OpenMPはworker内1 threadとしてnested oversubscriptionを避ける。共有設定を変更しない。PSI/他worktree変更だけで停止しない。
- 固定SOURCEを実行前にcommit。private raw/cacheは `/home/AbeHiromu/projects/h4-df10-price-v2-independent-q/20261110/run01`。既存directory/出力を上書きしない。失敗をSTOPとして保存し、自動retry/resume・追加cohort・係数変更をしない。開始済みの他の有効taskは保全して回収する。
- RESULT/CSVにdirect価格・M0/Mq・signed/absolute/relative error、bias/MAE/RMSE/max、PR残差variance/SD、finite-cohort平均の記述的SEを保存。X/Yを対応する応答として扱い、n33をn66へ倍増しない。S1のq8結果は固定保存summaryから引用する。
- 小残差でも係数不確かさ・systematic・希少eventを0扱いせず、law期待値/長q/q4096/全RPE費用/CIは未資格。残差0は診断、一律1/2/5%閾値は採択しない。
- GitHubへ公開しremote SHAと主要成果物を照合後mandatory STOP。V1/V2設計/S1、旧SOURCE/input/raw/grant/ledger/marker/KEEP/STOP、他worktree/process・共有環境は保全する。
