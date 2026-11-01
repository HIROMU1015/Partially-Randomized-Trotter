# P3 Stage 3：保存済み未取得13件のprocess分離回復

2026-11-02 JST。基点はrun02結果`2da49fa4e8578227c130fc6061a0b3061276f893`、
同runの科学SOURCEは`9c4006bff3e76ea2ba7fbe4a10e679a8693c3a5c`。
ユーザーは未取得13件に限定したprocess分離回復とStage 3の36件完成を明示指示した。
旧run01/run02とStage 2のSTOP・SOURCE・結果は保全する。

対象はlinear H4/1Å/STO-3G/中性singlet/system8/full256、保存DF tol-only1e-8/rank10、
案A、M_B/M_C、T=.8/S2/q8/r4/K4、P=0、cosine/+Im sineの固定cohortのみ。
新namespaceは`artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run03`。

## 再利用と唯一の変更

run02の`pending_targets.json`に固定された13 targetと、同commitの`native_evidence.zip`内の
保存input QPY原bytesを使う。入力は圧縮bytesとdecoded QPYの両hashを照合する。
新sampling、trajectory/event選び直し、native回路再生成、81項目のnative preflight再実行、
取得済み23件の再compileは行わない。run02の23 metricsはそのまま完成cohortへ継承する。

[回復runner](../../scripts/recover_h4_p3_stage3_process.py)は既存Stage 3の`price`とQPY保存関数を呼ぶ。
同じ専用runtimeとcompiler（Qiskit1.3.0、rz/sx/x/cx、level1、seed17、無接続制約）を使う。
同一process内のthread並列transpileを廃止し、spawnした新interpreterに1 targetだけ渡す。
`max_tasks_per_child=1`によりtaskを完了したprocessを再利用しない。
独立processは空きCPUに応じ並列化し、共有環境や他利用者には変更を加えない。

## 検証・実行・成果物

[人工tests](../../tests/test_h4_p3_stage3_process_recovery.py)でsaved-QPY破損検知、
worker数より多いtaskのprocess非再利用、既存compilerとの価格/fingerprint一致、
23件の非上書きmerge、予定済み`F-R-E+Empty`式と軸対応を確認する。
これは人工実装検査であり、H4の全compiled operator作用保証ではない。
最終人工検査は4 PASS。live compiler dataclassのtupleと保存JSONのlistの表現差は、
同じJSON表現へ正規化して照合する。compiler設定・hash・回路・数値は変更しない。
人工検査後に新SOURCE commitを固定・公開し、保存13件を各1回compileする。
科学入力と実際に使うSOURCEを照合し、無関係worktreeの更新は停止理由にしない。

新13 output QPY/metricsとworker別PID・例外、23+13の36価格と由来、予定済み16非加法性診断を保存する。
各行と観測min/maxまでを表示し、母平均・tail・統計的方式順位・新B/N/Gは算出しない。
native検証の再利用と新compileの証拠を区別し、旧23件がthread競合と同じprocessで正常返却された来歴も残す。
36件取得は全compiled作用精度や全lawρの保証を意味しない。

失敗時は各targetの例外と正常出力を保全し、既に走るworkerを回収して自動retryせずSTOPする。
成功時は36価格・非加法性・未解決事項を報告し、可逆QPY保存・remote照合後mandatory STOP。
finite bias未解像、finite-bit law平均、母平均費用、同精度総資源は未解決のまま。
追加標本・新条件・Stage 4・主要claim採択・論文改訂へ自動移行しない。
