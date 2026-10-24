# H4 D1：採用SOURCE基準・保存full QPYによるone-shot

2026-10-24 JST。ユーザーの「次の作業に進んで大丈夫」を、直前に具体化した
SOURCE基準Bの採用と、保存full4＋隔離/empty5・最大9費用compileの承認として受けた。
[承認記録](../artifacts/non_df_h4_d1_source_execution/2026-10-24/approval.json)。
これは旧grantの再使用ではなく、新SOURCE・新namespace・新one-shot grantによる有限D1である。

固定H4/隣接1.00Å/STO-3G/n8、Pauli184/prefix27、DF tol1e-8/rank10/prefix1＋one-body。
S2/T=.8/q8/r4/K4、cosine・trajectory_index0、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17。
Full Hadamard wrapper・状態準備費用除外、DF raw_concatenationを維持する。
旧N/G null・C/L BLOCKED・compiler certificate欠測・元native UNKNOWNは変更しない。
別系列H2/Yoshida4やD2/D3へは進まない。

## 新接続と科学的な主張範囲

基点`f173b76c7aba56012cbf1d51cb8f27dd062b3bb2`をGitHubで取得確認した。
旧科学SOURCE0c329b30、保存D1 SOURCE48d4d27a0を保全し、
新しい[controller](../src/trotterlib/h4_d1_source_control.py)と
[runner](../scripts/run_h4_d1_source.py)だけを接続した。
元`h4_d1_adapter`、recursive fingerprint、Pauli/DF primitive、wrapper/compiler、
CallBudget/durable save/native witnessは再利用し、既存SOURCEを改変していない。

Stage Aは既存の読取inspectorを流用してSOURCEの12辺・保存tokenを照合する。
4つの保存full QPYをbyteコピーし、現在recursive hashを保存D1 hashと照合する。
hash処理に必要なlazy local definitionの復元は行うが、新しいfull emitterやcontrol再構成は行わない。
旧hashの3一致／Pauli-PR不一致を独立に残し、採用済みSOURCE基準Bによる対応と混同しない。
QPY自己一致だけで旧回路同一性を認定せず、旧payload・不一致原因UNKNOWNを維持する。

全4入力gate後だけfullを元順序で各1回compileし、旧RZ
255500/9652/135355/23021の最初の不一致でSTOPする。
全4価格一致後だけPR retained2/event2/empty1を各一度build/compileする。
phase所有・signed相互作用式は[旧設計契約](../artifacts/non_df_h4_d1_reproducibility_design/2026-10-24/contract_DRAFT.json)を維持する。
隔離回路は有効な新simulationではない。全RPE費用・母平均・一意の因果寄与率・certificateを主張しない。

## 人工検査・封印・実行状態

新controllerの[人工10件](../tests/test_h4_d1_source_control.py)と既存17件がPASS。
未採用/無grant拒否、旧grant/STOP保全、full build0、予算非返還、現在QPY不一致保存、
Pauli-PR旧hashFAIL維持、Stage A拒否でcompile0、最初の価格不一致で後続0、
full4の後だけ隔離5という順序を確認した。主agent自身の検査であり、独立レビューではない。
SOURCE commitとSHA/runtime/入力の固定は[seal.json](../artifacts/non_df_h4_d1_source_execution/2026-10-24/seal.json)へ保存する。

このSOURCE固定時点では実科学起動前であり、実行結果は新run01のstage/ledger/STOPを正本とする。
CPU/RAMの人工上限を追加せず、実資源を読み取り観測し、共有設定・他ユーザー・他worktreeを変更しない。
Stage Aの独立4確認とStage Cの独立5対象は並列化し、Stage Bは失敗時の即STOPのため逐次とする。
新engineと全taskをnative waitで回収し、旧監視証拠の代用にはしない。

専用branch `codex/non-df-h4-d1-source-execution-20261024`。
新grantは一度だけ消費し、結果・終了監査・commit/push後mandatory STOP。
今回の担当はCodex：承認された有限D1と保存まで。追加実行やD2/D3はユーザーの別判断を要する。
