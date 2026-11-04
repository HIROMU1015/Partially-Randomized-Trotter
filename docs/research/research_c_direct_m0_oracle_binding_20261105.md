# 研究C：保存own-DF oracleのproducer対応と固定15-call案

2026-11-05 JST。**保存numerical oracleの入力対応をsource/JSONだけで確定した。Arb192の固定m0・15-call検証案を科学SOURCEへ固定する。今回の実H4数値評価は0。**

Branch：`codex/research-c-direct-m0-oracle-binding-20261105`。
独立worktree：`/tmp/prt-c-direct-m0-oracle-binding-20261105`。
起点 `e409111ccd1427c343c2c44a4e87a3230f70a824`。
[前段m0準備](research_c_direct_m0_runner_preparation_20261105.md)のmodule・契約・SOURCE・STOPはそのまま保持した。
正式共通指示2026-10-25の§8.1・§8.4・§10を適用。重要科学レビューや実H4計算を開始する許可は今回与えていない。

## 確認できた対応と保証の範囲

前段では保存Hとoracleの対応を未確定として外部の確認に戻した。
今回、既存のproducer・旧固定契約・SOURCEを追い、**既存の保存numerical own-DF referenceの範囲では、同じHが保存groundの入力だった**ことを確認した。
Hのexact symmetryを実行前に宣言させる必要はなく、数値評価を許可された実行の`target_identity`内で検査する形へ具体化した。

| 対応 | 固定された事実 |
|---|---|
| 旧契約のoracle | `C_NEAR_PRACTICAL_TWO_ENDPOINT_COMMON_R32_DOWNSTREAM_V1`のpolicyは`own_DF_ground_state_oracle` |
| Candidate | PR endpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a` |
| H入力 | 旧契約`endpoints.PR.reference`は今回の`roles.sector`と同じ`physical/PR_update016.npz` |
| Ground出力 | 旧fresh出力の`PR/A/ground_candidate.npz`と今回の`roles.ground`が一致 |
| Producer | `prepare_arm → ground(sector['H']) → loader.ground → MolecularBackend._ground → np.linalg.eigh(matrix)` |
| SOURCE | 上記3 Python sourceは旧SOURCE manifestのhashと一致。新しいproducerを仮定していない |

旧契約・SOURCEは`artifacts/research_c_near_practical_r32/2026-11-03/run01/`。
確認したsourceは`c_near_practical_r32.py`、`c_pr_aware_df_stage_b_runner.py`、`c1_stage3_molecular.py`。
機械可読の対応は[oracle_lineage.json](../../artifacts/research_c_direct_m0_oracle_binding/2026-11-05/technical/oracle_lineage.json)。
これは入力provenanceであり、exact Hermiticity、固有値分離、状態距離を新たに評価した証拠ではない。

旧solverはtolerance付きHermitian検査後に`eigh`を使う。今回の将来検証は**保存Hがexact real symmetricであることを要求**する。
PASSならsolverの片三角の解釈差はなく、同じ保存Hの一意なgroundへ状態距離を包囲できる。
FAILならSTOPする。対称化、三角completion、再fit、別CAR target、polar化、状態の正規化による置換は行わない。
保存Hの実配列を今回decodeしていないため、PASSするかは未確認である。

保証する対象はこの**既存の保存numerical own-DF reference**である。
理想積分からexact算術で再構成したDF Hamiltonianや、generator構築丸めを除いた別CAR targetとのground同一性は主張しない。
その橋渡しが必要な科学的主張へは拡張しない。今回の契約は固定m0のnative/literal信号診断で、全law、RPE accuracy、actual CAの資格を返さない。

## 固定した科学検証候補

対象条件は直鎖H4・隣接1.00 Å/STO-3G、number CDF8/元順LD2、partial ordered S2/s1/r32/K0、h=δ=.1。
Sector36/system256/controlled512、own-DF reference、状態準備費用除外、P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx、既存Gaussian/relative fusion、MPFR160 declared phaseを維持する。
対象はm0、一つの保存word、F6:0/F7:6,7交互16回・32 event。

Call内訳は前段の15 grouped unitsを維持する。

1. metadata load
2. arrays load
3. target identity（exact real symmetryを含む）
4. compiled head/X/Y shell decode
5. 数値固有値分離候補を1回生成
6. exact rational LDL inertia・正規化残差・state-distance certificate
7. sector embedding
8. 元vector centre＋距離のbinding
9. 保存literal blockの準備
10. compiled/ideal head作用
11. 同じwordのnative/literal forward作用
12. 同じforwardのactual inverse state witness
13. X readout
14. Y readout
15. summary

Arb192、外向き丸め、complex rectangleのexact rational endpoint、非unitary literal norm成長を保持する。
初期状態を黙って正規化せず、元centreのnorm差とground方向誤差を包囲する。
Inverseは公開済みexact common-unitary削減を使う指定状態の診断で、Full512 operator保証ではない。
Head/X/Y shellは直接作用させ、旧shell誤差・分子Head/B2/categorical項を重複加算しない。

Raw/P0再build/compile、fresh sampling、reference再生成、旧Stage P/7-call再実行、PF/RTE、B2再解析、shots/energy、期待bias校正、Stage C、actual CAは0。
新ground vectorのeigensolveは0。Call5の固有値**候補**生成1件は新しい科学的数値評価であり、実行許可の範囲に明示している。

RuntimeでSOURCE/input/word/phase/register/inverseが不一致、Hが非real/非exact-symmetric、LDL zero pivot、ground非分離、state bound不十分、非有限、例外/割込みなら部分結果を保存してSTOP。
最大15 units、retry/resume=0。失敗後に別shift、別candidate、別namespaceを自動実行しない。

## 実装、SOURCE、入力

新adapter：[c_direct_m0_bound.py](../../src/trottertracks/c_direct_m0_bound.py)。
既存m0 evaluator、oracle primitive、Calls、Arb伝播を呼び、旧moduleを変更していない。
新契約：[research_c_direct_m0_bound_contract_20261105.json](research_c_direct_m0_bound_contract_20261105.json)。
旧契約を上書きせず、今回の固定検証候補を別に記載した。旧grant・残call・STOPは転用しない。

Scientific SOURCE commitは `098e1bac4a9aa0bd6d1c3c61d19a38a1ac47bc6d`。このcommit内のmanifestと実worktreeのbytesもpreflightで照合する。Numerical sourceのmanifestは
[technical/SOURCE.json](../../artifacts/research_c_direct_m0_oracle_binding/2026-11-05/technical/SOURCE.json)。
25 source fileを識別し、その他の継承依存は起点commitへ固定する。
前段の13実入力roleに、旧producer契約/SOURCEと親m0準備契約の3 provenance fileを追加し、16 fileのbytesをpinした。
SOURCEやhashは科学的再現性のためのもので、ユーザー承認の証明ではない。

入口は`PYTHONPATH=src <固定venv>/bin/python -B scripts/run_research_c_direct_m0_bound.py`。
**Defaultはbytes/JSONだけのpreflight**。Scientific SOURCEを固定し、最終preflightは`FIXED_SCOPE_READY_AWAITING_EXPLICIT_EXECUTION_PERMISSION`。`execution_ready=true`は条件が具体化済みという意味で、実行許可や科学的GOではない。
実行は別の明示許可後に同じ入口の`--execute-science`を一度だけ使う。今回はこのflagを使っていない。
Fresh科学出力は`artifacts/research_c_direct_m0_oracle_binding/2026-11-05/science-run01`で、今回作成していない。

## 技術検証、資源と保全

新テストはproducer/candidate/H/ground pathの対応、SOURCE変更、固定word・scope・精度・calls、未評価symmetryの偽PASS、SOURCE commitの不一致、無許可entry、passive import、partial/割込み/STOPを検査する。
新規25件＋前段m0の26件＋Directの46件、合計97 PASS。初回95 PASSログも保存した。件数とログは[technical](../../artifacts/research_c_direct_m0_oracle_binding/2026-11-05/technical)。
人工テストでは架空の値・callbackを使い、保存実H4のNPZ/QPY/Arb評価はしない。
ローカル人工検証で、CI・外部独立再現や実H4の数値結果ではない。

将来runはcompiled11386 gateの1状態をFLINT局所blockで伝播し、inverseの共通unitary削減を再利用する。
順序依存のある単一m0 stateへ恣意的なworker上限を設けず、CPU affinity/quota・空きRAMと非干渉を実行直前に確認する。
CPU/RAM/wall capは新設せず、共有環境・他job/cache/lockを変更しない。実H4の新しい速度・RAM実績は主張しない。

## 次の実行とmandatory STOP

今回完了するのはproducer対応、scientific SOURCE/固定契約の準備、人工検証、公開照合まで。
実H4科学実行・保存科学数値再評価・重要科学レビューは0。
次担当は**ユーザー**：この固定15-call検証の科学実行を明示的に許可するか判断する。
許可された場合に限り、**Codex**が固定SOURCEと入力を再照合し、fresh出力へ一度実行し、結果/partial/STOPを公開する。

旧原本と停止条件を保持し、今回は公開照合後**MANDATORY STOP**。重要科学レビュー、全law、校正、Stage Cへ自動進行しない。
