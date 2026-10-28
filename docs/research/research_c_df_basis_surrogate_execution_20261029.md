# 研究C：Basis-aware saved64再解析・basis-key照合で技術STOP

2026-10-29 JST。**3/最大148 logical units attempts＝2 COMPLETE＋1 FAIL、retry/resume0。** M4_Proposedの`static_features`入口で`EXACT_SAVED_BASIS_KEY`が発生し、固定契約に従って依存処理を停止した。Trajectory特徴量・raw QPY census・相関/モデル比較は全て未実施。複合モデルCの性能は未判定である。

## 固定対象・SOURCE・契約

対象は保存直鎖H4・隣接1.00Å・STO-3G、通常number CDF8/LD2、partial ordered symmetric S2 δ=h=.1/s1/r128/K0、PR-aware Proposedとtail-P4の固定32 trajectoryずつ。Native政策は共通P0/Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1/initially-zero=false/MPFR160、relative head・exact-key tail共有・cross-head/tail一般relativeなし。元κ・scalar・frame順・各保存J/U/lawを変更しない。新DF fit/sampling/build/compileは範囲外である。

- 起点／科学SOURCE commit：`b1985ff010c1e3cdac12a04fed05b510c6046bdf`。実行開始後無変更。
- 結果branch：`codex/research-c-basis-native-saved64-execution-20261029`。独立worktree/fresh run01で一度実行。
- [固定契約](research_c_df_basis_surrogate_contract_unapproved_20261029.json) SHA256：`345f81f7784335af87cbe0eff144997ffdad4aa452d89d7018f3f9027c03fa43`。
- [SOURCE99](../../artifacts/research_c_df_basis_surrogate/2026-10-29/technical/SOURCE.json) SHA256：`7fd98dee2b241bbd991c58fa73c10cb42a12f3b477531ed01b3ab42d3b16fd12`。
- [設計・再利用・人工43 PASS](research_c_df_basis_native_cost_surrogate_design_20261029.md)、[実行前preflight](../../artifacts/research_c_df_basis_surrogate/2026-10-29/execution_preflight.json)。

今回のユーザーの明示的実行許可により開始した。契約の歴史的`UNAPPROVED_UNEXECUTED` metadataは変更していない。承認証明・承認原文保存・新grantは作成していない。

## Preflightと実行結果

SOURCE99件、保存input267件、旧保護1458件のhash・依存package、人工43 PASSログ、fresh output不在を確認した。人工suite自体は既存固定結果の照合で、再実行していない。

| 登録call | 今回の実績 |
|---|---|
| saved_bundle/M4_Proposed | COMPLETE：保存U/tail/SOURCE/STOP読込 |
| identity_gate/M4_Proposed | COMPLETE：元実行SOURCEと332-call STOPを確認 |
| static_features/M4_Proposed | **FAIL：EXACT_SAVED_BASIS_KEY** |
| 他cohort、trajectory64、QPY64、models12、最終要約 | NOT_RUN |

Static feature評価は開始したが、既存catalogのbasis keyとの照合で停止した。完了したstatic feature recordはなく、q/run/overlap等の新しい科学結果を報告しない。新たなモデル間MAE/RMSE、相関係数、予測性能・採否は全て未取得/nullである。失敗をC不適格やbasis-aware研究全体の否定とは扱わない。

[Failure/traceback](../../artifacts/research_c_df_basis_surrogate/2026-10-29/run01/failure.json)、[call journal](../../artifacts/research_c_df_basis_surrogate/2026-10-29/run01/science_calls.jsonl)、[STOP](../../artifacts/research_c_df_basis_surrogate/2026-10-29/run01/STOP.json)を保存した。残call枠を別runや再試行へ転用していない。

## SOURCEの静的確認と人工入力だけの原因切り分け

生産側の既存経路は`c1_general_k_adapter._matrix`で各要素をcomplex化し、`_spin_lift`でoff-spin要素へ明示的に`0j`を入れ、そのtupleを`split_native_model`で`sha256(repr(rotation))`へ変換する。

新geometry helperは`_rotation(np.kron(mat,np.eye(2)))`を使っていた。数学的なspin liftは同じでも、負の実成分と0の積は符号付きゼロになり、`repr`が既存tupleと一致しない場合がある。新helperはここで既存のlift関数を再利用すべきだった。人工テストのkey一致ケースが主に単位行列だったため、この差を捕捉できていなかった。

STOP後は**保存H4を追加decode・再評価せず**、人工2×2 `diag(-1,1)`だけで両経路を確認した。

| 人工確認 | 結果 |
|---|---|
| 数値配列の要素一致 | true |
| Python tupleの値としての一致 | true |
| `repr`一致 | false |
| `repr`からのSHA256一致 | false |

[人工key診断](../../artifacts/research_c_df_basis_surrogate/2026-10-29/artificial_key_diagnosis.json)。これは技術的な原因候補の再現であり、保存実Uのどのentryが原因かの追加数値診断ではない。保存JU/catalogの破損・物理的不一致を立証したものでもない。

次の修正は既存`_matrix`/`_spin_lift`と同じ宣言tuple構成を使い、負成分・符号付きゼロ・複数frameの人工回帰を加えるのが最小案である。旧basis keyや旧rawを変えたり、許容差によるkey照合へ緩和したり、gaugeを変更して通す案ではない。**今回の固定SOURCE・契約には修正を加えず、retry/resume/fresh再実行もしていない。**

## 資源・保全・次の作業

32 affinity CPU、available RAM約52GiB、低負荷を実行前に確認した。小さい保存I/O・4次元幾何の契約どおり逐次schema gateで起動し、CPU/RAM/worker/wallの人工capは追加していない。Task-local venv/cache、依存read-only、共有設定/他jobの変更0。

[GNU time](../../artifacts/research_c_df_basis_surrogate/2026-10-29/process_time.txt)：wall0.20秒、user0.70秒＋system1.41秒、最大RSS39,488KiB、swap0、exit1。これはSTOPまでの時間で、148-call完了時間ではない。

[Postflight](../../artifacts/research_c_df_basis_surrogate/2026-10-29/execution_postflight.json)でSOURCE99/input267/旧保護1458件の不変、契約コピー・実行SOURCE、3 attempts/2 COMPLETE/1 FAIL、retry/resume0を照合した。新rawは7件/100,574 bytes。新DF fit・optimizer・sampling・build/compile・QPY census・trajectory特徴量・モデル比較はいずれも0。

**Codexの次の提案作業**：旧run01/SOURCE/STOPを保全した別revisionで、key再構成を既存liftへ接続し、人工回帰と新SOURCE/契約を固定する。**ユーザー／研究C**：その具体的修正後SOURCEとfresh outputを対象に、別の限定科学再実行を許可するか判断する。今回の契約はretry/resume0のため、このSTOPを無断で迂回しない。

性能比較、新しいDF、η/scale/目的関数採択、重要GPT詳細科学レビューへは進まない。部分証拠・報告をcommit/pushし、remote SHA/tree・SOURCE・旧新原本の一致確認後、**mandatory STOP**。
