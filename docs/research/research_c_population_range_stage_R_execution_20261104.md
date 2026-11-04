# 研究C：Population range Stage R・限定実行の部分結果

2026-11-04 JST。**固定SOURCE・契約を変更せずfresh stageR-run01を一度実行したが、10 call目の`event_catalog`で`SAVED_ANGLE_MODEL`となりSTOPした。10/256 unitsを使用、9 COMPLETE・1 FAIL、retry/resume0。** Population norm a・誤差range M・216event bound・13round profileは未取得。値を0やPASSで補完しない。

## SOURCE・契約・対象

- 科学実行SOURCE：`6c1fbe5399b2418e680acaa4f6e8aaa5e27d6dbd`。直接の技術起点：`bd560a45410409573ae849ca5aaf4050343910bf`。
- [固定契約](research_c_per_word_population_range_contract_unapproved_20261104.json)、scope `C_PR_R32_LITERAL_POPULATION_NORM_STAGE_R_V1`、最大256 logical units。
- 契約SHA256：`62e91a0598a0cd9357ff283771be6d343a69e9f9222df095f8dc3e2d306ee69c`。
- 技術SOURCE manifest SHA256：`863f26abe5d199d4bd9b0fcca9073f524361d68624e6a90fe5d8db21c7087f26`、154 file。
- [準備報告・267人工回帰PASS・旧4件除外](research_c_per_word_population_range_readiness_20261104.md)。チャットの実行許可をこの固定Stage Rへ適用し、歴史的unapproved metadataを保持した。承認証明・承認原文保存・新grantなし。
- [Preflight](../../artifacts/research_c_per_word_population_range/2026-11-04/stageR_execution_preflight.json)、[Calls](../../artifacts/research_c_per_word_population_range/2026-11-04/stageR-run01/science_calls.jsonl)、[partial failure](../../artifacts/research_c_per_word_population_range/2026-11-04/stageR-run01/partial_failure.json)、[STOP](../../artifacts/research_c_per_word_population_range/2026-11-04/stageR-run01/STOP.json)、[保存照合](../../artifacts/research_c_per_word_population_range/2026-11-04/stageR_execution_verification.json)。

Linear H4・隣接1.00 Å・STO-3G、Nα=Nβ2、sector36/system256/controlled512、number CDF8、元frame順LD2、partial ordered symmetric S2/s1/r32/K0、h=δ=.1、m0..12。CA=.00159 Ha・DF reserve=CA/8、own-DF ground-state oracle・状態準備費用除外。Endpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。元g補正κ/scalar/frame順、literal U、P0 Qiskit1.3/opt1/seed17/rz,sx,x,cx/MPFR160 declared phaseは変更していない。今回P0回路自体は生成していない。

## 実行前gateと実行範囲

独立worktree、clean HEAD、対象remote `HIROMU1015/Partially-Randomized-Trotter`、SOURCE154 file、保存入力34件のhash、NumPy1.26.4/SciPy1.14.1/Qiskit1.3.0/python-flint0.8.0/FLINT3.3.1・FE_TONEAREST、fresh出力不在を確認した。準備時から実装は無変更で、科学実行中の修正・fallbackはない。

既存`c_per_word_range_execution`→`execute_range`を使用した。固定input/endpoint、既存general-K/native model型、full CAR、既存raw head producerを再利用した。ここでの`native_model`はデータ型・係数・基底計画の準備であり、QuantumCircuit build/compileではない。

| call区分 | attempt | COMPLETE | FAIL |
|---|---:|---:|---:|
| evidence/input/candidate読込 | 3 | 3 | 0 |
| candidate binding | 1 | 1 | 0 |
| modelデータ準備 | 1 | 1 | 0 |
| full256 CAR context | 1 | 1 | 0 |
| raw head G0/G1/G2 | 3 | 3 | 0 |
| event catalog metadata | 1 | 0 | 1 |
| 合計 | **10** | **9** | **1** |

以後のgrid support、保存sector headの照合、head norm3、frame Gram8、event norm216、population組立、13round、summaryは0 call。未使用246 unitsを新しい試行へ転用しない。

Runnerは`result.json`を生成せず、6件・39,148 bytesの部分原本（SOURCE/contract/Calls/partial_failure/STOP/resources）を保存した。新しいreference_heads.npzやpopulation.jsonも存在しない。完了したhead計算の中間配列はrunnerが保存する段階より前にSTOPしたため、これらを公開operator結果やhead identityの証拠とは扱わない。

## 直接原因：保存meanタグとIIDモデル辞書のschema混同

固定runnerの`execute_range.metadata`は、保存`ideal_event_mean.json`の`model`を契約の`identity.model`と直接比較していた。

| 比較対象 | 実際の固定値・型 |
|---|---|
| 保存`ideal_event_mean.json.model` | 文字列 **`ideal_saved_atom`** |
| 契約`identity.model` | `id=IID_UNIFORM_53_BIT_WORDS_COUNTERFACTUAL`等を持つ**辞書** |

Stage A producerの`c_iid_grid_bridge.execute`は、ideal側を`ideal_saved_atom`、grid側を`MODEL['id']`という文字列として明示的に保存している。前者はevent平均の種類、契約辞書は独立性・mapping等を含む全体確率モデルであり、同じschemaではない。両者はこの比較では等しくならない。

したがって今回の直接原因は**私が準備したadapterのschema照合誤り**である。保存入力hashの破損や、H4のnorm/rangeが科学条件を満たさないことが確認された結果ではない。正しい比較対象をSOURCE調査で確認できたのに、実行前に取り違えを検出できなかった。

人工一体テストも`mean.model=MODEL`という辞書をfixtureへ入れていたため、この不一致を再現していなかった。267人工回帰PASSと実H4の科学資格を同一視しない。今回の[原因確認metadata](../../artifacts/research_c_per_word_population_range/2026-11-04/stageR_execution_diagnosis.json)はJSON型/タグとproducer/consumer SOURCEの静的照合に限定し、保存残差・分布・angle・operatorの新しい数値再解析を行っていない。

将来の最小修正は、angleの出典種別タグと全体IIDモデルを別項目として照合し、固定Stage Aの実schemaを使う人工fixtureを追加すること。Wrong tag/wrong IID modelをそれぞれ拒否し、文字列を辞書へ勝手に変換するfallbackは使わない。科学SOURCE・契約を別版へ固定してから、必要な新規実行範囲を明示する必要がある。**この報告では修正を適用せず、旧SOURCEを保全する。**

## 資源・数値資格・保全

起動時CPU affinity32、空きRAM約52.34 GiB、CPU/memory pressure avg10=0。独立head3件に既存resource_planが3 workerを選択し、BLAS1/workerでoversubscriptionを避けた。Arbの小さなscalar normへは未到達。GPU非対応の既存CAR/Arb CPU経路を使用し、共有venv/OS/cache/scheduler/他job/lockの変更はない。CPU/RAM/worker/wallの人工cap追加なし。

保存runner resource値はwall1.496519472450018秒、peak RSS497332 KiB。Process CPU時間はこのrunnerで記録していない。速さやspeedup、期待する次試行時間は未測定として扱う。

得られなかったものはa/M、m12 growth、B2/Aとのhead identity、per-word C/R/D誤差、confidence/期待bias、RPE/CA/費用。対応するunknown/nullとStage_C_ready=falseを維持する。新fit/optimizer・rank探索・PF/RTE誤差・ground solve・native build/compile・sampling・CI・shots/energy・Stage P/C/Aはすべて0。

固定SOURCE・契約・入力hashと、旧4word86-call/Stage A38-call/B2 27-call/前段技術STOPを保全する。今回の科学STOPも変更せず、残callの流用・retry/resume・別namespaceの自動試行を行わない。公開後にremote SHA/tree、主要本文、call journal/STOPと6件のrun原本を照合する。

**本Stage Rは部分結果を公開してmandatory STOP。** 次担当はCodex：別の技術作業としてschema照合の最小修正・実schemaを反映した人工回帰・新SOURCE/実行契約を準備する。ユーザー／研究Cはその固定版の再実行範囲を判断する。今回のretry/resume0契約を黙って変更しない。
