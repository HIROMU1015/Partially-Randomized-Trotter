# 研究C：schema修正後のStage R・256-call限定実行結果

2026-11-04 JST。**固定SOURCEを変更せず、fresh Stage Rを一度実行し256/256 calls COMPLETE、correctness FAIL・retry・resumeは0。** `SAVED_ANGLE_MODEL`は再発せず、全216 eventのnorm包囲と13round growthを取得した。Population norm上界はa=1+約1.76183e-16、条件付き誤差rangeはM=2+約1.76183e-16。保存B2のhead bitsとは3件とも不一致で、別のhead bridgeが必要。全law synthesis・期待bias校正・actual CAは未資格のまま。

## 固定SOURCE・契約・対象

- 科学SOURCE：`60cdba8fe475acb728e86e7733694ac8211d2ec7`、tree `24984140a94190b1e3b2b0eff4c686ea334de848`。
- [固定契約](research_c_population_range_schema_contract_unapproved_20261104.json)：`C_PR_R32_LITERAL_POPULATION_NORM_STAGE_R_SCHEMA_V2`、最大256 logical units。
- 契約SHA256：`cf1b4dbecda546914e63ba9032eb42bcdce8bfe8043a85e2138a4cdd37fffada`。
- [SOURCE manifest](../../artifacts/research_c_population_range_schema/2026-11-04/technical/SOURCE.json) SHA256：`ac0159e3137679f5c2a693033f71f4bd2b260af71ea6ed351673bb1f4c0e2b59`、157 file。保存入力35件。
- [準備報告・人工48/関連回帰313 PASS](research_c_population_range_schema_preparation_20261104.md)。ユーザーの今回の実行許可を固定V2契約へ適用した。歴史的なunapprovedファイル名・metadataは変更せず、承認証明・承認原文保存・新grantを作成しない。

対象は保存直鎖H4・隣接1.00 Å・STO-3G、number CDF8、Nα=Nβ=2、sector36/system256/controlled512、元順LD2、partial ordered symmetric S2/s1/r32/K0、h=δ=.1、m0..12。CA=.00159 Ha、DF reserve=CA/8、own-DF ground oracle・状態準備費用除外。Endpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。元g補正κ、scalar、frame順、literal Uとfull-CAR head定義を保持する。P0 Qiskit1.3/opt1/seed17/rz,sx,x,cx/MPFR160 phaseも変更していない。今回、新しい回路build/compileは行っていない。

独立branch `codex/research-c-stage-R-schema-execution-20261104`、SOURCEから作った独立worktree、fresh出力 `artifacts/research_c_population_range_schema/2026-11-04/stageR-run01`を使用した。旧10/256-callの[部分失敗](research_c_population_range_stage_R_execution_20261104.md)、旧SOURCE/raw/契約/STOP、残246 unitsは保全し、再利用していない。

## 実行前・call実績

[実行前照合](../../artifacts/research_c_population_range_schema/2026-11-04/execution_preflight.json)で公開SOURCE SHA/tree、clean HEAD、契約・SOURCE157 file・入力35件のhash、実schema、fresh不在を確認した。Python3.11.1、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、python-flint0.8.0/FLINT3.3.1、FE_TONEARESTを確認した。準備済みrunner・source・数値条件に変更はない。

[Calls](../../artifacts/research_c_population_range_schema/2026-11-04/stageR-run01/science_calls.jsonl)と[STOP](../../artifacts/research_c_population_range_schema/2026-11-04/stageR-run01/STOP.json)は256 attempt/256 COMPLETE、FAIL0、retry/resume0を記録する。

| 区分 | COMPLETE |
|---|---:|
| evidence/input/candidate読込、binding、modelデータ、CAR context | 6 |
| raw head G0/G1/G2 | 3 |
| event catalog、grid table、保存head読込、head binding | 4 |
| exact support | 1 |
| head norm | 3 |
| frame Gram/norm | 8 |
| event norm | 216 |
| population組立 | 1 |
| 13round growth | 13 |
| summary | 1 |
| **合計** | **256** |

`native_model`は既存データ型・general-K分割の準備で、QuantumCircuit buildではない。理想eventタグ文字列、gridモデルID文字列、全IID設定辞書は新schemaで別々に照合され、旧誤比較へfallbackしていない。Saved angle・signed atom・support・frame・係数を保持した。

## Population norm・range

[Population原本](../../artifacts/research_c_population_range_schema/2026-11-04/stageR-run01/population.json)、[全結果](../../artifacts/research_c_population_range_schema/2026-11-04/stageR-run01/result.json)のstatusは`VALIDATED_LITERAL_POPULATION_NORM`／`LITERAL_POPULATION_NORM_STAGE_R_COMPLETE`。正q-supportは216、exact count=0による除外は0。Sample最大や確率閾値による除外を使用していない。

Literal Dのnorm上界aとrange上界Mの保存値は次のexact rationalである。

```text
a = 1569275433846670467439326996090883863351856440850483005231
    /1569275433846670190958947355801916604025588861116008628224
M = 3138550867693340658398274351892800467377445301966491633455
    /1569275433846670190958947355801916604025588861116008628224
```

表示用には **a−1≈1.761834625567096e-16、M−2≈1.761834625567096e-16**。保存endpointは正確にa>1、M>2であり、binary64表示で1や2へ丸めて採用しない。3 headはexact Hermitianとは判定されず、literal Uの非直交性も保持した。Arb192のoutward norm評価は、元のexact dyadic head/frame入力を対象とする。

Dについては全正q length32 wordのnorm上界を取得した。Cについてのnorm1は受理された理想P0 gate列に対する性質である。全q-wordでcompilerが成功するというtotal-domain前提は未資格なので、`range_qualification=conditional_on_total_compiler_domain`を維持する。Mはこの前提下でd=||C−D||を包む粗いpopulation rangeであり、実際のsynthesis誤差を測定した値ではない。

### 13round growth

以下は保存`D_product_norm.upper_exact`からの表示のみで、新しいnorm評価ではない。各行はD積のnorm上界1+ε_mを示す。Exact/dyadic endpointは原本JSONへ保存し、全行でclipped=false。

| m | coherent occurrences | ε_m（表示値） |
|---:|---:|---:|
| 0 | 1 | 1.761834625567096e-16 |
| 1 | 2 | 3.523669251134192e-16 |
| 2 | 4 | 7.047338502268384e-16 |
| 3 | 8 | 1.409467700453677e-15 |
| 4 | 16 | 2.818935400907357e-15 |
| 5 | 32 | 5.637870801814721e-15 |
| 6 | 64 | 1.127574160362947e-14 |
| 7 | 128 | 2.255148320725908e-14 |
| 8 | 256 | 4.510296641451866e-14 |
| 9 | 512 | 9.020593282903936e-14 |
| 10 | 1024 | 1.804118656580868e-13 |
| 11 | 2048 | 3.608237313162062e-13 |
| 12 | 4096 | 7.216474626325427e-13 |

この固定targetではm12までのnorm成長上界が小さいことを確認できた。Kernel誤差、期待bias、校正N/δ、RPE十分条件やCA達成は今回算定していない。

## B2とのhead identity・残件

[Head binding](../../artifacts/research_c_population_range_schema/2026-11-04/stageR-run01/reference_head_binding.json)はG0/G1/G2の全件で`exact_head_bits_equal=false`、`B2_head_identity_qualified=false`、`additional_head_bridge_required=true`。今回のfull-CAR producerのhead bitsと、B2側で保存sectorから定義したhead bitsを同一targetとして移送できない。

これは登録済みのprovenance診断で、今回のliteral range算定を停止するcorrectness FAILではない。不一致の大きさ・原因・追加bridge norm上界は今回評価していない。差を丸め誤差として無視したり、片側のheadを置換したりしていない。[Full head原本](../../artifacts/research_c_population_range_schema/2026-11-04/stageR-run01/reference_heads.npz)を保存した。

`all_q_words_compile_success_qualified=false`、`calibration_ready=false`、`all_word_synthesis_error_bound=null`、`Stage_C_ready=false`、`actual_CA_achieved=null`を維持する。新fit/optimizer、PF/RTE誤差再計算、ground solve、回路build/compile、sampling、confidence/CI、shots/energy/総費用、Stage P/C/A、重要科学レビューはすべて0。

## 資源・保存・停止

実行直前のCPU affinity32、空きRAM約52 GiB、空きdisk約90 GiB、CPU/memory pressure avg10=0。既存resource_planは独立3 headに3 worker、BLAS1/workerを選択した。Scalar Arbは小さい既存経路をprecision lock付きで用いる。CPU/RAM/worker/wallに恣意的なcapを追加せず、共有venv/OS/cache/schedulerや他jobを変更していない。保存resource値はwall1.6288365283980966秒、peak RSS497488 KiB。Speedup比較は行っていない。

[実行照合](../../artifacts/research_c_population_range_schema/2026-11-04/execution_verification.json)はSOURCE157・入力35件、256 COMPLETE、原本9 file/3,641,866 bytes、13行、未資格nullを確認した。旧失敗run treeは`e267f48302eefda1cfa1e0e5e5b4085868fcca14`のまま。科学SOURCE/inputと旧失敗原本の重複排除198 fileは固定SOURCE時と同一である。

公開時はSOURCE・raw・journal・STOPを保存し、remote SHA/tree、主要本文とbinary blobを照合する。**公開照合後mandatory STOP。** 次の担当はユーザー／研究C：追加head bridgeとper-word誤差・compiler domain前提のうち、次に固定する検証範囲を判断する。Codexは新しい範囲の明示許可を受けてから作業し、今回の残枠や別stageへ自動進行しない。
