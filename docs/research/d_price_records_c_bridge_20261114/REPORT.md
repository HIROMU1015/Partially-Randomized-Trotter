# D：CのPauli physical-ID bridgeへの接続と価格record分類の修正

2026-11-14 JST、Codex D。**技術修正、人工テスト、metadata照合まで行った。** 実H4のsampling・回路build・compile・価格取得・fit・N/G・契約変更は行っていない。

固定SOURCE（修正後のcallerとtest）：**`62ae1bd56718287de5d958cf0ed6411f5b7395c9`**。その直前の `1e5c2c15` は、8b28dce0とbyte一致の取込みcommitである。[HANDOFF](HANDOFF.md)／[次の有限compile](NEXT_EXECUTION.md)／[test結果](TESTS.txt)

## branchとSOURCEの扱い

- **新branch**：`codex/d-price-records-c-bridge-20261114`。基点はCの `c628e9e8`（Pauli bridge planner・guard・paper会計・model検査）。
  - 理由：修正したPauli callerは、実行時にCの正式planner（`I.plan(..., implementation_sources=...)`）・`price_context`・`pauli_requests` を **tree内moduleとして** 呼ぶ必要がある。Cのbranchへはcommitせず、mergeもしていない。
- **D固有のfileの持ち込み**：Dの4本（Pauli接続、T3、Full PF、Pauli pilot）とそのtestを、まず `8b28dce0`（＝`f2a2c3af`）と**byte一致のまま**取り込んだ（commit `1e5c2c15`、挙動変更なし）。今回の修正はその上のdiffとして確認できる。
- **旧SOURCEの保全**：`04bead11`（S-1/P-1/T3）と `8b28dce0`（Pauli pilot／Full PF）は元のbranch `codex/d-kappa-s1-price-builders-20261114` に残しており、上書きしていない。S-1・P-1・κ helperはこのbranchへ持ち込んでいない。旧SOURCEのS-1・P-1が実行時に使うのは旧treeである。
- **新旧の区別**：新しいrecordには `record_schema='D_PRICE_RECORD_B_KINDS_V2'` と実行commitを付け、旧版と区別する。byte検査（自分自身・接続module・native・providerがHEADと固定blobに一致すること）は無効化していない。

## 作業A：Pauli physical candidateをCのcanonical bridgeに接続

`scripts/run_pauli_ld1_m0_pilot.py`、`src/trottertracks/pauli_tail_price_connection.py`

- `canonical_request()`：Cの `RealPauliBridge` と同じ入力でCの `I.plan` を呼ぶ。入力は次の3つ。
  - signal SOURCE＝A handoff（`714247619`）
  - target＝B LD1（`91349e70`）
  - round別numerical SOURCE＝A carrier recordの `source` field（値は読まない）
  
  結果は `provider='A_PR_PAULI'`。
- price contextとphysical candidateは **`I.price_context(req, 0)` を正本**とする。sidecarとrecordの土台には、Cの `rpe_d_price_connection.pauli_requests(root, req)` が返すrowを使う。このrowはDの固定 `round_request` との照合と、route/lowering定数の照合を経ている。Dで新しいID生成器は作っていない。
- `legacy_physical_candidate` はこの版のmoduleから削除した。旧形式IDはtestの拒否caseでだけ組み立てる。
- preflightで確認した値は次のとおり。
  - configuration `d905d219…`、route `PR_PAULI_TAIL`
  - canonical physical ID `RMSE:62a9a6b68c2c88951f1055d1da7f9b6a89da94a91092c9488cf5fce503288e54`
  - T=0.1、q=1、R_outer=3、K=2、prefix=1、event_count=3
  - 辞書SHA `dbdde5a1…`。splitとlawの辞書は同一
- **変えていないもの**：
  - 辞書184項（順序・係数・hash）、D2 lowering（`PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF`、9c4006bf）
  - sampler（PCG64・K2のnormal law）、scalar c＋s_Rの一回適用
  - seed：key＝canonical candidate ID（bridgeのphysical IDと同じ文字列）。値は `14788159701244282917` のままで、preflightで固定値と照合する
  - cohort `D_PAULI_LD1_M0_WIRING_V1`、trajectory定義、子processの回路生成・oracle・compile経路（`pilot_core` と `child` は不変）

## 作業B：価格recordの分類をB/Cの正式schemaへ

| 方式 | classification | expectation | price_status | 旧名の扱い |
|---|---|---|---|---|
| Full PF（`run_full_pf_shortlist_prices.py`） | `DIRECT_COMPILED_DETERMINISTIC` | `DETERMINISTIC` | `DIRECT_DETERMINISTIC_AVAILABLE` | `diagnostic.record_role='EXACT_DETERMINISTIC_CIRCUIT_PRICE'` |
| Truncated p7（`run_truncated_p7_prices.py`） | 同上 | 同上 | 同上 | 同上 |
| Pauli LD1/m0 | `DIRECT_COMPILED_FINITE_SAMPLE` | `FINITE_COHORT_ONLY` | `DIRECT_FINITE_COHORT` | `diagnostic.pilot_role='DIRECT_SINGLE_PATH_WIRING'`、`law_expectation=false`、`qualified_expectation_promotion='FORBIDDEN'` |

- Pauliの `n_paths=1`。`law_context` は既存の `B.p.measurement(row, axis, saved_sampler=True)` の形を使う（`price_law=PCG64…`、`ideal_sampler_distance_qualified=false`）。sidecarの `measurement` は従来どおり既定形で、C guardの要求に合わせた。
- Full PFのfit入力は、tree内のC `price_model_validation.record` をそのまま呼ぶ（以前はblobから読んでいた）。kindは `DIRECT_COMPILED_DETERMINISTIC`。
- 回路、cell、gate、件数、出力先は変えていない。**価格値は生成していない**（testの値は架空で、そう明記している）。

## C guard・paper会計・modelとの接続結果（人工価格）

- **Pauli**
  - C guardは、`PR_PAULI_TAIL` policyのもとで、canonical候補のsidecar（X/Y）を受理した。
  - 拒否されたもの：旧形式ID、辞書SHA、law辞書、native SOURCE（commitなし）、wrapper SOURCE、scalar（1e−12ずれ）、route欠落、ZZ lowering、ZZ emitter、scalar quantity。DF用 `PR` policyやZZ policyでの評価も拒否された。
  - `paper_cost` ではm0のcellがtier `FINITE_COHORT_SCENARIO` になり、他roundは欠測のためG_paper=None（0にはしない）。
  - `QUALIFIED_EXPECTATION` を手で付けても、samplerが未資格なので昇格しない。決定論の分類は拒否される。旧名は `Unknown compiled price kind` で拒否される。
- **Full PF・Truncated p7**
  - `FULL_PF`／`TRUNCATED_PF` と `NOT_APPLICABLE_NO_RTE` のsidecarを、同じpolicy（共通のnative policy、scalar quantityは方式別）で受理した。
  - `paper_cost` ではtier `DETERMINISTIC_DIRECT`。旧名は拒否、`QUALIFIED_EXPECTATION` を付けた決定論価格も拒否された。
  - scalar quantityを取り違えると拒否された。
- **model**：Cの `record`/`fit` の形式が整合した（架空の値でinterfaceだけを確認）。

## テスト

[TESTS.txt](TESTS.txt)。

- Dのtest 37件はすべて**PASS**。
  - Pauli pilot（parent 8、circuit 5）、Full PF（parent 6、circuit 2）、Truncated（parent 3、circuit 5）、Pauli接続8。
  - circuit testは、P0・index0・full-2^n oracle・scalar・X/Y・controlled作用・bit順・辞書不一致の拒否を含み、回帰がないことを確認した。
- Cのtest 48件（`check_paper_g_price_routes` 34、`check_price_implementation_guard` 14）も、このtreeで再実行してPASSした。

作者自身の技術確認であり、独立レビューではない。

## 残るUNKNOWN

- **Full-H P-1**（旧SOURCE `04bead11`）のrecord名も、B kindではない（`DIRECT_SINGLE_PATH_WIRING`／`NOT_A_LAW_MEAN`）。今回は修正範囲外のため変更していない。G会計に使う前に、Pauliと同じ `FINITE_SAMPLE`／`FINITE_COHORT_ONLY` 版が必要になる（HANDOFF参照）。
- Bの新decoder canonical bridge（C PAULI_BRIDGE §4）が採用されると、canonical IDが変わる。その場合、既存cohortのseed keyは旧IDのまま残す方針である。新IDでの再発行（a）か、guardの写像（b）かは、ユーザーとBが決める。
- 実H4のcontext読込、回路規模、compile時間、保存M5 m4の受入、q-affineの外挿、law/rare、N/G。

commit/push・remote照合後 **mandatory STOP**。
