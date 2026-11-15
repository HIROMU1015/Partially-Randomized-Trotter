# Bへ：決定論PFの条件付き採用価格60軸

**結果固定commit：`e4458cd7548977e6e11a5a7b0c8dca0d049535d4`**。

正本path：`docs/research/pf_closure_prices_20261115/run01/PRICES.json`、selector **`rows`**（60行）。CSVは同じpathの`PRICES.csv`。[REPORT](REPORT.md)／[実行直前照合と96直接軸索引](run01/CHECKS.json)／[TESTS](TESTS.txt)。科学SOURCEは`0ecd30f3b955d90d9898d0b09ff8bc6b15d93ce6`。

## 受取方法

既存の固定Git JSON reader（Cの`price_implementation_guard.evidence`/共通SavedArtifacts reader）から、公開固定commit/pathを読める。実際にこのrefの60行と整数値を読取確認済み。読取時のfit/predict/会計は0。

```python
ref = {
    'commit': 'e4458cd7548977e6e11a5a7b0c8dca0d049535d4',
    'path': 'docs/research/pf_closure_prices_20261115/run01/PRICES.json',
}
table = saved_reader.read(ref)
by_key = {
    (r['configuration_id'], r['physical_id'], r['m'], r['axis']): r
    for r in table['rows']
}
# Integer per-shot RZ: by_key[key]['compiled_RZ']
# The same value is also in mean.rz_count and exact_RZ (denominator=1).
```

結合キーは **`(configuration_id, physical_id, m, axis)`**。`canonical_id`は元alignmentの値で、total-decoder用IDへ価格SOURCEを付け替えない。Bがdecoder bridgeを使う場合は元物理候補への対応を保持して結合する。方式名や価格が同じだけで他候補へ移送しない。

各行はroute/formula/q/T/actual幅、元physical_candidate、input_binding/split/fragment順、scalar、signed_structure、compiler/native profileを保持。`frozen_model`、`closure_audit`、`terminal_diagnostic`、`recommendation_row`が固定出所への入口。元係数は行の`coefficients`にもcopyしているが、Bで再fit・再予測する必要はない。

## scope・件数

Full PF p10・B0I p7 × S2 16h/7・標準Yoshida4 16h/3・M5 16h × X/Y。m7/m8/m10/m11/m12、12 group×5 round＝**60/60確定、保留0**。全件exact非負整数RZ。

既存直接価格はm0〜m6及びm9の**96/156軸**。今回の60軸との交差0、和集合は156 unique axis。保存M5/m4の4軸は元EXPLORATION receipt（C受入）、m3の12軸も元C受入を維持し、較正/holdoutへの再編入なし。元直接96軸の固定locatorは`run01/CHECKS.json#preflight.direct_inventory`。価格の算出/再平均はその96軸について行っていない。

## 採用条件とrecordの意味

利用者が採択したC推奨：`c3f761f0af5b21207e64051c9c619858064450fe:docs/research/pf_closure_price_recommendation_20261115/`。

- 固定H4/1Å/STO-3G/tol-only DF10、元H/state/snapshot、Full/p7・公式・固定実幅・fragment/signed pieceを保持する。
- scalarは元定義を総Tで一回、symmetric-directional、測定付きfull controlled Hadamard wrapper、P=0。Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、coupling mapなし・元compiler経路に限定。
- 監査の障壁・反復構造・終端runが対象qまで持続するという閉包前提、installed Rustと参照sourceの対応、未調査の規模依存分岐に関する前提を明示する。監査の最終motif観測を全passの不変性証明と呼ばない。
- 用途は**同じ物理候補・固定shot配分の全13 round条件付き比較**。今回N・G・winnerはない。N十分性やdecoderの資格はBの既存契約を維持する。

行は`classification=MODEL_PREDICTION`、`evidence_kind=FIXED_COMPILER_CLOSURE_DERIVATION_CONDITIONAL`、`price_status=CONDITIONAL_CLOSURE_DERIVED_AVAILABLE`。`direct_observation=false`、`unobserved_circuit=true`、`conditional_use_adopted=true`を必ず残す。`adoption_policy=PF_FIXED_COMPILER_CLOSURE_CONDITIONAL_60_AXES_V1`。

元凍結モデルの`qualified_for_candidate=false`/長q資格falseは履歴として保全。今回の限定採用は別policyで表し、無条件`QUALIFIED_MODEL`や`DIRECT_COMPILED_DETERMINISTIC`へ書換えない。既存tierが`UNQUALIFIED_MODEL`を返すことと、利用者がこの有限scopeの条件付き利用を採択したことを区別する。Bの共通API・guardをCが変更/無効化したものではなく、Bはこの採択条件を明示する既存model/scenario受入経路で利用する。

未観測60軸に2%保証はない。誤差上界・CI・共有systematicはnullのまま。末尾runやcompiler/入力の前提が変わる場合は影響domainをUNKNOWNとして保留し、0補完・自動refit・別候補の単価移植をしない。m0〜m3の別幅価格をこの式で置換しない。

## 実行・検証・ロードマップ

既存exact `price_model_validation.predict`を60回だけ呼び、係数fitは0。テストは人工/metadata6＋既存exact回帰3＝9件PASS。SOURCE固定後のfresh run01がexit0、JSON/CSV一致、60行/12group/5roundと直接96との非重複を確認した。保存モデル・旧推奨・直接receipt・QPY監査結果/旧STOPは不変。

**ロードマップ差分：決定論PF6候補の価格入力は直接96軸＋条件付き導出60軸で全156軸に対応。Cの60価格待ちは解消。固定shot全13 round会計はBの後続範囲で未実施。** ロードマップ本体は統合担当が更新する。

公開・remote照合で**mandatory STOP**。追加compile、QPY再監査、再fit、N・総費用・winnerへ進まない。
