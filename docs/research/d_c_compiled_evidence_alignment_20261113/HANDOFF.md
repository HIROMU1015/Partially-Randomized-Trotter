# D → C/B：選択direct証拠の受渡し

[REPORT](REPORT.md)／[必要セル](NEEDED_CELLS.md)／[有限案・STOP](FINITE_SCIENCE_PLAN.md)。Cは `1914d0764115d82ec860e08810dafeb50f8d9a42` を読取参照した。C380→1914のD転送2 moduleはbyte不変。**実価格はまだ渡していない。**

## 既存APIの接続

選択metadataはDの修正済みselectorを使う。下はAPI説明であり、科学実行用callerではない。

```python
from trottertracks import ld12_price_requests as L

d = L.request_for(root, 'PR-LD2', selected_m, axes=('X', 'Y'),
                  execution_source=explicit_frozen_execution_source)
pc = d['physical_context']

# Cの既存API。totalを再計算せずquantity/provenanceも一緒に転送する。
pending = I.price_requests(root, C_request,
    input_source=pc['input_source'], execution_source=pc['execution_source'],
    axes=d['axes'], expected_scalar=pc['expected_scalar'])
```

C_requestは同じA固定configurationを使う。Cのprice_requestsは全scheduleのpendingを作り得るが、**それは実行cellの追加ではない**。選択したD request/round/axesだけを科学callerへ渡し、他roundはpendingのまま。C scalar bindingはDのrequest metadataをforwardするだけで、A tail-only値・診断の数値・LD2の規則から独自のscalarを作らない。

D runtimeではSOURCE付き保存H/stateを復元後、実prepareからc_H+s_Rを求め、expected totalと元のhex gateで照合する。`input_source.scalar_binding`の3 quantityがkeyへ含まれる。request/context ID生成式は元のまま、実行SOURCEが変われば再生成する。今回の [FINITE_CELLS.json](FINITE_CELLS.json) はtechnical tag付きなので、そのIDを実行receiptへコピーしない。旧748/752/240のID・結果を新requestへ付け替えない。

## 承認後に受け渡すrecord

Dは既存 `Adapter.acquire` の `CANDIDATE_PATH`（quantity=`PR_CANDIDATE_PAIRED_PATH_PRICE`）を維持する。source/input/execution、request/context、cohort/index/conditional、resolved context、event digest/seeds/law、軸別metrics/fingerprintを保持する。S=`FULL_WRAPPER_PATH_STRUCTURAL_RZ_COUNT`、C=`FULL_WRAPPER_PATH_COMPILED_RZ_COUNT`、同path/axisのDを区別する。

Cは既存 `rpe_d_price_connection.bind_results(..., paths=[...], prices=(), bindings=())`でpath bindingだけを確認できる。未知の他軸/roundを補完せず、平均価格はpendingのまま。pathをC1914の`PriceProvider`へ平均recordとして渡さない。Cが統計集計・qualification・CIを新たに作る入口でもない。

外部producerが将来適格mean/区間を別SOURCEで供給した場合だけ、既存B sidecar（identity/law/axis/cohort/parent price_source/physical_context/profile_extension/mean_interval）とD request/context keyを添付する。C1914の`FULL_WRAPPER_MEAN_COMPILED_PRICE` recordやMODEL quoteへの変換にはその根拠が必要。今回DIRECT mean record/モデル係数/価格区間は作成しない。1 pathでもN個shotでも、同じlawの適格meanを得たと認定しない。

## 既存証拠の扱い

- LD1 m0/m6はP1/P2の元direct参照を使い、再compileしない。P2 physical metadata一致は確認したが、新A/B/C requestへの完全な科学的bridge・mean資格は未取得。元development/fresh/replayを維持する。
- Full PFはe7d `HANDOFF_C/COVERAGE_V2/receipt_trace`の同domain/profileのEXACTを再利用し、Cで監査を再実装しない。別lowering・truncated・別q/Tへの変更は別問題。
- 旧26 E[S]は固定LD1/理想lawの構造平均。LD2、実sampler residual、Full-H等へ暗黙移送しない。
- S＋残差、加法性、q-affine、C/S比、outer-windowは現時点で正式価格を供給する保証ではない。未検証のlong-q/R、rare、law距離、finite-cohort/共有誤差を0にしない。

## 担当・残条件

**C**：E0接続receiptが必要か、以後どの仮説・固定予測・domainを検証するかを明示する。E1はqとRが同時に変わるためR係数の識別点にしない。平均モデルの精度判定には別の有限n/不確かさ設計が必要。

**B**：比較family・N/riskの資格・許容価格誤差/比較差を供給し、正式Gが未知であることを保持する。全候補の全round高精度化や一律2%/95%をDへ課す前提にしない。

**ユーザー**：必要なcellだけの完全SOURCE・axes/n/cohort/index/seed・出力先・科学call範囲を別途明示承認するか判断する。現在E0/E1とも科学実行未承認。

**Codex D**：承認対象が定まれば既存pipelineを使う最小caller差分を人工確認・commit固定し、そのSOURCEへの承認範囲内だけ取得する。今回の作業は技術成果のcommit/push・remote照合後 **mandatory STOP**。m12/T2 retry、追加sampling/compile、fit/mean/CI/Gへ自動進行しない。
