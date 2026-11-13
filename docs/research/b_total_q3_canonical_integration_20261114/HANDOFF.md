# B total decoder/Q3 canonical binding API

固定SOURCEと到達範囲は[REPORT](REPORT.md)。全APIはopt-in。`evaluate`、`cost`、`compare`を実データで数値適用するには別の有限科学実行許可が必要。今回実行したのはmetadata照合と人工テストだけである。

## 1. Cインターフェースと保存Pauli signal

```python
from trottertracks import rpe_c_interfaces as C
from trottertracks import rpe_pauli_signal_intake as A
api = C.interfaces(str(root))       # C fixed c628e9e84; live B dependencies
items = A.accept_saved(root, integration=api.pipeline)['results']
item = items[0]                    # LD1; index 1 is LD2
```

Cの新しい`plan(..., implementation_sources=...)`と`provider_binding_check`を既存intakeが再利用する。元26recordとC物理contextの一致を確認でき、前回の`UNKNOWN_PAULI_PHYSICAL_ID_ADAPTER_REQUIRED`はこのSOURCEでは解消した。新contractへの移行は次のB bridgeで行う。

## 2. canonical生成とsignal binding（metadata）

```python
from trottertracks import rpe_total_binding as T
source = dict(A.HANDOFF, selector=['packets', 0, 'configuration'])
bridge = T.create(root, item['configuration'], original_source=source)
view = T.bind_signal(root, bridge, item['signal_packet'])
new_config = view['configuration']
new_signal = view['signal_packet']
T.validate(root, bridge, new_config)
```

| 識別子・条件 | 扱い |
|---|---|
| 元configuration・SOURCE | 完全な元configurationとGit selectorを保存。固定Git JSONと再照合 |
| 新configuration | B.generate/B._validateが生成・検査。decoder/risk契約を含むためIDは別 |
| round candidate ID | 元/新を明示対応。Pauliではphysical candidate IDもcanonical由来のため変わる |
| law binding | configuration/candidate ID部分は変化。実際の測定law定義、κ、event mappingは同一 |
| physical circuit | H/state/snapshot、T、q、実幅、R、K、分割、辞書/hash/λ/scalar/bit順、allocation、compiler/control/P0を保全 |
| raw/normalization/数値SOURCE | 元carrierを再取得・一致確認してコピー。新しいmeanを計算せずSOURCEを付替えない |
| provider binding | Cの元/新planのcontextを比較。意味上の差をcanonical physical IDだけに限定 |
| origin | 新packetに元packet、各meanに元identity/law/provider bindingを保持 |
| 科学資格 | 元UNQUALIFIEDを保持。新contractはPROPOSED_NOT_ADOPTED。十分N/global/Gはnull |

H、時間、R、K、辞書、law、profile等まで違うconfigurationをbridgeのID変更として扱うと拒否する。固定R診断や別LDのsignalを使って旧IDを再ラベルすることも拒否する。

## 3. Q3数値caller（今回は実データで未実行）

```python
records = T.evaluate(root, bridge, new_signal, allocations,
    evaluation_target=explicit_target, statistical_contract=explicit_contract,
    source=execution_source, options=explicit_options)
```

- `options.estimator`は固定total IDを明示する。`total_state_refinement`、phase radii、exact-affine規則、count/transition有限範囲、all-event model lossは外部入力。
- 独立Codexの`Q.evaluate`をそのまま使う。旧count/closure検査に必要な元契約入力を渡すことを`deterministic_input.kernel_input`に明示し、total推定量のriskだけを新canonical契約へ結合する。旧部分DPはこのcallerでは禁止。
- 出力の`diagnostic.deterministic_risk`はtotal MSE上下界・lemma・total DP・moments・有効上界のmin・Q3診断。`diagnostic.partial_decoder_risk`は旧decoderの参照診断。混ぜて加算しない。
- `deterministic_input`には新configuration/signal input、estimator/error contract/SOURCE、bridge、旧kernel入力の両方が入る。外部N証拠はこれと完全一致する必要がある。
- `n_packet/sufficient_N/global_RMSE_bound/G/G_qualified`はnull。UNKNOWNとengineering allowance、target/raw/lawの未資格は独立Codexのまま保持。

## 4. 保存価格receiptを保全した結合

```python
from trottertracks import rpe_price_implementation_guard as R
bound = T.bind_prices(root, bridge, original_prices,
    bindings=original_sidecars, cohorts=original_cohorts, policy=explicit_policy)
# metadata guard only: no G arithmetic
check = R.check(root, new_config, bound['prices'], cohorts=bound['cohorts'],
    bindings=bound['bindings'], policy=explicit_policy, canonical_binding=bound)
```

元C guardを**元configuration、元record、元sidecar/assertion SOURCE**で検査し、その後で元/新physical同一性をbridgeで検査する。元価格値・source・cohort・native/lowering・scalar・wrapper・測定軸は変えない。viewの元receipt/assertionを保全し、`receipt_reissued=false`。new IDを元C guardへ直接通す単純付替えは拒否される。

モデルの適用domain・区間の科学的coverage・law期待値はdecoder変更だけで認定しない。旧MODELの新candidate資格はfalseにして新domain証拠を要求する。元MODEL recordは変えない。cohortはAPIで既存B同様`(candidate_id, axis)`のtuple keyを使う。

正式Gの入口（**将来、適格N/Cと実行許可がある場合のみ**）：

```python
out = R.cost(root, new_config, external_N_packet, bound['prices'],
    cohorts=bound['cohorts'], price_bindings=bound['bindings'],
    comparison_policy=explicit_policy, canonical_binding=bound)
# Or use existing rpe_shot_allocation.compare(..., canonical_binding=bound).
```

この入口は算術直前に元receiptとbridgeを再検査する。`B.cost/B.select`は直接primitiveのまま。`R.select`は共通policyを検査して既存選択器を使う。

`T.compare(..., model_limit=..., qualifications=...)`は既存`D.compare`→`W.compare`への薄い接続。新canonical ID・error contract・total estimator・deterministic inputに対応する外部N packetだけを受ける。nominalは拒否。新しいN producer、資格規則の緩和、round別の独立最適化はない。現opt-in configurationは科学statusが未採択なので、SCIENTIFIC packetだけを供給しても正式適格にはならない。

## 5. Cとの境界

- Bはdecoder版間のcanonical/configuration/signal/receipt対応を担当する。
- Cは各版内のphysical-ID bridge、辞書/round SOURCE検査、route/lowering guard、価格受入を担当する。BはC plannerを再実装しない。
- 新版で発行された価格sidecarならCの通常経路を使う。元receiptを保持した版間結合は今回のB正式入口を使う。固定C `compare`にBのbound viewだけを渡し、元assertionが新IDで発行されたかのように扱わない。
- 固定CとB直接guardのPauli人工fixture結果は一致。旧Full-H RZZ/新ZZ混在、DF/Pauli route混在、Full/Truncated PFのRTE誤指定は拒否を維持する。

## 6. 第1水準（Cの会計を再利用）

```python
paper = api.paper
nominal = paper.nominal_packet(root, original_config, original_signal, source=nominal_source)
g_paper = paper.paper_cost(root, original_config, nominal, prices,
    cohorts=cohorts, price_bindings=sidecars, comparison_policy=policy)
paper_menu = paper.paper_compare([g_paper, ...])
```

これは実入力なら新しいN/G数値評価であり、今回は人工データだけで確認した。既存Bのnominal式をCが呼び、`G_paper`とtierを返す。`G_qualified/sufficient_N/qualified_winner`はnullで、Bの正式MSE比較へ混ぜない。前回Bのpassive `rpe_evaluation_levels`は保持し、その独自packet形へCの数値結果を自動再計算・昇格する処理は加えていない。

## 7. 次の担当

- **ユーザー**：Z1＋T1/Q3の正式採択、モデル診断と物理資格の範囲、target/law/loss/数値allowanceと有限実行範囲を判断する。
- **Codex B**：許可後に[有限2評価案](FINITE_SCIENCE_PLAN.md)だけを実行する。今はSTOP。
- **C/D**：価格証拠を作る場合、元/新canonical ID、Pauli D2 route、実lowering/source、cohort/lawを明示する。今回この担当へ新科学計算を依頼・起動していない。

新しい科学MSE/N/G評価・shot探索は未実行。**mandatory STOP**。
