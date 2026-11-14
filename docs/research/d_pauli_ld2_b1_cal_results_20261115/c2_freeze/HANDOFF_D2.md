# C2 → D2: PR-Pauli LD2価格モデル・観測前予測の凍結

**状態：LD2有限cohortで較正済み・係数と予測を凍結・独立holdout未評価。**
2モデル・4予測を凍結した。Cのmerge・Cによる再fitを引渡し条件にしない。

## 固定SOURCEと読取

- C2 branch：`codex/pauli-ld2-price-freeze-20261115`
- C2基点：`12372fa6b53c878dc5dd124e036572ba0cd82061`（固定C fit実装を含む）
- **凍結結果commit：`b74e4e536392c2bbe94cd371436615fc56c0564f`**
- **結果path：`docs/research/d_pauli_ld2_b1_cal_results_20261115/c2_freeze/run01/FROZEN_MODELS.json`**
- [機械可読HANDOFF](HANDOFF_D2.json)：`frozen_model_source`と各軸モデル・各予測へのselector。
- fit実行SOURCE：結果の`fit_source.commit`と`fit_source.path`。
  PR_STEP_V1 / `fit_identified`は`7a0c9468b2499fac9fa1cd279ae2980c9dab048e`の
  `src/trottertracks/finite_price_design.py`を変更せず使用。
  exact OLSは同入口が指定するC `c628e9e8447c11c984064b922cbce6c52dc72d48`の
  `src/trottertracks/price_model_validation.py`。C2基点の同file・直接依存とbyte一致。
- 正本入力：D2 `5dac74959ed9ea49465d393fc792ef5b4727edba`の
  `docs/research/d_pauli_ld2_b1_cal_results_20261115/C_HANDOFF.json`。
- cell/cohort/seed/canonical IDの正本：`043c7567451e8f502a1cf0f425d32efe0a50f5dc`の
  `docs/research/d_pauli_b1_caller_preparation_20261114/CAL_AFTER_W_METADATA.json`、
  `dictionaries.LD2`。D2取得jobの7 cellは同SHAのJOB_TEMPLATESと一致し、採択flagだけが異なる。

D2ではbranchをfetchすれば結果blobを直接読める。checkout/merge/refitは不要。

```bash
git fetch origin codex/pauli-ld2-price-freeze-20261115
git show b74e4e536392c2bbe94cd371436615fc56c0564f:docs/research/d_pauli_ld2_b1_cal_results_20261115/c2_freeze/run01/FROZEN_MODELS.json
```

```python
import json, subprocess
reference = {
    'commit': 'b74e4e536392c2bbe94cd371436615fc56c0564f',
    'path': 'docs/research/d_pauli_ld2_b1_cal_results_20261115/c2_freeze/run01/FROZEN_MODELS.json',
}
frozen = json.loads(subprocess.check_output([
    'git', 'show', reference['commit'] + ':' + reference['path']]))
x_model = frozen['models']['X']
y_model = frozen['models']['Y']
predictions = {(r['m'], r['axis']): r for r in frozen['predictions']}
# 将来のD2 job用metadata: job['frozen_model_source'] = reference
# ここではjob実行・holdout価格の取得/開封/判定を行わない。
```

各exact値は`{numerator, denominator}`。`inputs`の10項目に、正本record selector、
保存reader check、sidecar、元raw公開SOURCE・実行SOURCE、辞書、physical request locator、
cohort/seed、個別receipt/fingerprintを保持している。

## 2本のモデル

H4直線1Å・STO-3G・保存DF rank10・L_D=2・finite-K=2、DF head S2＋Pauli tail、
測定付きfull controlled Hadamard wrapperのcompiled RZ数を対象とする。state preparation費用は除外。
compiler Qiskit1.3.0、basis rz/sx/x/cx、optimization1、transpiler seed17、P=0、
control symmetric_directional、lowering `PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF`、
scalarはc＋保存s_Rを総Tで一回適用。完全な入力・profile・辞書locatorは結果の`scope`にある。
LD2辞書SHA-256は`dd986e52057aa01a330a83dd147316696dbae669dc8a51828ee2a54734016ea7`。

式は軸aごとに `C_a(q,R)=β0,a+β1,a q+β2,a qR`。
特徴量順は`[one, pf_step_count, event_count]`＝`[1,q,qR]`。
**各軸5 cell、各cell等重みのexact OLS**。X/Yをpoolせず、18 receiptを18観測としてfitしない。

| 軸 | β0（one） | β1（pf_step_count） | β2（event_count） |
|---|---|---|---|
| X | 26878722857/90343469 | 692339588141/361373876 | 2289251695/361373876 |
| Y | 26878722857/90343469 | 692339588141/361373876 | 2289251695/361373876 |

両軸の数値は一致したが、入力locator・domain.axis・係数・予測は別々に保存。
`calibration_residuals`はS66、S526、S4205、Q8R66、P2R66順の絶対残差で、fitの診断のみ。
係数の統計区間、law全体の期待価格、独立path数の増加を認定しない。
モデル適用範囲はこのLD2有限較正と宣言済みm5/m6予測まで。候補価格への資格付与はfalse。

| cell | q | R | T | n_paths | X有限cohort RZ | Y有限cohort RZ | receipt数 X/Y |
|---|---:|---:|---:|---:|---|---|---|
| S66 | 1 | 66 | 0.1 | 2 | 2642 | 2642 | 2/2 |
| S526 | 1 | 526 | 0.1 | 2 | 11045/2 | 11045/2 | 2/2 |
| S4205 | 1 | 4205 | 0.1 | 2 | 28854 | 28854 | 2/2 |
| Q8R66 | 8 | 66 | 0.8 | 2 | 37935/2 | 37935/2 | 2/2 |
| P2R66 | 2 | 66 | 0.2 | 1 | 4977 | 4977 | 1/1 |

元8 fresh path＋1 derived path。P2R66はS66 path0→path1の順で連結し、親と相関する。
`paired_dependence.ordered_parent_locators`は親raw公開commit
`ff536ed2a6effc6c01899a7dcde76e5361448b85`と親実行SOURCE
`355e7074df806b60a31160a51899ec15d6b54e33`を区別して保持する。
B0/Wは較正に含めていない。元のSOURCE/raw/receipt/STOPは変更していない。

## 観測前に固定した4予測

| round | 軸 | q | R | T | exact予測RZ | 後続fresh path数 |
|---|---|---:|---:|---:|---|---:|
| m5 | X | 32 | 33 | 3.2 | 6169957875465/90343469 | 2 |
| m5 | Y | 32 | 33 | 3.2 | 6169957875465/90343469 | 2 |
| m6 | X | 64 | 66 | 6.4 | 13521761923033/90343469 | 1 |
| m6 | Y | 64 | 66 | 6.4 | 13521761923033/90343469 | 1 |

m5 canonical physical ID：`RMSE:01489f6e391e9917e96eaa56a0006a91b6099e60956cb5fcea3a516b24df5802`。
m6 canonical physical ID：`RMSE:8826b5c0b36fdb91115109a85c6f2eca40cf1989bb3b3e67d8cbcd1598043529`。
特徴量はそれぞれ`[1,32,1056]`、`[1,64,4224]`。
各予測にround・軸・canonical ID・cohort・固定fresh seed・n_paths・metadata SOURCEを保持。
D2は同round・同軸の有限cohort観測値と対応付ける。4予測に対して後続compile予定は6件。
現時点の`observed_RZ`/`holdout_pass`はすべてnull、Pauli許容差もnull（未採択）。
決定論PFの2%規則は適用していない。

## 検証・未解決事項・STOP

[preflight](PREFLIGHT.md)の§10.5照合範囲では同等凍結モデル・進行中jobは見つからず、
D2の保存済み10 record/18 receiptと5 reader checkを再利用した。
Cの既存PFモデルは辞書・特徴量・科学domainが異なり、今回のfitを代替しない。
新しい科学処理は実価格2 fit＋4予測だけ。追加compile/samplingは0。

人工5 test PASS（[TESTS](TESTS.txt)）。実保存値では正規方程式のexact直交性、
残差encoding、4予測のexact算術を再fitなしで照合しPASS。
[D2_INTERFACE_CHECK](D2_INTERFACE_CHECK.json)は上の凍結commitを読み、固定D2
`13ab1acfaebe5f33958e1db67862f5c45725ab91`の`frozen_model_gate`本体を変更せず実行して
`FROZEN_METADATA_CHECKED_NOT_MODEL_QUALIFICATION`。
seed lookupは固定計画の保存seed表に束縛したmetadata検査であり、D2全runnerやcompileは実行していない。
著者であるC2の技術確認で、別reviewerによる独立レビューではない。

独立holdoutは未評価。実LD2のorder2 product・非自明イベント位相は
`UNKNOWN_UNOBSERVED`を維持。law期待価格E[C]、CI、N、G、LD1移送、全方式の総費用順位は未評価。

**ロードマップへの影響**：従来の「LD2較正10 record引渡し済み・fit待ち」から
「有限cohortで軸別2モデルと4予測の凍結完了・D2独立holdout未評価」へ進んだ。
凍結成果を受け取るためのC merge/refit依存はない。次の取得・開封・判定とPauli許容差の採択は残る。
ロードマップ本体は統合担当が編集する。

**mandatory STOP**：引渡しまでで終了。m5/m6の取得・開封・判定、追加sampling、LD1移送、
E[C]・CI・N・Gには進まない。C・D2の作業branchへのcommit/push、共有環境の変更は行わない。
