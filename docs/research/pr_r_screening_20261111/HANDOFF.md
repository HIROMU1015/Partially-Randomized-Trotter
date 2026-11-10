# B/C向け：R候補スクリーニング

A実装SOURCE `e9d0bb4d266b365da73185b2449e135355d44aec`。A前段`5ca2dce3234f3e4a971f4a8c852da676f050e315`とB interface `7bca8f9642d1c42ca3fede4d31b2cdd8dcef46ae`を前提とする。既存A branchを継続し、B/Cのworktree/source/envは変更していない。

B branchへ接続する際は、A5ca2dce32の`pr_r_dependent_n.py`と今回のscreen module/CLIを同じcheckoutへ用意する。既存の共通pipeline・integer helperを再利用し、Bの費用会計拡張をA側へ複製する必要はない。

## 生成入口とSOURCE依存

```sh
python -B scripts/plan_pr_r_screening.py --system H4 \
  --factors '["1/2","1","2"]' --reference-r-by-round '{"12":[8]}' \
  --output <fresh-metadata.json>
```

CLIはmetadata生成だけ。lambda_R/B/N/C/Gを評価する機能はない。H6も同じ入口で、対象はm4..12。倍率とreference Rは有限pilot候補を作る指定で、科学的上限/採択ではない。

B moduleを改変・複製せず、固定Git objectからbuild_menu/request_context/coverageの関数本体と必要な定数だけを読み込む。費用evaluatorを露出しない。single-branch clone等で7bca8f964がなければ、使用前にその固定commitを取得する必要がある（例：`git fetch origin 7bca8f9642d1c42ca3fede4d31b2cdd8dcef46ae`）。latest branchや別版へ自動fallbackしない。B側で本来のrpe_r_selectionを使うときも、このSOURCEとの互換性を確認する。

## packet

- `base`：Aの凍結one-candidate-per-arm/round task。
- `menu`：Bのbuild_menuによる全有限候補。PFとH6初期roundを保持。
- `rows`：変更対象PRのdescriptor。system/m/T/qRPE/qPF/δ/formula/prefix/K/R、structureのsigned weights/allocation/event_count、candidate_id、`request.identity/law_binding`、baseline flag、normalization方法/未評価値、詳細N要否、未検証事項を含む。
- `seeds`：anchor qPF/Rとcaller倍率。H6はqRPEをseed計算に使わない。
- `R_upper_bound=null, range_complete=false`：生成bandの外は未探索。新旧uniform gridの十分性を主張しない。

N/C producerに渡すrequestは`row.request`をそのまま使う。B schemaや候補台帳は再実装していない。target lawの宣言はactual samplerとideal lawの距離の証明ではない。

## 将来のnormalization科学call

実分子では別途承認後だけ呼ぶ。

```python
from trottertracks import pr_r_screening as screen
# tailは既存保存prefix tailのlambda。新しいDF/再分割はしない。
tail = {"binding": packet["menu"]["binding"], "prefix": 1,  # H6は7
        "lambda_R": saved_lambda, "source": pinned_saved_tail_source}
scored = screen.screen_normalization(root, packet, tail,
    shot_inflation_budget=None, budget_source=None)  # 無閾値・記録だけ
# 感度帯Fを使う場合だけ、F>1と根拠/SOURCEを明示する。
```

scoreは既存finite式のlog B、2log B、small-tau主項（採否に使わない）、max|τ|、1 ULP累積感度を返す。bが1へ丸まる等は精度未資格、予算境界は保留する。新Nや位相PASSは生成しない。`detailed_N_required=True`は後回し候補でも維持する。

`priority_menu`は次の詳細計算queue用。**Bでの比較対象は全menuのまま保持し、後回し/未評価候補があればcoverage不足を残す。** normalizationだけで費用劣位を宣言しない。全Rの探索・科学的採択・適応的追加・retryを自動起動しない。

## AのN providerとC価格へ

```python
request = screen.A_request(root, packet, candidate_id, numerical_source)
a_task, a_row = request["A_proposal"]["task"], request["A_row"]
# 承認された範囲でA.RBatchを使い、このcandidateのrecordを作る。
# A identity == B_request.identity はmetadataで検証済み。
```

Aは一つのroundに一つのRを持つtaskを扱うため、同roundの別Rは別taskにする。異なるroundの一候補ずつは既存RBatchでまとめられる。context/referenceの再利用はH/state/T/SOURCEを照合し、保存baselineも比較時は同じR-aware allowanceでscalar再解析する。その再解析も別途承認範囲へ含める。

`law_binding`は実際の科学SOURCE/定義・R/K/qPFがB_requestと一致した証拠を伴って返す。metadataだけを既存Nへ機械的に付与して、新RのN資格を作らない。actual implementation/law・rigorous・full RPEの未資格と巨大整数NはAの既存interfaceを保持する。

Cも同じrequestのR、qPF、δ、formula、prefix/K、signed配分、full-wrapper/compilerを使用する。価格の新R/full candidate/domain/law期待値/共有不確かさの資格はC担当。固定R4/R16価格を代用せず、B/attenuationをCへ再乗算しない。Bのexact費用会計・R選択・不足/uncertaintyはその既存実装へ渡す。

## 未実行の有限案

EXECUTION_PLAN.jsonのphase0は代表6round・24候補のnormalization-only。phase1案はH4 m12/R8、H6 m12/R32の新N各1と保存baseline各1の同じ数値policyでの比較。どちらも未承認・未実行で、phase0完了をphase1や価格/Gの実行承認に読み替えない。
