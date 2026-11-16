# B2 round03の通常同期への接続修正

PR-Pauliの方式全体の保存最安は **410,199,237 RZ**、絶対κ=2限定の保存最安は **3508632333/7 RZ（501,233,190.43）**。異なるconfiguration・配分の2点であり、元κの倍率×2と絶対κ=2を同一視しない。結論は保存候補集合内の、保存Q3と未資格MODEL価格規則による条件付き比較に限定する。

正本：[比較表](../h4_saved_rpe_cost_catalog_20261116/run01/REPORT.md)／[CATALOG](../h4_saved_rpe_cost_catalog_20261116/run01/CATALOG.json)／[同期状況](../h4_saved_rpe_cost_catalog_20261116/SYNC_STATUS.md)。[HANDOFF](HANDOFF.md)に通常入口を記載する。

## 止まっていた箇所

着手時のC HEADは`1f8ff9fcb6e94a731754fa88df299bb5fa53efa8`、最終cronは2026-11-16 12:00 UTCのUNCHANGEDだった。公開branch検出・fetch・成果ファイル検出は成功しており、`codex/b2-rpe-split-preparation-20261116`上の`d7a976a629a6db68197b24f8f86734e109c8bf5d`のround03はPUBLICATIONSに存在した。

不具合は受入段階にあった。`ROUND_RESULT`のconfigurationが既存点と一致するだけでINTEGRATEDと記録し、新しい配分・費用・MSEを追加していなかった。また最終`KAPPA_ABS_RESULTS.json`と公開`best_points`は検出対象外だった。round03の`ROUND_RESULT.methods.PR_PAULI.best`自体も410,335,030.43 RZの途中snapshotであり、最終選択410,199,237 RZと異なる。configuration一致やそのsnapshotのbestを最終選択の代用にしない。

## 通常経路の修正

- 既存10分cronの同じ`sync_saved_rpe_publications.py`へ、最終索引と公開`best_points/*/REQUEST.json`の検出を追加した。特定round名・commitを実装へ埋め込まない。final索引がない次roundでも、公開されたselected REQUESTから受入できる。
- `REQUEST -> EVALUATIONSの元OUTCOME hash -> 公開OUTCOME/CASE_INPUTS -> configuration/signal/total view/binding -> 保存26軸価格coverage/実runtime registry`を既存Git readerで照合する。MSEはOUTCOMEのexact値を保持し、再評価しない。
- `ROUND_RESULT`はconfig・N_X/N_Y・status・exact費用の全一致を要求する参照に変更した。不足するsnapshotは理由付きPENDINGとし、別途完全な公開selected pointの受入を妨げない。
- `method_minima.PR_PAULI`と`restricted_minima.PR_PAULI_KAPPA_ABSOLUTE_2`を分けて公開する。最安比較器を再利用し、新しいQ3・価格予測・optimizerは起動しない。

## 公開証拠

固定B2 commit：`d7a976a629a6db68197b24f8f86734e109c8bf5d`。以下のpathは`docs/research/rpe_method_optimizer_h4_20261116/kappa_abs_round03/`からの相対位置。

|選択範囲|configuration|X/Y各配分|元公開path|OUTCOME sha256|
|---|---|---|---|---|
|方式全体|`cb1aa7487fbfb938c629961014ea5911cc5f3e29b9fc45e16b872ac8fb60c052`|[48,45,42,39,22,27,14,20,16,16,8,7,2]|`best_points/LD5_kappa_x2_edd63e0befd3af591809/`|`875ccc86920db1ee7051fc83cfd98678b06ec0f246f3e222e5b60962297276d2`|
|絶対κ=2|`ff6b21aee0a04845c587e73f13622e982bd5ed80c303e66aab83c89d8a6168eb`|[48,45,42,39,36,32,14,21,17,14,7,7,5]|`best_points/LD5_kappa2_e3cb2cfda719117d6222/`|`9abac97b2cadc5faee22ce420f3c823e5a8dd06822e1643799a30fe84230434f`|

元hash・費用・配分は`run01/EVALUATIONS.json.rows[78]`／`[74]`、最終選択は`KAPPA_ABS_RESULTS.json`。両点とも保存判定`MODEL_BOUND_MEETS_ADOPTED_THRESHOLD`。κ絶対値・元κ倍率はcanonicalから別fieldへ保持する。

価格は`run01/NEEDS.json`の同configの`MODEL_REFERENCE:26` summaryと、`ROUND_RESULT.round.price_rules_overrides`が指定する`PRICE_RULES_REGISTRY.json`に束縛した保存総額。元MODELは`fb0d60b0a1e7c31aee73b2b31f802dda7b2f0125:docs/research/a_pauli_ld5_cal_price_20261116/PRICE_26AXIS.json`、実装SOURCEは`c1f95e0806f5478a1ee5ce9ecb148ce9d10461aa`。26の新単価は公開されていないためnullのまま、SOURCE付きcoverageを保持する。全26軸MODEL・外挿、`qualified_for_candidate=false`、正式E[C]・CI・Gは未認定。

旧410,470,823.86 RZ点・旧予算・原結果・STOPは保全する。探索の停止理由`ONLY_SMALL_OR_DEFERRED_LEFT`を保持し、最適性・収束の認定にはしない。

## 検証と運用

保存catalog回帰14件、同期7件、計21件PASS。通常discoveryで最終索引と公開2点を検出し、両最安・exact OUTCOME hash・26軸coverage・旧点保全を確認した。hash・配分・価格bindingの改変はPENDING、他のFull PF結果は反映する。同じ公開入力を再同期してcatalogが一致し、最終索引なしで2公開pointだけからも読めることを確認した。費用だけの変更でロードマップ本文を変更しない既存検査も再利用した。[実行記録](TESTS.txt)。

primary agentが同期/consumer実装・回帰と運用確認を担当し、sub-agentがB2公開結果の所在・hash・metadataの読取調査を担当した。独立科学レビューではない。新signal・sampling・compile・fit・Q3は0件。通常10分cronによる反映の固定commit・運用結果はHANDOFFへ追記する。
