# PR-2 M2 transfer契約 amendment v2：usable B2判定の統一

作成日：2026-10-04。外部review：`REVISE_M2_CONTRACT_BEFORE_IMPLEMENTATION`。
対象branchのreview済みHEAD：`622834832d371c7cc7b4ca3e4371d1ab0805c47f`。

## 1. 修正範囲と来歴

[契約v1](pr2_matched_accuracy_m2_held_out_transfer_contract_v1.md)の§4–6とplanのratio/support規則を、
以下の定義で置き換える。v1文書・schema・planは監査履歴としてbyte-identicalのまま保存する。
v1契約SHA-256は`a78d70c493125ef5321a989468dc41961ee5d3e053e6879a75db4b68d73e2f67`、
v1 plan SHA-256は`7b9a2c4e9fbf4e9c58b6a5e5402239b9527c49d27ccf8e8f0f1a5ae080097132`。

既存判定コードのratio分子は既にusable B2へ限定されていた。一方、v1契約・planの文言には
その限定が明示されていなかった。本修正は研究方針変更ではなく、文書・schema・plan・判定コードの
解釈を揃える結果前修正である。候補5件、accuracy式、10%閾値、32 trajectory、196 wrapper、最大5 workers、
追加0、全status後mandatory STOPは変更しない。M1-B1 evidenceも変更しない。

## 2. 共通の集合定義

`usable B2 := method=B2 AND accuracy_eligible=true AND major_cost_underestimate=false`。
ここでmajorはprimary `rz_count`に対する10%超のunderestimateだけを指す。secondary 5指標を理由に
この集合を変えない。eligible endpointはaccuracy-eligibleな固定B0/B1/B3構成とする。

- 6指標point Pareto自体は全accuracy-eligible構成から作る。supportの証人に使えるB2はusable集合だけ。
- primary ratioの分子は`min_usable_B2 G_RZ`、分母は`min_eligible_B0_B1_B3 G_RZ`。
- major-underestimated B2が最小costまたはpoint Paretoであっても、どちらのsupport経路にも使わない。
- usable B2またはeligible endpointが空ならratioは`null`。空集合のminimumを計算しない。

## 3. terminal規則と優先順位

以下を順番に適用する。研究判断は自動実行せず、どのstatusでも停止する。

1. implementation/source/data/correctness/numerical/resource gate不通過 → `IMPLEMENTATION_GATE_FAILED`。
2. usable B2が空 → `TRANSFER_NOT_SUPPORTED`。endpointも空の場合はこの規則が優先する。
3. usable B2があるがeligible endpointが空 → `TRANSFER_INCONCLUSIVE`。
4. usable B2の少なくとも一件が6指標point Paretoに残る → `TRANSFER_SUPPORTED`。
5. usable B2だけから計算したprimary ratioのupper 2SEが1.10以下 → `TRANSFER_SUPPORTED`。
6. usable B2がpoint Paretoに残らず、同ratioのlower 2SEが1.10超 → `TRANSFER_NOT_SUPPORTED`。
7. その他（lower 2SE ≤ 1.10 < upper 2SE）→ `TRANSFER_INCONCLUSIVE`。

paired-axis 32 trajectoryのdelta-method 2SEはengineering intervalでありformal CIとは呼ばない。
P≥0の状態準備感度はsecondaryのままで、上記規則を変更しない。

## 4. schema・plan・source固定

v2 planは共通`usable_b2_predicate`、ratioの分子/分母集合、terminal順序を明示する。
v2 result schemaはcandidateの`transfer_support_usable`が上のpredicateと一致すること、
`TRANSFER_SUPPORTED`にはusable B2が少なくとも一件あることを要求する。
classifierはratioに使ったusable B2/eligible endpointのIDを返し、synthetic testで除外を検査する。

未commit sourceから生成するplanは`M2_TRANSFER_CONTRACT_DRAFT_EXECUTION_NOT_AUTHORIZED`、
`source_binding_status=WORKTREE_DRAFT`と表示し、正式freezeとは呼ばない。runnerの`--draft`だけで生成できる。
正式v2 planはsource commit後、指定source全件がそのcommit blobと一致する場合だけ生成する。
旧v1 planをv2として認可することは禁止する。

## 5. claimの限定と次の停止位置

M2が検証するのはdevelopmentで固定した5構成のtransferだけである。`SUPPORTED`でも、
H4 1.30 ÅでB2 methodが最適、rank 3/q1が一般的最適、r4/r8の厳密winnerとは主張しない。
言えるのは固定intermediate-partial構成が未使用H4条件でもaccuracyとresource competitivenessを
維持したかどうかまでである。

本修正はheld-outのresolve/stat/hash/load、signal、trajectory、circuit、compile、transferを認可しない。
次は本修正のsource/plan固定とscience module/runner/testの結果前実装である。
science source commit → 別commitのresult-prior authorization → 最終pre-execution reviewを経るまで
held-outを開かない。一回のM2後は必ず停止して研究方針を全面再評価し、追加96やH5/H6へ自動進行しない。
