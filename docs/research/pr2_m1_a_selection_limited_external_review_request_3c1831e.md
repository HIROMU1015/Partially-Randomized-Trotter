# GPTへの研究方針レビュー依頼：PR-2 M1-A `SELECTION_LIMITED`後

## 依頼

GitHub上のcommit `3c1831e326c27c5f679b3820997f27916d26ed9f`を固定対象として、PR-2の次の研究方針を
批判的にレビューしてください。今回求めるのは追加計算や実装ではなく、次の三択判断です。

1. `PROCEED_BOUNDED_COMPILE_EXPANSION`
2. `NARROW_TO_TECHNICAL_NOTE`
3. `STOP_RESOURCE_STUDY`

M1-Aのhard barrierは正式に`SELECTION_LIMITED`を返しています。現行authorizationではM1-B、held-out、
S3、追加diagnosticを実行できません。結果を見てcandidate、threshold、selectorを変更する救済も禁止です。

## 最初に読む資料

1. [`docs/pr2_matched_accuracy_m1_a_validation.md`](../pr2_matched_accuracy_m1_a_validation.md)
2. [`artifacts/pr2_matched_accuracy_m1_execution/2026-09-30/pr2_matched_accuracy_m1_a_result_v1.json`](../../artifacts/pr2_matched_accuracy_m1_execution/2026-09-30/pr2_matched_accuracy_m1_a_result_v1.json)
3. [`docs/research/pr2_matched_accuracy_prior_art_gate_v1.md`](pr2_matched_accuracy_prior_art_gate_v1.md)
4. [`docs/research/pr2_matched_accuracy_resource_contract_v1.md`](pr2_matched_accuracy_resource_contract_v1.md)
5. [`docs/research/pr2_matched_accuracy_m1_preexecution_amendment_v2.md`](pr2_matched_accuracy_m1_preexecution_amendment_v2.md)

実装identityまたは失敗監査が必要な場合だけ、次も参照してください。

- [`docs/research/pr2_matched_accuracy_m1_implementation_contract_v1.md`](pr2_matched_accuracy_m1_implementation_contract_v1.md)
- [`docs/research/pr2_matched_accuracy_m1_execution_authorization_v1_1.md`](pr2_matched_accuracy_m1_execution_authorization_v1_1.md)
- [`VALIDATION_STATUS.md`](../../VALIDATION_STATUS.md)
- [`artifacts/validation_manifest.json`](../../artifacts/validation_manifest.json)

旧presentation案、notebook、旧protocol、旧S0系列を現行仕様として使わないでください。

## 固定された事実

- 対象：H4 linear 1.00 Å、STO-3G、DF rank 12、sector 8 qubitのdevelopment snapshot。
- 物理時間：`T=0.8`。`q={1,2,4,8}`、random `r={1,2,4,8,16,32}`、`K={2,4}`。
- 評価：208 base候補＋2件の事前規則によるr64 boundary候補、計210候補。
- accuracy適格：206/210。random B2/B3は194/194適格。
- proxy frontier：64候補。
- 結果前selectorによる選択：16候補。
- cap外に残ったproxy非支配候補：52候補。
- barrier理由：`unselected_proxy_nondominated_candidates`。
- status：`SELECTION_LIMITED`。
- compile job、circuit、compile、trajectory、full wrapper、quantum shot：全て0。
- held-out H4 1.30 Åのpath/stat/hash/load/signal/cost/ranking：全て0。
- result fingerprint：`422f898bba1e3849d0f45830082b76d4f42da436e2b49796e562cd79fc716c9e`。
- result file SHA-256：`1f960d7a33296e2dcb74d497e360572b26409dc9aeae01522335e7b91ed81086`。

M1-Aはcompile前action proxyまでです。partial、deterministic、random-dominant、全体winnerのいずれも
確定していません。52件という数は「52件全てをcompileすべき」という意味ではありません。

## レビューで判断してほしい点

### A. compile上限拡張に科学的価値があるか

- 64件のproxy frontierと52件のcap外候補は、16-cell選抜が結論不能である十分な証拠か。
- それとも、proxy目的が弱くfrontierが広がっただけで、compile数を増やしても研究価値が低いか。
- matched-accuracy、discard baseline、DF-prefix、finite cutoff、normalization、full-wrapper compileを組み合わせた
  現scopeは、先行研究との差として追加compile費用に見合うか。
- limited result自体が、有用なnegative resultまたは方法論上の知見として十分か。

### B. 拡張する場合の最小の結果前契約

`PROCEED_BOUNDED_COMPILE_EXPANSION`を選ぶ場合は、次を結果前に固定できる形で具体化してください。

- compileするcandidateの選抜規則と正確な上限。結果を見た手選びは禁止。
- 16件済みという仮定を置かない。実際にはcompile 0なので、全taskを新規に数える。
- trajectory数、replication、process上限、wrapper上限、停止・再開規則。
- deterministic/discard baselineを含むmatched-accuracy比較規則。
- `CONTINUE_TO_FROZEN_TRANSFER_REVIEW / NARROW_TO_TECHNICAL_NOTE / STOP`の数値gate。
- compile後もheld-outを開ける前に停止する規則。
- 追加計算で解消できる不確実性と、解消できない新規性・一般化限界の分離。

「とりあえず全64件」ではなく、研究結論を変え得る最小範囲と、その範囲で結論可能になる理由を示して
ください。有限の妥当な上限を結果前に定められない場合は、拡張を推奨しないでください。

### C. technical noteへ縮小する場合

`NARROW_TO_TECHNICAL_NOTE`を選ぶ場合は、追加計算なしで成立する最小claimを示してください。例えば、

- fixed-stepの有望性がmatched-accuracy候補展開で多数frontierへ分解したこと。
- 小さいcompile capではresource winnerを識別できなかったこと。
- normalization・shot・deterministic-prefix costの共同評価がcandidate selectionを難しくすること。

をどこまで主張できるか、逆にcompiled winner、held-out transfer、一般分子への優位性など何を主張しては
ならないかを明示してください。

### D. 終了する場合

`STOP_RESOURCE_STUDY`を選ぶ場合は、重複性、effect size、新規性、計算費用、結論可能性のどれが決定的かを
示し、保存すべきnegative resultと、今後再開するために必要な新情報を分離してください。

## 禁止事項

- held-out H4 1.30 Åを開く提案を、この判断より先に置かない。
- M1-A action proxyをcompiled costまたは最終total costとして扱わない。
- partialがwinnerだった、またはdeterministicがwinnerだったと推測しない。
- 16-cell selector、candidate grid、accuracy thresholdを結果後に変更してlimitedを解消しない。
- 先行研究gateで既知と整理した一般的なimportance sampling、部分ランダム化、single-ancilla QPE、
  end-to-end resource estimationを新規claimに戻さない。
- H4 1.00 Å一条件から一般分子またはH12へ外挿しない。

## 回答形式

次の順序で回答してください。

1. 三択の判定codeを一つ。
2. 最も強い根拠を3点以内。
3. この結果から成立するclaimと成立しないclaim。
4. `PROCEED_BOUNDED_COMPILE_EXPANSION`の場合だけ、最小の結果前M1-B契約案。
5. 次のmandatory STOP地点。

境界的なら三択を曖昧に併記せず、主判定を一つ選んだ上で、判定を覆す最小条件を一件だけ示してください。
