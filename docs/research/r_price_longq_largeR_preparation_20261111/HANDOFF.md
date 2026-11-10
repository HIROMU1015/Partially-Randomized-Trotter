# B向け：A P1と長q・大R価格計画の接続

[REPORT](REPORT.md)／[EXECUTION_PLAN](EXECUTION_PLAN.json)。metadata SOURCE `f0daeb96bf53b43ce4e6e1d61ed351a4f7c688b0`。科学実行未承認・未実行。新しいprice recordは空配列で、S/C/予測/残差/Gはnull。Bの共通API、N provider、optimizerは変更していない。

## 固定公開成果とcoverage

| 役割 | 公開SOURCE |
|---|---|
| A P1・12件の同policy N | `71ba49bb28b295aa28e51101bc3b1bf07d500767:docs/research/pr_r_detailed_n_p1_results_20261111/run01/N_HANDOFF.json`、科学SOURCE `a1a79213a9b5331145e20e8778a3be9d9fc4e497` |
| C1の8価格セル | `2e6a0cdf5a647058cc7afeeb5741bfafa24c974b:docs/research/r_price_phase1_results_20261111/run01` |
| C1凍結係数 | `7eeabf0ae482ee610d667e28470692d209f2bb21`、同dir `FROZEN_MODELS.json` |
| B受入/API | `724d313b9c8445763f0fe9c41feb573df28ba5d5`、`rpe_r_price_intake.py` / `rpe_r_selection.py` |

PLANの`A_P1_coverage`はA保存N_X/N_Yを整数のままcopyし、12 ID/完全identity/同policyを照合した静的表。Nを再計算せず、費用比・感度・ランキングは付けていない。P1のC1 direct/model保存重複はH6 m4/R16・R32だけで、いずれも校正点。C1の独立検証点と一致するP1候補は0。残る10点は確認したcatalogではC未取得、C1数式/旧固定R proxyで自動補完しない。

Bが別途行うC1取込みはB既存`C1PriceIntake.read_published`に委ねる。Cはその作業を重複実装・実行していない。C1 recordの有限cohort/nominal covarianceを全law期待価格へ上げない。既定`qualified_expectation`で使える全law価格を得たとは扱わず、directは明示した`finite_cohort_scenario`での利用資格を別途確認する。modelの`qualified_for_candidate=false`は保持する。

## 提案domainと順序

第一案P1_DIRECTはPLANのcomponent Q＋QR、次の4候補・40 full wrapper。全てAの同policy Nが保存済みである。

| 系/m | qPF | R | ID | classical n / wrapper数 |
|---|---:|---:|---|---|
| H4/6 | 64 | 4 | `65c51c5cb3e39a2266e3` | 4 / 8 |
| H4/6 | 64 | 128 | `2a8c6718220eec59d419` | 6 / 12 |
| H6/8 | 16 | 16 | `53797d9e5a2eeec31a13` | 4 / 8 |
| H6/8 | 16 | 128 | `55a8eb0ae50ec3f7c713` | 6 / 12 |

原因分離に必要なら、R-only（H4 q8/R128、H6 q1/R128）を別16 wrapperとして検討する。SHORT（H4 q1/R4、H6 q1/R11）も別の任意16 wrapper案。全packageを実行する前提・自動移行はない。n4/n6はpilot予算案で、精度充足やCI保証ではない。R256、H4 q1/R2/R8、m12はmissingを保持する。

候補・signed weights/allocation・qRPE/qPF・law・compiler・profile・P0はPLANのfull `request` / `structure`に保存済み。H4 S2/δ.1/p1/K2、H6 m5_best/δ1.6/p7/K2/11signed piece/R≥11、元Qiskit1.3.0/opt1/seed17/測定付きfull wrapper。別δ/prefix、H6早期別構成へ移植しない。

## 将来のprice record受入

新価格取得前の現時点では、C providerはこの計画を新価格として返さない。欠測行は[]またはB既存のmissing表現とし、mean=0を作らない。C1の`describe_request.R_evidence_status`は旧固定Rに対するmetadataで、今回のfrozen-model検証計画が再fitを要求するという意味ではない。

将来の許可された価格取得後はBの既存record形式で次を供給する。

- `identity`はfull candidate/時刻/qPF/qRPE/R/K、H/state/snapshot、configuration、contract、compiler/profileと完全一致。`law_binding`は元C sampler宣言を保ち、BのN lawと文字列だけで同一化しない。B724dのsource対応付けはideal/sampler距離の保証ではない。
- `classification=DIRECT_COMPILED_FINITE_SAMPLE`と`MODEL_PREDICTION`を別cohortで保存。n_pathsとpaired X/Y、観測unitを保持し、quantum Nと混同しない。有限meanを全lawへ昇格せず、normalization BをCへ再乗算しない。
- 新点はC1凍結係数に対する独立transfer検証で、model domainはその実行で許可/取得した有限点だけ。資格は未採用のまま。新model/fitにデータを転用する場合は役割変更と新holdoutを別の科学計画へ記載する。
- `source`は新科学SOURCE/固定plan/publication/各axis fingerprint・cohortを追跡可能にする。C1の旧SOURCE・役割・STOPを変更しない。calibration shared group、paired covariance、nominal係数共分散を別に持ち、未知R-q systematic/cross-R/round/joint/total CIはnull。

Bの現`C1PriceIntake`は固定8セルの検査器なので、新C2 publicationをその入口へ混ぜない。追加が必要なのは、C2の有限point/role/sourceを読むB側の小さい受入入口である。共通会計やR選択器の再実装は不要。Cは受入schema案と価格側SOURCEを渡し、共通変更はBへ委ねる。

旧`identity.binding.T`/model.binding.Tは参照task値.8を含む。実時刻は`identity.candidate.T`とqPFδ、回路signed timeで照合する。C recordだけを正規化してBのexact identityを壊さず、必要ならBのmenu/N/Cで一貫した派生metadataとして扱う。候補全体・R・lawを省いた価格cache keyは使わない。

## 価格精度・全費用比較への不足

第一案の直接有限価格を取得しても、全13roundやR256、全law、model採用、cross-R/round同時不確かさ、RPE成功・選択後保証は残る。S計数の効率化も、巨大回路でのcompiler対応資格とは別である。C1係数だけでは未取得Sは得られない。

将来BはNを固定し、価格区間・共有依存を保持して候補費用差を識別できる精度を検討できる。|ΔG|≤ΣN|ΔC|は設計上の関係で、今回その実数値評価をしていない。N×C/G・ratio/ランキング/最適Rの新計算は本HANDOFFの許可に含まれない。

次：ユーザーが有限packageとpilot目的・予算を判断する。Codex Cは選択したbounded runner/SOURCE/seedを準備し、別途具体的な科学実行許可がある範囲だけ実行する。Codex Bは自身の許可scope内で保存成果の取込みを担当する。公開照合後mandatory STOP、追加科学計算へ進まない。
