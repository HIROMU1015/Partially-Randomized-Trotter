# Track B BF-0 external review request

日付: 2026-10-04 JST  
status: `READY_FOR_EXTERNAL_REVIEW` / review response: **未受領**  
parent decision: `PROCEED_BF0_DESIGN_AND_NOVELTY_AUDIT`  
`science_execution_authorized=false` / `runner_creation_authorized=false`

## 1. 今回依頼する判断

Track BのBF-0三文書をレビューし、**必要な最小修正を経て、一回限りのBF-1を事前登録する価値があるか**を判断してください。研究Bの中心仮説は次です。

> 同じ次数条件を満たすPF familyの自由度を、通常のPF誤差・costだけでなくrandomized tailのfinite-RTE burdenまで考慮して設計すると、既存の最適化済み公式とは異なるdecision-relevantな設計が得られるか。

現在は五stage対称四次family内の限定hypothesisであり、新規method・性能改善・compiled advantageは未確定です。問うのは、同じfamily・自由度・探索予算に対して、ordinary-PF / known-leading-tail / finite-RTE objectiveの違いを最小pilotで判別する情報価値です。係数が変わるだけでは成功としません。

利用者指定の進行順序は、**BF-0三文書review → 必要なら最小修正 → BF-1事前登録・authorization → BF-1一回 → mandatory STOP → 研究BのRQ・新規性・着地点の全面再評価**です。reviewで重大な問題が判明した場合はpilot前に修正・縮小・停止します。問題が閉じた場合は、この中心仮説の判別を先に行い、研究方針の追加の全面再設計を先行させません。

この依頼のPASSは事前登録・準備へ進む判断です。三文書はproposalであり、PASSだけでsourceや入力が固定された実行authorizationにはなりません。BF-1 runner、Hamiltonian/signal計算、trajectory、compile、GPUを今回実行しないでください。外部レビューそのものもまだ実施・受領していません。

## 2. Review対象とidentity

| 項目 | identity |
|---|---|
| repository | `HIROMU1015/Partially-Randomized-Trotter` |
| B branch | `track-b-algorithm-codesign` |
| B worktree | `/home/abe/Project/prt-worktrees/track-b-algorithm-codesign` |
| B base / review準備時HEAD | `b6e65c6123475add5e620ec1064f361378bead95` |
| review対象 | 本bundleに収録する下記三文書のreview用草案。実行契約ではない |

読む順序は数学契約、claim matrix、pilot proposalです。レビューはMarkdown四件を含むTrack B資料commitを対象にしてください。M2 resultのbase commitだけをcheckoutしても、これらの新しい文書は入りません。

| 文書 | 主なreview箇所 | bytes | SHA-256 |
|---|---|---:|---|
| [bf0_mathematical_contract.md](bf0_mathematical_contract.md) | §§2–8: native kernel、係数domain、F/S、finite mean/normalization、allocation、accuracy、I0/I1/I2; §§9–10: 未閉鎖事項と進行順序 | 16092 | `b34f251e0332dc90f4206e00327853b6463ee61d7884392bbb06b8d51992a79a` |
| [bf0_prior_art_claim_matrix.md](bf0_prior_art_claim_matrix.md) | §§2–4: 一次本文locatorと最後の二claim; §5: 旧STOP; §§8–9: provenanceと利用者方針 | 26547 | `ba011ecde71df6f4b584bf3659b8a7e68abf284ed9b445a61f6444c7472e98af` |
| [bf1_minimal_pilot_proposal.md](bf1_minimal_pilot_proposal.md) | §§2–9: input、O/L/F、optimizer、grid、baseline、資源上限、materiality; §§10–11: BF-A/B/C、mandatory STOP、未固定事項 | 22335 | `4ba9bdad46b8e5ec195e551ea5f62e4bcbf854347d9dee7873c196365ad831a6` |

hashは今回の最小修正後の版を識別します。内容を修正した場合はreview対象のidentityを更新し、どの版への判定かを明示してください。SHA一致は内容同一性の確認であり、科学的正しさの証明ではありません。

## 3. 六つのgate

| Gate | 判定してほしいこと | BF-1前のblockerとなる場合 |
|---|---|---|
| G1 novelty / information value | Morales、SPRINT/GRADE、PR等の本文に対し、同じfamilyを有限RTE resource目的で再設計する限定候補が残るか。既知coefficient norm/leading modelだけで説明できる差かをL対照で検査する価値があるか | 先行methodが実質同じ問題を既に解き、method deltaの判別価値がない。具体的な本文・式・定理を示す |
| G2 fair comparison | O/L/Fが同じdomain・自由度・初期点・refinement rule・係数評価budgetを持ち、各armの全探索係数を共通finite参照でq/R再最適化するか | 弱いordinary baseline、予算の偏り、baselineの欠落、不公平な再最適化が結論を作る |
| G3 oracle information | I2を使うdevelopment mechanism pilotを、この段階の到達点として受け入れられるか。O/L/Fへ同じ情報accessを与え、I0/I1 methodと混同していないか | I2依存が隠れる、あるいはI2-only pilotの着地点を研究上受け入れられない。現時点でoracle-free実装済みとは主張しない |
| G4 F/S semantics | exact simplification → finite insertion → circuit exact simplificationが明確か。signed time、scalar/control phase、Kの規約、finite meanとphysical meanを一致させられるか | FとSを同じfinite algorithmとして比較する、有限biasをcompiler変更で隠す、入力/phase/時間順序が定義できない |
| G5 isolate coefficient design | r_jは固定rounding、Kは共通値と事前escape policy、q/Rは小さい固定候補であり、係数設計の効果を判別できるか | r_j/K_j/algorithm/split自由度が過剰。FのscoreがO/Lの探索を誘導する等、armの差が曖昧 |
| G6 decision / STOP | materiality、numerical guard、secondary decisionの扱い、BF-A/B/Cの境界を結果前に固定でき、全結果後STOPが保証されるか | 係数差だけでGO、何かpositiveならGO、q/Pareto変化を結果後に成功条件へ追加、成功時の自動BF-2進行 |

G1は新methodの有用性をpilot前に証明する要求ではありません。既知法への重複が明らかな場合は走らせず、狭い仮説が残る場合に一度の判別を許すかという判断です。scoped auditの「未確認」を文献の不存在と扱わないでください。

## 4. 特に閉じるべきproposal事項

development案はH4 linear 1.00 Å、STO-3G、8 qubit、DF rank12、L_D=3、T=0.8。primary epsilon_sig=0.01、alpha=0.05、epsilon=0.05は係数再探索なしのbridge採点です。新geometry、split、random alternative、held-outは含めません。

提案は32係数評価/arm、O/L/Fの三arm、q={1,2,4,8}、R_bud={5,10,20,40,80}、共通K2と事前条件付きK4。最大4000 distinct finite cells、8 CPU-hours、4 wall-hours、2 workers、aggregate RSS 8 GiB、GPU操作0です。これらはreviewする予算値であり、実装が上限に収まることは未検証です。

次の点は、外部reviewでscopeを判断し、その後の結果前contractで詳細を閉じる必要があります。

1. **Baseline scope**：Morales v3 Table I左列の21-stage unprocessed eighth-orderをfixed対照にする案と、general fourth-order/processed/near-integrable classを今回は探索しない限定が十分か。legacy new-fourthの係数精度問題を黙って除外して優位性を作らない方針でよいか。
2. **Numerical specification**：全feasible曲線の枝・端点・arc-length初期点、係数精度、exact time ledger、fusionの退化case、数値bias guardの実装・予算を結果前に閉じられるか。
3. **Decision rule**：action proxyの5% materiality、ratio guard、±2%重み感度は妥当なpilot用提案か。feasibility/q/Paretoを独立のBF-C routeへ使うなら、判定閾値とtieを事前登録する。登録しないrouteはsecondary診断に留める。
4. **Oracle endpoint**：pilotはI2の有限signal/action proxyに限る。実用I0/I1則、compiled resource、独立validationは結果後の研究方針再評価で必要性を判断する。この限定で仮説を落とすための情報価値があるか。

input/state identity、source/algorithm version、manifest、output path、numerical guardの根拠、implementation照合、実行authorizationはまだありません。本文locator、baseline scope、threshold等に修正が必要なら具体的な最小修正を示してください。問題を避けるためにgeometryやalgorithmを増やすproposalにはしないでください。

## 5. BF-1後の結果分類と停止

BF-1は最良PFを探す研究ではなく、method deltaとdecision relevanceを判別する一回のpilotです。詳細参照は現在のfinite coherent-signal/action proxyであり、compiled-resource referenceと呼びません。

| Case | 支持される限定解釈 | mandatory STOP後の全面再評価 |
|---|---|---|
| BF-A | ordinary/leadingと実質同じdecision。finite-tail固有の機構差が支持されない | B-Fを主研究として停止することを第一候補にする |
| BF-B | 係数/finite burdenの機構差はあるが、materialなresource/登録decision差はない | mechanism noteへ縮小。明確な理論的選択原理が残る場合だけ継続の価値を評価 |
| BF-C | 同じ参照・予算で事前固定したdecision条件と機構差を満たす | partial-randomized-task coefficient designを正式主研究候補として再設計する価値を評価 |
| inconclusive / incomplete | 数値margin、baseline、上限、adapter、未評価等が未解決 | BF-A/B/Cへ無理に割り当てず、不足点を報告して全面再評価 |

**全caseで先にSTOP**します。記録はmandatory_stop_reached=true、next_stage_authorized=false、automatic_next_stage=null。BF-CでもBF-2 actual compile / BF-3 independent validation / oracle-free design ruleを自動開始しません。BF-A/Bでも追加geometry/grid/seedでpositive resultを探さず、B-Sや別co-designへの移行は全面再評価後の別判断です。

## 6. Review回答の形式

次のいずれかを、三文書の版に紐づけて示してください。これは今回のreviewの判定ラベルであり、科学実行のstatusやauthorizationではありません。

| Review decision | 意味 |
|---|---|
| `PASS_FOR_BF1_PREREGISTRATION` | 限定hypothesisを一回判別する価値があり、残る実装/入力/source固定事項を事前登録へ閉じてよい |
| `MINOR_REVISION_BEFORE_BF1` | 中心仮説を保った具体的な最小修正が必要。修正内容と再確認するgateを示す |
| `BLOCK_BF1` | 重複・公平性・oracle・意味論・自由度・判定規則に重大な問題。pilot前に方針修正または停止が必要 |

回答には、判定と理由、blocking/minor finding（file/sectionと最小修正）、baselineとI2 endpointの許容、結果前に固定すべき事項、全結果後mandatory STOPの確認を含めてください。外部review通過、最小修正確認、事前登録/source/authorization固定は別の状態として記録します。

## 7. Evidence / repository境界

M1/M2はTrack Aのsource-bound local evidenceです。H4 1.00 Åは既知development条件、M2 H4 1.30 Åは開封・採点済みで、Bの新held-out/blind/independent replicationではありません。BF-1にAのruntime/cacheを流用せず、Aのresult/source/contract/authorization/artifact/status/pathを変更しません。

prior-art matrix §8は前回BF-0監査のsource観測履歴です。今回のreview準備では、Aの索引冒頭と総覧の最新停止位置だけをread-onlyで再確認しました。観測したA HEADは`d49555f0f1144aa1de6925ace249dc380ae6f4a6`で、Bのbaseを進めていません。A側の別authorization/commit/push指示はBへ継承しません。

| 今回再確認したA text | bytes | SHA-256 |
|---|---:|---|
| `PROJECT_MAP.md` | 42996 | `5a4c25205e36e497d8d2ec24f920b1d2bef3848c4ee71f04d6b09e0a6499a6b7` |
| `docs/research/研究概要・現状.md` | 121890 | `8787744a6f6ea5c887ce6f0ccaae0562ac37b8a67f3a6a665aa7a3f172f70acb` |

review準備の成果はB側Markdown四件（既存三草案の最小修正と本依頼文）に限ります。科学run、NPZ resolve/stat/hash/load、Hamiltonian生成、signal評価、trajectory生成、circuit build/compile、GPU操作、research/full test suiteは行っていません。review準備時点ではstage/commit/pushと外部の相手への送信も行っていません。review対象三文書のhashとlocal link/table等を静的確認しました。

その後の利用者指示により、レビュー資料四件だけをB branchへcommit/pushします。この資料公開のauthorizationはBF-1の科学実行、runner作成、外部review通過を意味しません。公開後も外部review待ちで停止します。
