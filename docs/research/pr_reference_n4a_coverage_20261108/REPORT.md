# 新A N4-A：全RZ roleの静的coverage・共通policy接続準備

2026-11-08 JST、SERVER 2。固定参照 **002ba1141fcdbfb9bd59b8173fb594da9a89388b**、branch `codex/pr-reference-n4a-coverage-20261108`。

**静的監査を完了し、mandatory STOP。** 保存m7のnumeric key supersetでは、C0未収録は**共通RTE frameの84 positive literal keyとB′ event b2の1 key**。85 keyはすべて既存43-word辞書の入力域内だった。追加基本word合成を要する根拠は見つからず、次は既存providerのrole接続・局所資格が中心となる。

判定は **SOURCE_ROLE_COVERAGE_IDENTIFIED／C0_PRIORITY_PRESERVED／COMMON_POLICY_STATICALLY_COMPATIBLE**、ただしcompiler条件Cとrole consumerの資格が条件。**実共通policy・全回路correctness・全support CT RPEは未資格**。85は新しい到達集合や全vacuum keyの件数ではない。

## 固定条件・今回の実行境界

Linear H4・1.00 Å・STO-3G・DF12、元H厳密ground供給・準備費除外。Partial S2／LD3／h=.02／finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、q128/r1/m7・保存73 bits。Qiskit1.3.0/compiler A、rz/sx/x/cx・全結合、最終strict Frobenius10^-11を維持。

B′4-CX controlled-RZZ、Reference F/G・actual SOURCE inverse・vacuum P・2 register・8-CX CSWAP、D融合・basis共有・scalar/event・X/Yは不変更。保存199 shots/axis、energy1.5936e-4 Ha、failure.05、phase.4、alpha1/720も変更していない。辞書b36／j=-36…6／43 word／内部10^-13、exact W、全体actual adjoint、固定runtime／inplace経路を維持する。

今回実行したのはSOURCE読取、保存JSON／archiveのschema・hash照合、literal keyの正規化／域照合、mock metadataテストだけ。**provider生成・provide/emit呼出・新role certificate・guard・合成・回路build/compile・科学数値再解析・価格再集計・RPE再会計は0**。

## Role別の結論

詳細は[ROLE_COVERAGE](ROLE_COVERAGE.md)、正確なkey表は[ROLE_COVERAGE.json](../../../artifacts/pr_reference_n4a_coverage/2026-11-08/ROLE_COVERAGE.json)。

| Role | 保存numeric集合／既存証拠 | 静的な再利用候補 |
|---|---|---|
| D deterministic・境界・実SOURCE inverse | 対応する保存keyはC0内 | C0_FIXED_WORD。Dの全support移送はC条件付き |
| 共通RTE basis frame | 両armで同じ216 positive key、132 C0／84 miss | 84全てDICTIONARY_DOMAIN_COVERED、role接続は未実行 |
| RTE中央 | armごと2 positive key、合わせて既知4 key | C0_FIXED_WORD、half/fullと符号を保持 |
| Product Z/CZ | 保存固定templateのkeyはC0内 | C0_FIXED_WORD。価格0をgate／誤差0としない |
| B′ event | r1保存supersetの3非zero key中2 C0、b2 miss | b2は辞書域内。canonical到達性は今回未確定 |
| Reference vacuum | 保存SOURCE recurrenceの域が辞書のLと一致 | 既存N3-B1b-1の局所provider資格を再利用。exact reachable keys未列挙 |
| Scalar・wrapper・CSWAP | 保存keyはC0内 | C0_FIXED_WORD。B′にはCSWAPなし、Ref event符号はvacuumへ入る |

Dのsigned境界補正とinverse supplementは互いに重なる台帳で、表のkey数を物理gate数として加算していない。全supportの意味は、有限SOURCE alphabetと**compiler条件C下のtemplate閉包**に依存する。2保存trajectoryを全supportの観測と扱っていない。

### 既存N2欠損

- frame `0x1.08ba32d420f2cp+2`：C0なし、辞書域内。両armの保存inverse-basis templateにもこのkeyがある。
- B′ b2 `0x1.921fb54442d18p+2`：C0なし、辞書域内。binary b2を0や理想2piへ置換しない。

b2はr1/K2・identity抽出済みtailの「order2＋負productがちょうど1個」で生成し得るSOURCE分岐。canonicalでその分岐に非zero finite-bit cellがあるかは今回確認していないため、到達性だけはSOURCE_SUPPORT_UNKNOWNとした。安全な3-key supersetが既存辞書域に収まるので、この未解決を理由に新規基本合成やsamplingを要求しない。

## Compiler条件C・固定価格

[CT-5B STRUCTURAL_CERTIFICATES](../pr_reference_ct5b_20261107/STRUCTURAL_CERTIFICATES.md)の完了済み監査を再利用した。固定1.3.0、同値則／post-unroll gate族、実binary引数とSOURCE inverse、Dのevent非依存性、frame pairのSOURCE multiset、RZに作用しない最終CX/X cancellationが前提。新gate族・pass・同値則変更を発見していないが、任意context不変性を新たに認定したのではない。

代表frameとsymbolic Pの保存contextを次の局所検証対象として特定しただけで、全4 contextのcompiler監査を全面再実行していない。[INTEGRATION_TARGETS](../../../artifacts/pr_reference_n4a_coverage/2026-11-08/INTEGRATION_TARGETS.json)にarchive member／record名を固定した。

固定C0の299採用sequenceと43基本wordのhashは一致した。N1保存値 **K=160222、d0=106、d2=114、B_P=6072、6072<173790** は引用であり、新しいT差・極値・期待値ではない。C0を置換しないのでD／中央／scalar等の価格anchorは維持可能。新frameの実価格を旧C0価格と同一視せず、Cと共通policyの下のpaired key multiset一致を用いて価格差を記号的に取消す。**frameの合成誤差は取消さない。**

## 接続で残る具体的な不足

既存providerはrole引数に依存せず、同一positive literal keyに同じpositive sequenceを選び、負側には全体actual adjointを対応させられる。しかし、今回その85 keyでproviderを実行していない。

既存lower_nativeはrz/sx/x/cxだけを受け、reset／measure／barrierやclassical bitsを保持するwhole wrapperではない。共通policyのlookupとは別に、元のqubit/clbit・reset・readout・global phaseを保持する最小consumer adapterが必要。prefixのphase初期設定、inplace compose、inverseの実経路をN3の上界へ対応させ、別位置の誤差や初期phase増分を消失／二重計上させない。

保存local loweringにglobal phaseがない場合、0を補ってSOURCE unitaryを再構成してはならない。局所native gate productの置換試験とSOURCE→native検証を区別し、全jointの実global phaseは元native metadataから保持する。詳細は[COMMON_POLICY_INTEGRATION](COMMON_POLICY_INTEGRATION.md)。

## 次に必要な最小作業・STOP

[NEXT_EXECUTION_SCOPE](NEXT_EXECUTION_SCOPE.md)に、既存providerを使う85 key、代表basis／P template、consumer、全位置誤差と199-shot条件への接続を提案した。これは**未実行・別途明示承認が必要**。新規Clifford+T合成、precision変更、N2取得runner再開を既定作業にしない。

[metadata helper](../../../scripts/audit_pr_reference_n4a_coverage.py)は標準ライブラリだけを使い、CT-0 positive関数のみAST再利用。Qiskit／provider／科学moduleはimportしていない。[CHECKS](../../../artifacts/pr_reference_n4a_coverage/2026-11-08/CHECKS.json)と[SOURCE参照](../../../artifacts/pr_reference_n4a_coverage/2026-11-08/SOURCE_REFERENCES.json)に保存。小さい読取処理なので1 process、資源cap／共有設定変更なし。著者Codexの技術監査であり独立科学レビューではない。

正式共通運用指示2026-10-25、親／worktree AGENTS.md、PROJECT_MAPと固定研究資料を適用し、旧SOURCE／raw／result／grant／ledger／marker／STOP、他Track／他ユーザーを変更していない。

静的scope完了、残る科学実行へは進まず[mandatory STOP](../../../artifacts/pr_reference_n4a_coverage/2026-11-08/STOP.json)。**次担当はGPT／ユーザー**：既存辞書を再利用する最小role接続・correctness／誤差資格のscopeを判断・承認する。N4-B／RPE再会計を自動開始しない。
