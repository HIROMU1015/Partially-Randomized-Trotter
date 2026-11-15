# H4・5方式：候補・固定shot配分の選択と費用比の主張範囲

2026-11-15 JST、Codex B。**保存記録の整理のみ。新しいMSE・価格・G・費用比・ランキングを計算していない。**

[固定ロードマップ][ROAD] §4.4・§5.1–5.2の候補整理を再利用し、配分の選定根拠と比較結果の解釈だけを補う。物理候補の正本は[ALIGNMENT][ALIGN]、評価後の状態は各固定Q3成果を優先する。価格の26軸表、B2の費用集計表は複製しない。

## 1. 評価済み配分と初回比較の選択

以下はすべて m=0..12 順、**N_XとN_Yそれぞれに同じ整数列**を用いたwhole-shot配分である。略号は本資料内だけの表記で、元allocation IDを置換しない。

~~~text
V0    = [64,60,56,52,47,43,38,34,30,25,21,17,12]
VP    = [48,45,42,39,36,33,29,26,23,19,16,13,9]
VD    = [59,55,51,47,43,39,35,31,27,23,19,15,11]
VH1.5 = [96,90,84,78,71,65,57,51,45,38,32,26,18]
VH2   = [128,120,112,104,94,86,76,68,60,50,42,34,24]
~~~

表のIDは元configurationの先頭8桁。全桁・total用ID・round別physical IDはALIGNMENTと各成果のINPUTS／C_RESULT_JOINを正本とする。**適合は採択Q3モデル閾値への適合**であり、物理的十分Nの認定ではない。

| 方式 | 実際の物理候補・配分と保存判定 | 初回費用比較に使う候補・配分／選定理由 | 評価済みの別選択肢・残る余地 | 固定SOURCE・該当箇所 |
|---|---|---|---|---|
| PR-DF | LD1 S2幅0.1/K2（a4dda8f7）、LD2同条件（1fa33556）。元fixed-grid・候補固有κ/Rを保持。各V0で適合 | **LD1・V0**。構造量26軸、P1/P2、較正・独立holdoutなど既存価格証拠の整備が先行する代表 | LD2もV0で適合。同law期待価格によるLD1/LD2比較は未完了。**方式内費用最小とは未確認** | [DF/Full-H結果][DFQ3]「固定モデル・MSE上下界」、[旧配分の出所][V0PLAN]「配分推奨の根拠と限界」、ROAD §5.2、[LD1価格計画][DFPRICE] §1 |
| PR-Pauli | LD1（d905d219）、LD2（d7dd4f52）、各S2幅0.1/K2・元Pauli辞書/κ/R。各VPで適合。別途各V0の旧opt-in診断あり（注1） | **LD2・VP**。LD2配線確認後、小さいtail列の較正計画を先行する記録がある。VPは両LDで採択閾値適合済み | LD1・VPも適合。LD2がLD1より低費用とは未確認。V0診断を初回比較の配分へ戻す必要はない | [Pauli結果][PQ3]「B1：結果」、[配線後の優先順位][PSELECT] §6、[旧V0診断][PCONTROL]「資格・限界とSTOP」 |
| Full PF | S2・16h/7（11cf8ca6）、標準Yoshida4・16h/3（a799697f）、Morales M5・16h（70f7b317）。各VDで適合、Q3は部分保存（注2） | **S2・16h/7／VD**。B2の固定配分総費用がFull PF 3候補内で条件付き最小 | Y4/M5も同じVDで適合・費用集計済み。別幅・別配分における方式最適性は未評価 | [PF6 Q3][PFQ3]「件数・条件付き判定」、[PF配分計画][VDPLAN]「whole-vector案と根拠」、[B2][B2]「主結果」 |
| Truncated PF：B0I p7 | S2・16h/7（a6e55777）、標準Yoshida4・16h/3（defc5e88）、Morales M5・16h（424eecbf）。各VDで適合、Q3は部分保存（注2） | **S2・16h/7／VD**。B2の固定配分総費用がp7 3候補内、かつ決定論PF全6候補内で条件付き最小 | Y4/M5も同じVDで適合・費用集計済み。切断した系の別targetへは変更せず元full DF10 proxyと比較する | PFQ3「件数・条件付き判定」、VDPLAN、B2「主結果」 |
| Full-H pure RTE | one-bodyを含むκ2/K2（54ac3eb3）。V0は**不適合**、VH1.5とVH2は**適合**。R448は別候補で対象外 | **κ2・VH1.5**。評価済み3配分のうち、適合した2配分では全round・両軸のshot数がVH2より少ない | VH2は適合済みの保存参照。未評価vectorの適否・最小N・κ2以外の最適性は不明 | DFQ3、[2倍結果][H2]、[1.5倍HANDOFF][H15]「契約と資格の境界」、ROAD §4.4・§5.2 |

**配分選定の出所**：V0はPR-DF旧partial評価の保存controlを保持して推定量変更を切り分けるための選択で、Full-H固有のnominalではない。VPは既存controlのceil(3N/4)を限定評価したもの。VDはdeterministic B=1に対応する既存paper型scheduleをUSER_SUPPLIED controlとして初めてQ3検査したもの。VH2、VH1.5はユーザーが固定して追加評価した配分である。方式間でNを統一する条件はなく、このN差も資源差の一部である。どれも最小N・最適配分ではない。

注1：PauliのV0はcaf1417の限定opt-inモデル診断。原記録は「目標精度に対する追加filterなし」であり、50be68b8でも再filterしていない。**採択閾値への判定欄は未付与のまま**保持し、本件で新判定を作らない。初回比較には明示的な適合判定が保存されたVPを使う。

注2：PF6の保存component判定は全件適合だが、最終原API return・CPU・peak RAMは親driver不具合で欠測。復元component viewによる部分保存・運用UNKNOWNを引き継ぐ。完全なend-to-end保存PASSとせず、欠測を埋める再評価もしない。

### LD選択の理由として確認できる範囲

- **PR-DF LD1**：ROAD §5.2は「証拠整備では近い」とし、LD2には同条件価格が未同定としている。DFPRICE §1・P2の[固定条件][P2] §1はLD1への既存証拠の連続性を確認できる。一方、**当初LD1をLD2より先に選んだ比較上の根拠や、低費用を示す選定理由は、確認した記録上未確認**。記録の存在から費用優位を補って推測しない。
- **PR-Pauli LD2**：PSELECT §6は、両LDがVPで適合したこと、LD2のW配線依存が解消したこと、較正Rの小ささを**取得規模の理由**として明記する。headは大きく、期待価格・時間・Gが小さいとは未確認とも記す。その後の[LD2 CAL][PCAL] §1は利用者指定のLD2 5 cell取得である。初期A案のLD1先行からの変更はこの後続記録で辿れ、LD2を方式最小と採択した記録ではない。

### 決定論PFの比較対象を明記する

[B2「主結果」][B2]をそのまま引用すると、**Full PF S2は838,693,580 RZ、B0I p7 S2は593,555,352 RZ**（単位：固定shot配分での全13round総compiled RZ数）。前者はFull PF内3候補の条件付き最小、後者はp7内3候補かつ全6候補の条件付き最小である。

したがって「PR対Full PF」は前者、「PR対B0I p7」は後者を比較対象に明記する。「PR対今回の決定論PF最小」は後者と同じ候補を指し、Full PFだけの結果と呼ばない。本件では新たな比・差額を求めていない。

B2は**96直接軸＋60条件付き閉包軸**の集計である。60軸はPF_FIXED_COMPILER_CLOSURE_CONDITIONAL_60_AXES_V1の下で、固定compilerの障壁・反復・終端構造の持続、installed Rustと参照sourceの対応、未調査規模分岐等を仮定する。直接実測156軸や無条件価格保証とはせず、誤差上界・CIはnull、2%保証なしを保持する。

## 2. 費用比に付す条件と、そのまま使える主張文

共通条件はH4/1.00 Å/STO-3G/tol-only DF10、sector(2,2)、保存H/state/snapshot、h=0.1/M12。採択Z1＋T1 total decoder（consecutive_dyadic_nearest_branch_principal_energy_total_v1）、保存DF10 solver energy proxy −2.1663874486347634 Ha、全事象平方loss、十進εE=0.00159360010199040 Haのexact二乗τを維持する。精度側は保存raw・EXACT_AFFINE_OF_BINARY64_RAW・shots/X/Y/round間IID二項モデルに条件付く。PR/Full-Hは各々のideal finite-K law、決定論PFは元deterministic signalを使い、同じlaw名でも候補固有の辞書・normalization・κ/Rを混ぜない。

費用は同一物理候補・round・軸の測定付きfull controlled Hadamard wrapperの平均compiled RZ数、P=0との結合が前提。元receiptの生成IDとtotal decoder用IDを別に保持し、既存のsource-bound bridge/guardでlaw・compiler/lowering・scalar・wrapperを照合する。構造E[S]、有限cohort平均、未資格予測を対象lawのE[C]で代用しない。randomized 3方式の期待価格評価はD2/A/Cに委ねる。N、価格標本数、normalizationを二重乗算しない。

**初回に報告できる文章**

> 対象lawに対応する期待価格が揃った段階で、保存ideal rawとIID二項測定を条件とする共通Z1＋T1/Q3モデル閾値に適合した、今回評価済みの物理候補と各固定shot配分の総compiled RZ数と費用比を報告する。PR-DF LD1・V0、PR-Pauli LD2・VP、Full PF S2・VD、B0I p7 S2・VD、Full-H κ2・VH1.5を区別し、決定論PFの閉包価格条件とQ3部分保存、各期待価格の不確かさ・適用範囲を併記する。資源削減を述べる場合も、この候補・配分での条件付き比較の範囲に限る。

**方式の最適化後まで拡張する際の文章**

> 候補数とshot配分の探索範囲が方式間で異なるため、この費用比だけから、各方式を最適化した場合にもPRが優位とは結論しない。その主張には比較する設計領域を定め、結論を変え得るベースラインの候補・cost-aware whole-shot配分を検証するか、未評価部分が逆転できない根拠が必要である。同じshot数や全方式一律の探索、全候補の再評価を要求するものではない。

同じ精度**基準**への適合と、実際のMSE値の一致は別である。保存solverを厳密groundと認定せず、state/target誤差・実sampler lawへの移送・数値certificateの未資格を保つ。直接riskにAのδEを追加せず、元Minkowskiは別参照。十分N、物理的global RMSE、G_qualified、方式全体のwinnerを新認定しない。

## 3. 初回費用比の後、確認が意味を持つ場面

| 判明した費用関係／求める主張 | 最初に再利用する証拠・必要なら確認する対象 | 今は追加しないこと |
|---|---|---|
| PR-DF LD1と、PR-Pauli LD2／Full PF S2／p7 S2／Full-H κ2の差が判明し、LD選択でPR-DFの位置付けが変わり得る場合 | LD2・V0もQ3適合済み。必要になるのは**LD2固有の同law期待価格**。LD1の価格やE[S]を流用しない | LD2のQ3再実行、LD1/LD2のMSEの僅差から価格順位を推定 |
| PR-Pauli LD2の費用が競合に及ばず、LD1なら比較判断が変わり得る場合、又は方式内最小を主張したい場合 | LD1・VPのQ3適合を再利用し、**LD1固有の期待価格**の必要性を判断。LD2の同law総額と競合の差が判断材料 | LD1の新shot評価や、LD2の凍結係数のLD1への移送を先行 |
| PRが評価済みS2ベースラインより安く、当該固定候補での削減のみを報告する場合 | 上の限定主張には既存の適合・価格証拠を使う。別LDの未評価価格を全部埋める必要はない | 固定候補での優位を示すだけのための方式内網羅探索 |
| PR対Full PFと、PR対p7／決定論PF最小で結論が異なる場合 | B2の2つのS2総額を別々に比較対象として記す。Y4/M5の同じVDでの順位はB2ですでに判明 | 「PF」を一括した比較対象の取り違え、同じ6候補の再集計 |
| 高次PFが別配分ならS2を上回り得るかが、最適化後の優位という主張を左右する場合 | PF6の既存適合・bias/Var診断とB2総額を出発点に、**該当公式の少数whole-vector**を別計画で検討する。まずPRとS2の費用差が判明してから必要性を判断 | MSEの1/N則や単調性だけでshot削減を認定、全6候補の一律再探索 |
| Full-Hとの順位が重要になった場合 | 初回はVH1.5を使う。同一κ2・同じ非負の軸価格ならVH2は全成分のNが大きく、費用を下げる選択肢にはならない。最適Full-Hまで論じる場合のみ、PRとの費用差を見て未評価vector／別κの検討要否を判断 | VH2の再評価・別価格取得、V0不適合から未評価vectorも不適合と推論 |
| 期待価格の区間・適用条件を含めると費用の順序が定まらない場合 | まずD2/A/Cへ、どの候補対の費用区間が重なるかを引き渡し、必要な価格証拠を絞る | 順位不明をNの不足と決めつける、未観測価格の補完、新しい一律許容差 |

これは追加実行の採択ではなく、初回結果を見た後の分岐条件である。既存の固定配分で言える結論を先に利用し、最小N探索を初回比較の前提にしない。

## 4. 照合・保全・終了

指定の固定成果、関連するB計画、B2別branchの保存REPORT、D2の選定計画を読取専用で確認した。ROAD §5.2と同等の候補棚卸しは再利用し、新規framework／候補索引／価格表／集計器は作らない。静的確認は引用値・整数列・SOURCE/path・文書間整合に限り、科学module import、Q3/DP/二項確率/MC、価格・G・比の算術、人工科学testも実行しない。

静的照合PASS：15個の固定Git参照が存在し、5配分の整数列とB2の2金額は元REPORTの文字列と一致。11個のconfiguration先頭IDを元ALIGNMENTへ照合した。Pauli選定理由・旧control未filter・PF部分保存も固定文書と整合する。新しい科学的な再判定は行っていない。

変更対象は本REPORTと[HANDOFF](HANDOFF.md)のみ。ロードマップ本体、原結果、SOURCE、raw、grant/ledger/marker、旧STOP、他担当worktreeは変更しない。commit/push・remote照合後 **mandatory STOP**。

[ROAD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ad6ab45bbcc7139bf499760d28efad794e4eae9b/docs/research/RESEARCH_ROADMAP.md
[ALIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0b4b2d941cd669aaa45231e1236bc2d3158a9d12/docs/research/h4_five_method_candidate_alignment_20261115/ALIGNMENT.json
[DFQ3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a82b0f0ec7f0f11d9981ddb2343fcd5378a87503/docs/research/pr_df_fullh_q3_fixed_results_20261115/REPORT.md
[V0PLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0629a8cf359b3aa99400f79de8f61a5ef50d8162/docs/research/pr_df_fullh_q3_execution_plan_20261115/REPORT.md
[DFPRICE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/950577f416c17a20d309a8053a4bb016027b724d/docs/research/d_pr_df_ld1_ec_plan_20261115/REPORT.md
[P2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39ddd59edf54a3a901e5302207b2765ddbd03fbb/docs/research/d_p2_direct_pilot_20261113/REPORT.md
[PQ3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50be68b803a90fd893a1499ebce74e033225144d/docs/research/pauli_q3_three_quarter_20261115/REPORT.md
[PCONTROL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/caf1417ef20d0a5ab2030a2d957ee4f09443be43/docs/research/pauli_total_q3_accelerated_results_20261114/REPORT.md
[PSELECT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/043c7567451e8f502a1cf0f425d32efe0a50f5dc/docs/research/d_pauli_b1_caller_preparation_20261114/CAL_AFTER_W_PLAN.md
[PCAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5dac74959ed9ea49465d393fc792ef5b4727edba/docs/research/d_pauli_ld2_b1_cal_results_20261115/REPORT.md
[PFQ3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8dba096dc39788b8948af9eafe394f03d386c520/docs/research/pf_shortlist_q3_fixed_results_20261115/REPORT.md
[VDPLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50be68b803a90fd893a1499ebce74e033225144d/docs/research/pauli_q3_three_quarter_20261115/PF_FINITE_PLAN.md
[B2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1002f1dfea34de439ca1b704a0192bca473c5387/docs/research/pf6_allround_cost_20261115/REPORT.md
[H2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/422db93fc7bb599a7ea46c5b58e5dec0fda9ef08/docs/research/full_h_kappa2_scale2_q3_results_20261115/REPORT.md
[H15]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6c5569c3479013b080b2ba53842585f1d670d3c9/docs/research/full_h_kappa2_scale3o2_q3_results_20261115/HANDOFF.md
