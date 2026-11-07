# 新A CT-5B：全support構造と必要価格境界の限定調査

2026-11-07 JST、SERVER 2。固定参照 **`e3e852a35f24839b42a6d282083583bb611b6a9c`**、専用branch `codex/pr-reference-ct5b-20261107`。**静的範囲のStage A〜Dを完了、mandatory STOP。** 新科学計算の承認を求めることなく、許可済みの独立した静的項目をすべて進めた。

## 結論

固定SOURCEからD構造のevent非依存性とRTE frame pairの共通multisetを導ける。保存4 contextずつのliteral loweringはD/frameでcontext間一致し、frameはB′/Reference間でも一致した。whole-native価格への移送は、固定compiler/contextと採用recordの条件付きで資格する。

その条件の下で、共通frameの**価格記号の係数差は0**となる。中央回転は既存4 positive keyの価格で閉じ、productとwrapperは保存strict T=0 recordによって価格差を除ける。Referenceのproduct-Z消去でRZ数が変わっても、そのT加重残差だけは0にできる。

一方、共有frameにも未取得の保存keyがあり、B′ event位相supersetのb2（binaryの逐次2pi加算）もrecordがない。さらにvacuum全supportの取得/停止・価格境界は未資格のまま。frame価格の取消しやB′ event価格の非負性は、**価格差の証明に不要な数値を減らすが、実wordの取得・誤差資格を省略しない**。

全support T優越・canonical CT RPE成功・全18round総Tは認定しない。CT-3B/SKの実装が次の既定作業とも判断しない。

## 固定条件と参照

Linear H4・1.00 Å・STO-3G・DF12、元Hの厳密groundを各shotで供給し準備費除外。Partial S2/LD3/h=.02/finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、q128/r1、strict epsilon1e-11。Qiskit1.3.0/compiler A/rz,sx,x,cx、B′4-CX controlled-RZZ、Reference F/G・actual SOURCE inverse・vacuum P・2 register・8-CX CSWAP、D融合/basis共有/scalar/event/X/Yを維持。全失敗率.05、phase cap .4、保存199 shot/axisと旧native誤差/RPE条件を変更しない。

| 固定資料 | 参照した証拠 |
|---|---|
| [CT-5A REPORT](../pr_reference_ct5a_static_20261107/REPORT.md)、[CONDITIONAL_BOUNDS](../pr_reference_ct5a_static_20261107/CONDITIONAL_BOUNDS.md)、[STOP](../../../artifacts/pr_reference_ct5a_static/2026-11-07/STOP.json) | e3e852a…の条件式と未資格事項。旧文書は変更しない |
| [CT-4B REPORT](../pr_reference_ct4b_20261107/REPORT.md) | 8012c3e9547690f0d722edac9d0b9a9391ba2f8e。ROLE_T_COUNTS、PILOT_T_COUNTS、KEY_REUSE_AND_SYNTHESISの保存行を照合 |
| [CT-4A PREFLIGHT](../pr_reference_ct4a_20261107/PREFLIGHT.md)、[scope](../pr_reference_ct4a_20261107/SCOPE_PROPOSAL.md) | 対象/精度/科学call境界と固定入力 |
| [V3](../pr_reference_rotation_v3_20261107/REPORT.md) | 224440f740051ed5794e8db87a57908fc02dd02f。SOURCE、保存ANGLE_KEY_MULTIPLICITIES、LOCAL_INVENTORYとcohort/contextの適用範囲 |
| [CT-0](../pr_reference_ct0_20261107/REPORT.md) | f5f16fad64748236b0160284dc6c3017974b0132。KEY_COVERAGE、PHASE_SUPPORT、m7 precision/headroom |
| [CT-1](../pr_reference_ct1_20261107/REPORT.md) | 6ac8e1aa3104d8563188612410c1a7533ec47072。strict/emitter/actual adjointとfinite検証 |
| [CT-2](../pr_reference_ct2_20261107/REPORT.md)、[CT-3A](../pr_reference_ct3a_20261107/REPORT.md) | 6b0b6669…／c21a43f4…。全入力取得/停止/価格と有限成功の区別 |
| [V5](../pr_reference_depth_v5_20261107/REPORT.md)、[V5補強](../pr_reference_depth_v5_reinforcement_20261107/REPORT.md) | 固定depth/fence、a095a225…の実SOURCE inverse/4 context、旧STOP |
| [B′構造補遺](../pr_reference_bprime_20261107/STRUCTURAL_APPENDIX.md) | 固定pass/contextとSOURCE境界。CX/depthの保証をTへ自動移送しない |

共通運用指示2026-10-25、親/worktree AGENTS.md、PROJECT_MAPと既存研究概要を適用。一次Qiskit1.3.0のbasis_search/compose_transforms/mod/inverse_cancellationも該当箇所を読取確認した。SOURCEをimport/実行していない。

## Stage A〜D

| Stage | 今回の到達点 | 資格 |
|---|---|---|
| A：D | qと固定D primitiveでSOURCE構造・多重度が決まる。初段はRZ/XXPlusYYを消さず、最終passはRZを消さない。保存Dおよび実inverse Dのcontext配列一致を確認 | SOURCE構造は`PROVEN_FROM_FIXED_SOURCE`。whole-native価格は`CONDITIONAL_ON_COMPILER_CONTEXT` |
| B：frame | paired application列は同じbasis run pairを持ち、F†でもpair multisetを保つ。Cの下でnative key差が0 | SOURCE multisetは`PROVEN_FROM_FIXED_SOURCE`。実価格共通性はC/全必要wordの定義が条件 |
| C：有限key | 中央4 keyに同一eの保存価格あり。Z/CZ/wrapperは保存T=0。event supersetのb2と共通frameの一部は欠損 | 価格を引用できる範囲と未取得/到達未資格を分離 |
| D：必要境界 | 固定項＋中央差＋非負B′ event−vacuumへ条件式を縮約。pointwiseと期待条件を別記 | 数値閾値/極値/期待値は未評価、全support優越は`UNQUALIFIED` |

詳細は[STRUCTURAL_CERTIFICATES](STRUCTURAL_CERTIFICATES.md)、[FINITE_KEY_AND_UNKNOWN_PRICE](FINITE_KEY_AND_UNKNOWN_PRICE.md)、[CONDITIONAL_T_ADVANTAGE](CONDITIONAL_T_ADVANTAGE.md)。

固定SOURCEの構造だけでwhole価格を認定せず、局所templateのcontextとwordの選択を条件に残した。新しい科学的仮定を追加して条件をPASSにしたのではない。これは著者の技術的適合性判断で、独立詳細レビュー/新補題の正式採択を代行しない。

## 価格差を閉じるために残るもの

共通partial cacheを全必要keyへ完了した仮想policy Pと、compiler条件Cの下では

\[
\Delta T_x(\omega)=K+\sum_{i=1}^{q}d_{o_i}+E_B(\omega)-v_P(\phi_\omega),
\quad K=D_B^\star-D_R^\star-C_{sw}^\star+S_B^\star.
\]

frame差、product差、wrapper差、T加重取消し残差を条件付きで除いた。Kやd_oの値は新しく算出していない。pointwiseには同じωを保つ結合差の上界、期待優越にはその期待境界を使える。一様vacuum T capは十分な選択肢の一つであり、唯一の必要条件ではない。

shared frameの数値価格は差から消せても、全supportのframe word取得は必要。B′ event b2の価格は非負なので下界から落とせても、到達するならwordが必要。vacuumは取得/停止/strict保証と、差分を十分に囲う価格境界が残る。保存角度域や2つの有限価格で埋めない。

CT-4Bのnonrare1.865933%／rare1.881674%削減は、保存q128/r1・2 trajectory・採用wordの`SAVED_COHORT_ONLY`結果として維持する。今回、改善率や全support閾値を新計算していない。追加8qubit・旧native tradeoff、全18round/T-depth未資格も維持する。

## 実施範囲・STOP・次担当

実施したのは固定文書/SOURCEの読取、既存JSON/圧縮archiveのmetadata・保存lowering配列/key/価格行の照合、一次SOURCEの限定確認、記号的導出、5文書と最小索引/停止記録の作成。新規科学SOURCE/runnerは実装していない。

**新規合成、word価格計算、guard再評価、保存値の新しい価格集計/極値/期待値、回路build/compile、simulation/sampling、precision/shot/alpha最適化、全18round/T-depth計算はすべて0。** 既存SOURCE/raw/result/grant/ledger/marker/STOPと旧manifestを変更していない。共有環境/他ユーザー/別Trackも操作していない。

本質的SOURCE不整合は見つかっていない。未資格項を理由に他の静的Stageを途中で止めず、指定範囲を完了した。残る作業は新規数値再解析/合成、又は重要な科学的採択を要するため、**mandatory STOP**。

次担当は**GPT／ユーザー**。[NEXT_SCOPE](NEXT_SCOPE.md)の最小候補は、まず保存価格から必要vacuum境界を定量化する限定数値再解析（別途実行承認が必要）、又は共有finite cache/到達性を埋める限定scopeの選択。科学レビュー・実装・SK・全support/18round計算へ自動進行しない。
