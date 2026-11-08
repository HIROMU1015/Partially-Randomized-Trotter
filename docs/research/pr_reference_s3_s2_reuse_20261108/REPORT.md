# 新A：S3共役構造の静的資格・保存S2候補の再利用

2026-11-08 JST、SERVER 2。基準 **f0b4b143e9c01115cd010f4dd9f88354ba06c3b4**。Branch `codex/pr-reference-s3-s2-reuse-static-20261108`。

**SOURCE調査、保存metadata照合、保守的な構文照合、人工テスト、次の有限scope設計を完了しmandatory STOP。** 新しい数値誤差、guard、価格最適化、S1/S2再計算は実行していない。

## 1. 固定する結果・科学条件

[訂正S1](../pr_reference_noncentral_s1_coverage_fix_20261108/REPORT.md)のSOURCE限定非中央上界はB′約0.003597257／Reference約0.003595837で保存joint予算を超えた。全catalog版は未資格、SOURCE版のV1非悪化flagは全4行Trueという結果を保持する。V0/V1の負結果も変更しない。

背景はLinear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・準備費除外、Partial S2 LD3 h=.02 finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、compiler A／Qiskit1.3.0。B′4-CX、Reference F/G・actual SOURCE inverse・vacuum P・2 register・CSWAP、既存D融合・basis共有・scalar/event相対位相・X/Yを不変更。M17 q131072/r32/bits88、N_B626／N_R630、alpha1/720、failure.05、phase cap.4、energy1.5936e-4 Haを維持する。

## 2. A：C=A†は何を保証するか

[RTE _append_basis](../../../src/trotterlib/df_rte_qiskit.py)と[partial-S2 _append_block](../../../src/trotterlib/df_partial_s2.py)はSOURCEでは順序反転＋gate.inverseを使う。Compiler Aはその後joint全体をlowerし、[consumer](../../../src/trottertracks/pr_reference_common_word_consumer.py)は各native RZをliteral別に置換する。Negative RZには同じpositive wordのcomplete actual adjointを使うが、**近似basis全体の帰り側をA†として返す仕組みではない**。

この区別を保存templateで照合した。各RZはexact positive sequence hash（Wを含む）とwire／符号の記号にする。隣接する同wordの逆、CX²=I、SX⁴=I、X=SX²という確実な書換えだけで前後の命令積がidentityか調べた。Wordを合成・生成せず、行列、norm、数値phase、guardを評価していない。一般のClifford+T等価判定ではない。

| 対象 | B′ | Reference | 結果 |
|---|---:|---:|---|
| SOURCE RTE 7 basis×保存4 context | 28 pair | 28 pair | 全件NOT_PROVED_BY_LOCAL_REWRITES |
| D0…D3のstandalone basis×保存4 context | 16 pair | 16 pair | 同上 |

計88 pairで、この構文系によるinstruction-product adjoint一致は証明できなかった。**C≠A†の証明ではなく、etaの値も得ていない。** より強いsymbolic恒等式又は数値的なstrict defect評価を排除しない。保存local loweringにはnative global phaseの対応も不足しており、欠けたphaseを0にしてfull basisのstrict etaを認定しない。

D basisの上記照合はstandalone templateの性質であり、fused D／directional lift／actual SOURCE inverse Dをそのまま同じfactorizationに分割できるという新しい証明ではない。現SOURCE inverse-D supplementは保存のまま使い、forward native列のinverseへ置き換えない。

現行構成で使えるのは、通常の加法上界と、etaを未知のまま残す[条件付き共役式](IMPROVEMENT_OPTIONS.md#1-共役式と資格の段階)である。Eta=0や全runへのsmall-time係数を採用していない。

## 3. B：43 frame literalのSOURCE対応

保存[S2 RESULTS](../../../artifacts/pr_reference_noncentral_saved_numeric/2026-11-08/RESULTS.json)の非劣IDをそのまま読み、`fixed_current`が残らない43 literalを照合した。Frontier・dominance・guard/Tを再比較していない。

43/43 literalが、両armの全4保存contextでSOURCE RTE frameのnative RZ位置へ対応した。対応するbasis IDは次の3つである。

- `df-basis-8b103ae5327e20e021dcbb2eb64919b67de85b194d4d6c38b6093414f37e2fbe`
- `df-basis-cb0c4fc6dd74e883da97770cd31022dd8e2e647befe0f44f88d4018e214391e2`
- `df-basis-e6adc65ccc2a1c9d1b35402d8d04be1d1eae7774b30518cf839c36d8c7f4cd8e`

今回照合したD basis／D template／actual SOURCE inverse Dへの43 literalのhitは0。Native位置・wire・signed literal・archive/member・contextを[STATIC_FINDINGS](../../../artifacts/pr_reference_s3_s2_static/2026-11-08/STATIC_FINDINGS.json.xz)に保存した。有限cohortのwhole価格からの推測ではなく、SOURCE ID集合と保存local contextのmetadata対応である。全lawへの移送は既存compiler条件C・SOURCE閉包付きとする。

これにより43候補は、現lawで使わないbasisだけの候補ではなく、実際のSOURCE frameに関係すると確認できた。ただし**S1のmaxを決めるbasis、誤差／費用への寄与、43候補を適用した上界は未評価**。3 basisに対応することだけから全frame上界の改善を保証しない。

各literalの保存非劣IDを固定metadataのbackend recordへ対応させ、strict spec、保存guard method、sequence hash、T／T†／W等の原字段を引用した。新wordは採用せず、C0優先と現85 certificateは不変更。既存43基本wordもそのまま保全する。

## 4. Dの保存frontierと実装上の残項

Dのrole別集合を合わせた164 literalについて、保存frontierを引用した。全て現wordが非劣という既存S2結果を保持する。これは他の精度／T tradeoffがないという意味ではない。例えばD literal `0x1.0009c3e615d69p-8`の保存frontierにはstored T=120と100のpointがある（同一word／同一pointのprovenance重複もある）。今回guardの大小やTを再評価したのではなく、既存frontierの異なる保存字段を確認しただけである。

同じ有限poolで現wordが非劣なら、「strict guardを小さくし、stored Tも増やさない」別pointが現wordを支配することはない。精度を優先してTを増やす案、構造保証で要求精度を緩める案、価格を優先する案を分ける必要がある。D全体の最適word／m17実行可能価格は未取得。

現providerはC0 cacheのexponent=-11、fixed b36／43 basic等をconstructorで固定する。別exponentの既存recordをそのまま差し込めない。Consumerのsequence-only carrierの`-11`は転送用keyであり、backend recordの元specを-11へ偽装する理由にはならない。有限overlayは元record ID／spec／guard／T／Wを保持する別adapterとして資格し、両armに同じliteral共通selectorを与える必要がある。

現行consumerは既存wordのnamed gate列・negative actual adjoint・wire・inplace joinを再利用できる。一方、basis全体をactual adjoint化するには別の構成保持adapterとSOURCE→native／位相／terminal資格が必要。保存wordを再利用できる可能性があり、新規合成が必須とは判定しない。

## 5. 実施・未実行・STOP

[補助SOURCE](../../../scripts/audit_pr_reference_s3_s2_static.py)は保存metadata／命令列の照合だけを行う。[人工test](../../../tests/test_pr_reference_s3_s2_static.py)は5件PASS。Exact word hash／W／wireを区別し、確実な逆以外を相殺せず、missingと不完全な構文判定を未資格に残すことを確認した。人工PASSは科学性能の証拠ではない。

新しいoperator行列・eta・strict guard・合成・word生成／選択・回路build/compile・RPE・T最適化は0。S1/S2・frontier・中央precision探索も再実行していない。SOURCE読取は並行し、主agentのみで調査した。CPU/RAMの独自cap、共有設定・他ユーザーへの変更はない。旧SOURCE／raw／result／grant／ledger／marker／STOPと他Trackを保全した。正式共通指示2026-10-25、適用AGENTS／PROJECT_MAPと研究資料の優先順位に従った。

**Mandatory STOP。次担当はGPT／ユーザー。** [NEXT_VALIDATION_PLAN](NEXT_VALIDATION_PLAN.md)のA1（現wordの2 pairのinstruction-body defect）とB1（保存1 recordの有限interface fixture）は独立に判断できる。いずれも新しい科学実行承認が必要で、今回は未実行。Domain／central／emitter／shell／terminal、全RPE成功、全18round総Tは未資格のまま残す。正式な新規性・保証採択を代行していない。
