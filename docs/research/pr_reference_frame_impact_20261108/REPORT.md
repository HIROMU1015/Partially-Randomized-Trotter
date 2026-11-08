# 新A B1後：43 frame wordの寄与構造・S1改善検証準備

2026-11-08 JST、SERVER 2。基準 **340670387dbe58bdda5606fdf222408774be1932**。Branch `codex/pr-reference-frame-impact-prep-20261108`。

**静的照合、有限比較SOURCE、人工テストを完了し、科学数値実行前でmandatory STOP。** 43候補はSOURCEの3 basisのforward templateへ対応する。保存inverseには対応せず、S1の最大basis／contextは保存識別子がないため未特定である。今回guard/Tの数値比較・選択・再会計は実行していない。

## 1. 固定条件・既存結果

Linear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・状態準備費除外、Partial S2／LD3／h=.02／finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、compiler A／Qiskit1.3.0を保持する。既存B′／Reference、D融合、basis共有、SOURCE inverse、relative phase、X/Yは不変更。

M17 q131072／r32／bits88、B′626／Reference630 shots/axis、alpha1/720、failure.05、phase cap.4、energy1.5936e-4 Haを保持する。次表は[訂正済みS1](../pr_reference_noncentral_s1_coverage_fix_20261108/REPORT.md)からの引用で、新計算ではない。

| SOURCE限定・X/Y共通 | B′ | Reference |
|---|---:|---:|
| 非中央上界 | 0.003597257009846 | 0.003595837287171 |
| frames | 0.003506023501946 | 0.003506023501946 |
| D | 9.123060223391e-5 | 8.981222210257e-5 |
| 保存joint予算 | 9.504670287679e-6 | 3.517584922187e-5 |

S2の375 literal frontier、43 literalで現guard/T組が支配されたという保存結果、[S3静的照合](../pr_reference_s3_s2_reuse_20261108/REPORT.md)、[B1の保存1 record・4 fixture PASS](../pr_reference_s2_b1_interface_20261108/REPORT.md)を継承し、再実行していない。V0/V1と旧STOPも不変更。

## 2. 43 literalとSOURCE／S1の対応

[静的補助SOURCE](../../../scripts/audit_pr_reference_frame_impact_static.py)は既存STATIC_FINDINGS、S1 scalar結果、RECORD_METADATAを読み、literal／IDのmembership、命令位置の整数個数、保存字段の引用だけを行う。Fractionによるguard/T演算、min/max／argmax、word選択は行わない。43件すべての現word、候補record ID／archive/member／spec／guard／T/T†/W／sequence hash、SOURCE template/context／native position/wire/signは[STATIC_CROSSWALK.json.xz](../../../artifacts/pr_reference_frame_impact/2026-11-08/STATIC_CROSSWALK.json.xz)に保存した。

| SOURCE basis ID | 対応literal数 | 今回対応したtemplate |
|---|---:|---|
| `df-basis-8b103ae5327e20e021dcbb2eb64919b67de85b194d4d6c38b6093414f37e2fbe` | 6 | `basis:<ID>` forward |
| `df-basis-cb0c4fc6dd74e883da97770cd31022dd8e2e647befe0f44f88d4018e214391e2` | 26 | 同上 |
| `df-basis-e6adc65ccc2a1c9d1b35402d8d04be1d1eae7774b30518cf839c36d8c7f4cd8e` | 11 | 同上 |

各literalは1つの上記basisへ対応し、両armの保存全4 contextに現れる。計24 template/context bindingである。Native hitのsigned literalはpositiveであり、保存`basis_inverse:<ID>`へのhitは0。Negative RZを正しくemitできるB1の性質を理由に、別native literalを持つSOURCE inverseも変更対象としない。

保存contextはB′の`[false, false]`、`[false, true]`、`[true, false]`、`[true, true]`、Referenceの`reference_ccx1_rare0_Y0`、`reference_ccx1_rare0_Y1`、`reference_ccx1_rare1_Y0`、`reference_ccx1_rare1_Y1`。D basis／D template／actual SOURCE inverse Dへの43 literal hitは旧S3の照合で0、現metadataのroleも両armともframesだけである。有限保存contextの照合を超える移送は既存compiler条件CとSOURCE閉包を前提とし、新定理とはしない。

変更しないSOURCE basisは次の4 IDで、forward／inverseとも43-key置換の対象外となる。

- `df-basis-150788de4652e5a4b826cabcdf0d508d8057d7088e9c80be3d968e3b2e3127d7`
- `df-basis-52f0155e6e44596da79ae63c8729e4c52881a2de55a93de9203fd279279ee22f`
- `df-basis-62644a8d0e0fcdf41b6beda2d77574185ca174831264dd7b3023a7c01375c92f`
- `df-basis-892597dcece92fa4a9cde6f6ff2afea88f3910d2f5800f2f881dc6684b2a285f`

## 3. S1主要誤差との関係と未判定点

既存S1のtemplate誤差は`max_context Σ_key n(context,key) g(key)`、frame項は

`3 q r × max_SOURCE_basis [ forward_template_max + inverse_template_max ]`

である。`n`は実native RZ出現数、正負は同じpositive keyへ集約される。Forward／inverseのcontext-maxを**別々に**取り、その後basisごとに足す。対応するcontextで同時にmaxを取る別式へ変更しない。SOURCE基底確率や期待run数で全supportの最大値を置き換えない。D／products／CSWAPの既存物理多重度式も変更しない。

保存S1の`template_weights`はscalar上界のみで、最大を与えたcontext／basis IDやkey別weighted寄与を保存していない。旧SOURCEも`max(values)`／`max(pairs)`のscalarを返す。保存scalar同士の新しいargmax比較、key別guard×多重度の評価が必要なため、**43候補が現在のframe最大上界を決めるbasis/contextに寄与するかは未判定**である。3 basisに存在することは改善の証拠ではない。

43候補で変えられるのは3 forward templateのkey guardである。Inverse全7 basis、残る4 forward basis、D／products／CSWAPは現式では変わらない。対象外basisが最大ならframe最大は下がらない可能性があり、対象basisが最大でも未変更inverse項やcontext-maxの切替が残る。これらはSOURCEからの条件整理で、今回数値的な下限・削減率を求めていない。

## 4. 保存recordとB1再利用

43 literalの保存非劣IDだけを候補poolとして引用した。全候補でtarget hex、strict spec、保存guard methodとmetadata型を確認し、guard/Tは再比較していない。Precision labelは-13〜-6の保存8種類であり、新precision追加ではない。

B1の`SavedRecordCertificate`、positive sequence-only carrier、unchanged `lower_native`／consumer、negative complete actual adjoint、W／inplace joinの扱いは再利用できる。ただし`SingleSavedRecordOverlay`のconstructorは-13の単一recordを固定する。全候補をそのまま渡せるとはしない。必要な最小差分は、許された元specの検査と43-key tableのdispatchであり、今回は設計に留めた。元recordをtransport exponent=-11として偽装しない。詳細は[FRAME_IMPACT_OPTIONS](FRAME_IMPACT_OPTIONS.md)。

B1の局所PASSは1 record・4人工fixtureだけの証拠である。43 recordの選択、別W／exponentを含むconsumer経路、full basis inverse、SOURCE phase、全support成功まで無条件には移さない。

## 5. 準備済みSOURCE・人工テスト・未実行事項

[数値診断SOURCE](../../../scripts/diagnose_pr_reference_frame_impact.py)を準備したが、`main`は実行していない。保存frontierに対するG（guard優先）、T（T優先）、NT（現T以下でguard優先）の3有限規則を固定し、両armに同じliteral→recordを適用する比較を設計した。新しいLP／MILP、全体T最適化、provider実行は含まない。

[人工test](../../../tests/test_pr_reference_frame_impact.py)は8件PASS。人工guard/Tだけでtradeoff、保存ID限定、tie、current fallback、欠損非zero扱い、既存S1 weighted関数、forward/inverse別context-max、metadata非数値引用を確認した。旧S1の算術bodyとarm別式をAST再利用し、旧`main`／`evaluate`／frontier関数を実行しない。人工PASSは科学的改善の証拠ではない。

`artifacts/pr_reference_frame_impact/2026-11-08/numeric/`は未作成。保存guard/Tを使う新比較、argmax、43-key適用、S1/S2/B1再実行、合成、strict guard、回路build/compile/simulation、A1 defect、RPE／shot／全18round総Tは全て未実行。静的処理に重い計算はなく、主agentのみで関連ファイルに限定して調査した。資源の独自capや共有環境・他ユーザーへの変更はない。

## 6. 障害・mandatory STOP・次担当

保存S1ではDだけでも両armのjoint予算を超える。仮にframe項を大幅に改善できても、現加法十分証明のD項が残る。これは実誤差下界／RPE失敗の認定ではない。Dのkey/context会計、保存guard/T tradeoff、forward/inverse、eta、basis全体のactual adjoint化は別の課題として残す。S3の条件付き式を新しい全support保証として採択していない。

Domain（B′ scalar域外／Reference vacuum／event到達性）、central、emitter、shell、terminalは未資格・非zeroのまま。新規合成が必須とは判断せず、既存record再選択で足りる範囲と構造保証の課題を分ける。

**Mandatory STOP。次担当はGPT／ユーザー。** [NEXT_VALIDATION_PLAN](NEXT_VALIDATION_PLAN.md)の固定有限診断を実行するか判断し、必要ならその保存値再解析・局所T集計の範囲を承認する。実行時に追加の科学条件選択が不要となるよう、SOURCE、入力、選択規則、比較、出力、STOPを固定した。今回は43-key policyを採択・適用していない。
