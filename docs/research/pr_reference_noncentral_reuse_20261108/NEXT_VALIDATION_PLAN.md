# 非中央role：次の最小保存入力診断と条件付き後続

2026-11-08 JST。科学入力base **800554fbb64c21e584bb29a1a8f52490ab4163e0**。[REPORT](REPORT.md)、[十分式と前提](IMPROVEMENT_OPTIONS.md)。**ここに定める科学数値callは今回未実行。静的調査許可は実行許可ではない。**

## 1. 最初の有限scope：S1/S2

S1は既存policyを固定したweighted-template保証、S2は保存recordの有限guard/T frontierで、入力の一部を共有するが独立に評価できる。準備済み[diagnose_pr_reference_noncentral_saved.py](../../../scripts/diagnose_pr_reference_noncentral_saved.py)は両方をまとめて出力する。小さい有理演算・metadata処理なので準備SOURCEでは単一process。重い計算へ拡張しない。CPU/RAMの独自上限や共有設定変更はない。

### 固定入力・SOURCE

| 固定asset | 再利用する部分 |
|---|---|
| 本公開commitの`artifacts/pr_reference_noncentral_reuse/2026-11-08/RECORD_METADATA.json.xz` | D／framesのexact literal、現word、固定3 archiveの候補record、guard／Tの保存字段。SHA256 **396f11326d2a5bc86316d0997c744a315b60a856ce5912a1fab064676f285027** |
| `artifacts/pr_reference_ct4/2026-11-07/strict_records.tar.xz` | 現C0 299の保存guard。取得・guardを起動しない |
| `artifacts/pr_reference_bprime/2026-11-07/Bprime_local_catalogue.tar.xz` | B′の保存native lowering／post-unroll context |
| `artifacts/pr_reference_transfer_qualification/2026-11-06/local_catalogue_boundary.tar.xz` | request armがreferenceの保存loweringだけ |
| `artifacts/pr_reference_depth_v5_reinforcement/2026-11-07/stage_A_raw.tar.xz` | 実SOURCE inverseのnative lowering |
| `artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE1_STRUCTURE.json` | source_basis_probabilitiesの**keyだけ**をSOURCE RTE alphabetとする。確率値・期待値は使用しない |
| `artifacts/pr_reference_n5b0_static/2026-11-08/M17_INPUTS.json#/condition` | q/r/bits、保存joint予算、N626／630 |
| `artifacts/pr_reference_post_v0_v1_budget/2026-11-08/RESULTS.json` | 固定V1の非中央loadの引用比較。中央b表・V0は再評価しない |
| `src/trottertracks/pr_reference_ct0.py:count_assets,worst_counts` | 保存context-maxと非中央SOURCE多重度の元式。旧mainは呼ばない |
| 本公開commitの`audit_pr_reference_noncentral_reuse.weighted_template_max` | 人工test済みのkey重み付きcontext-max。Metadata audit mainは再実行不要 |

本公開commitの2 helperとmetadataを保持し、その他の科学入力は上記baseから読む。Metadataファイルをmaterializeした専用worktreeで実行する。新しいSOURCE／archiveを自動探索せず、SOURCE hashと固定入力来歴を保存する。承認原文・grant等の証明は作らない。

背景条件はLinear H4 1Å/STO-3G/DF12、元H厳密ground供給・準備費除外、Partial S2 LD3 h=.02/K2、ordinary diag(I,V)、canonical independent finite-bit law、compiler A/Qiskit1.3.0、既存B′／Reference／D融合／basis共有／scalar/event/X/Y。M17 q131072/r32/bits88、N626/630、alpha1/720、failure.05、phase cap.4、energy1.5936e-4 Haを不変更とする。

算術はPython integer／Fraction。保存float literalは元binary64へ復元して同じhex keyへ対応させ、cap／budgetは厳密有理比として扱う。新しい三角関数・guard評価・precision候補は不要。

### S1：Current policyの非中央weighted-template上界

1. C0の保存strict guard、現85 compositeの保存Q＋Gを各literalへ対応させる。旧emitter項はここで足さず、別の未資格項として保持する。Record選択を変更しない。
2. 保存catalogから`basis:*`／`basis_inverse:*`、B′ left_all/right_all/left_no0/right_no0/boundary_control、Reference block0 half/full・block1…3 half、actual inverse同template、cz／z／cswapだけを読む。Central family、symbolic P、H/Sdg wrapperは本scopeの数値和へ入れない。
3. 各template・保存contextのactual RZ keyを数え、key別guardの加重和を取った後にcontext-max化する。新回路・新loweringは生成しない。Missing literalは未資格とし0補完しない。
4. 元worst_countsのD／frames／products／CSWAPの式を評価する。`scalar_and_wrapper`の`+1`はRZ個数であってnormでないため、元関数全体に誤差値を代入しない。
5. Frame maxには(a)元catalog全basis、(b)保存SOURCE registryの2通りを与える。その他のtemplate／係数／wordは同じ。**2 variants×2 arms×X/Y＝8 aggregate行**を出し、保存V1非中央loadと比較する。元のscalar／中央／event／emitter／shellは別のunknownとして残す。
6. 保存joint予算から非中央上界を引いた残余を出す。正でも全体PASSではなく、負でも実誤差の下界ではない。S1のみでRPE式・整数Nを再計算しない。

### S2：同literalの有限guard/T frontier

1. 固定metadataのD／frames各literalについて、現wordとそのexact-literal recordだけを対象にする。新archive・literal・precision候補を追加しない。
2. PASS・strict spec・guard method文書付き候補の保存guardをFractionとして読み、保存Tを離散整数として扱う。数値的な比較は今回の静的metadata照合から進む**新しい保存値再解析**であり、明示承認の対象である。
3. `g_a<=g_b`かつ`T_a<=T_b`、少なくとも一方がstrictならaがbを支配するという有限比較で非劣record IDを残す。同一sequenceの複数certificate／provenanceも保持し、同じkeyだけを理由に消さない。
4. Policyを選ばず、C0優先を変更せず、S2結果をS1へ自動適用しない。総Tやexpected T、price envelope、LP／MILP最適化を行わない。候補が減っても世界的最適性・RPE可否を認定しない。

### 検算・実行入口・STOP

実行承認後だけの入口：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/diagnose_pr_reference_noncentral_saved.py
```

検算は固定metadata hash、SOURCE alphabetのcatalog包含、m17／N対応、全physical countの非負、guard coverage／missing、8行の件数と、SOURCE alphabet制限後の上界が全basis版以下であること。保存V1との非悪化flagも出し、偽ならcoverage／移送前提の不一致を疑って結果を未資格とする。Zero／missingを埋めてPASSにしない。

Fresh `artifacts/pr_reference_noncentral_saved_numeric/2026-11-08/` にRESULTS／SOURCE_INPUTS／STOPを保存する。完了、固定SOURCE／入力不一致、coverage不足、数値前提の不整合でmandatory STOP。完成した有限結果を公開し、record再選択や次段階を自動起動しない。

**S1/S2に含めないcall：** 新strict guard、Clifford+T合成、word生成、provider／consumer実行・変更、circuit build/compile、simulation／sampling、中央precision探索、domain拡張、RPE／shot再計算、総T／T-depth／最適化。旧研究のSTOPを解除しない。

## 2. 条件付き後続：S3共役pairの資格

S1/S2で保証がなお不足し、又は共役構造の利用価値を別に調べるとユーザーが判断した場合の候補である。現時点でS3科学実行は固定・承認していない。

最小の入口は、同じ保存basis／basis_inverse template、同じ現word、同じwireとnamed gate順で、`C=A†`が厳密なtoken恒等式から示せるか調べること。単なるinverse loweringの非一致はoperator不一致の証明ではない。必要なら有限local Gaussian familyからetaを別guardで囲うか、actual adjointを返す別consumerを資格するかを選ぶ。対象template／contextとnorm・precisionを先に固定し、guard・build/compileの承認なしに実行しない。

`eps_D + min(eps_f+eps_i, min(2,|t|W)eps_f+eta)`の静的証明だけでは、eta／W／epsのm17値・全context・terminal資格は得られない。特にproductsを含むRTE runへsmall-time weightを無条件に与えない。既存C2／fusionを再開する指示にはしない。

## 3. RPE・総費用へ戻す条件と担当

S1が予算内でも未評価role／domainを閉じる必要がある。S2に有利なrecordがあっても、literal共通の有限overlay、actual adjoint、strict W／relative phase、SOURCE→native C、consumer／terminalの資格が別途必要。新規合成が不可避と判断するには、採用する保証・予算・有限policyの不足を特定する。Archive非coverageだけで全面再合成を要求しない。

完成したlogical joint保証を得た後に、旧diskと新synthesis diskを同じraw単位で一度だけ加え、保存N626／630を保持するm17 X/Y十分条件を検証する。全18roundは各保存N・PF・r・bitsの行をそれぞれ資格し、両armへ同じ改善機会を与えた実行可能な総Tで比較する。有限Pareto表、LP下界、共通frameの価格取消しだけで強B′への優越を認定しない。

**現在は静的調査完了のmandatory STOP。次担当：GPT／ユーザーがS1/S2の限定数値診断を実行承認するか、S3の前提調査を優先するか、保留するか判断する。Codexはその判断前に科学callを開始しない。**
