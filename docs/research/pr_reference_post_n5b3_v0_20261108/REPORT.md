# 新A V0-A/B：保存word誤差の囲い・中央role予算probe

2026-11-08 JST、SERVER 2。実行SOURCE／scope commit **f8e4544526fa8d531836655ed9222d81ef89bf71**、科学入力base **2886ac07c1ec706a1180d107e0111745e74aec1f**。Branch `codex/pr-reference-post-n5b3-v0-20261108`。

**承認されたV0-A/Bを完了しmandatory STOP。**

- V0-A：保存b36の同じ4 wordについて、Frobenius・spectral下界がX/Y両軸の局所capを超えた。全16比較が `LOCAL_ALLOCATION_EXCLUDED_FOR_THIS_WORD`。
- V0-B：中央roleだけのsafe upperは、B′では保存joint予算を超え、Referenceでは正の余地を残した。
- これは局所割当と中央部分の条件付き結果。**m17のRPE成功／失敗、全support policy、T優越は未資格**。

## 固定条件と実行境界

[NEXT_VALIDATION_PLAN §2](../pr_reference_post_n5b3_policy_20261108/NEXT_VALIDATION_PLAN.md)の固定4角・16局所比較・2 probeだけを実行した。[準備済みSOURCE](../../../scripts/diagnose_pr_reference_post_n5b3_bounds.py)は変更せず、SHA256 `73454a184a734b974ba7d81a4b34cfbf8cdc80ac35b2090f1566ef43c1032ae8` が固定commitと一致した。旧SOURCE／結果／STOPを改変していない。

背景条件はLinear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・準備費除外、Partial S2 LD3 h=.02 finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、Qiskit1.3.0/compiler A。B′4-CX controlled-RZZ、Reference F/G・actual SOURCE inverse・vacuum P・2 register・CSWAP、既存D融合・basis共有・scalar/event位相・X/Yを維持。

対象はm17、q131072／r32／88 bits、B′626／Reference630 shots/axis、alpha1/720、failure.05、phase cap.4、energy1.5936e-4 Ha。新しいb／precision／domain／word policyを選んでいない。

保存[N5-B3 RESULTS](../../../artifacts/pr_reference_n5b2_numeric/2026-11-08/RESULTS.json)のdelta、G、Q+Gと、[M17_INPUTS](../../../artifacts/pr_reference_n5b0_static/2026-11-08/M17_INPUTS.json)のrole上界・予算を再利用した。旧8回のindex検算、guard、word生成やN5-B1 fixtureを再実行していない。

## V0-A：同じwordの誤差囲い

Wを固定b36・同じselected basic・同じwritten orderのlogical product、aをgrid角とする。保存guard和の前提は `||W-RZ(a)||_F <= G`。Delta=|theta-a|に対して

`D_F = 2 sqrt(2) sin(delta/4)`,

`L_F=max(0,D_F_lower-G)`, `U_F=D_F_upper+G`,

`L_2=L_F/sqrt(2)_upper`, `U_2=U_F`

を評価した。最後の行は一般の2×2 norm関係を使い、SU(2)の追加仮定は置かない。Exact W、unitary basic、保存guardの正しさ、complete actual adjointを前提とするlogical誤差である。Emitterの浮動phase誤差とは別で、phase fitをしていない。

Mpmath1.3.0・192-bitの共有sqrt(2)区間と、固定4 delta/4に対するsin区間を用いた。入力Fractionを先にfloatへ丸めず、区間端点は既存`endpoints`で正確な有理数へ変換した。

| Arm | Positive literal | Frobenius下界（概数） | Spectral下界（概数） | 両normの上界（概数） | X/Y比較 |
|---|---|---:|---:|---:|---|
| B′ | 0x1.7e4088e024c5fp-12 | 1.122645e-12 | 7.938301e-13 | 1.438247e-12 | 局所割当排除 |
| B′ | 0x1.fdab627b8d37ap-14 | 3.693575e-12 | 2.611752e-12 | 4.113143e-12 | 局所割当排除 |
| Reference | 0x1.7e4088e024c5fp-11 | 2.416810e-12 | 1.708943e-12 | 2.704974e-12 | 局所割当排除 |
| Reference | 0x1.fdab627b8d37ap-13 | 2.280379e-12 | 1.612471e-12 | 2.685701e-12 | 局所割当排除 |

保存capはB′ X/Yが6.902168834414281e-16／6.902168833913054e-16、Reference X/Yが2.5648979455928107e-15／2.5648979454057876e-15。全比較は表示用小数でなく、正確な有理下界と保存capのbinary64厳密比で判定した。上界cap内・閾値跨ぎの未決は今回なかった。正本は[RESULTS.json](../../../artifacts/pr_reference_post_n5b3_v0/2026-11-08/RESULTS.json)。

**この4角では、同じwordのguard包絡を鋭くするだけで旧局所capへ入れる経路を排除できる。** Negativeは同じcomplete actual adjointなのでnorm囲いも同じ。別word、別digit表現、role別予算、joint誤差の相関を排除していない。

局所capは保存一様会計の十分割当であり必要条件ではない。したがって下界がcapを超えてもm17 RPE失敗とはならない。各局所下界を足してjoint誤差の下界にすることもしていない。

## V0-B：中央roleだけの予算probe

各armで保存rotation上界L_rotと、同armの保存2中央Q+Gのmaxを使い、`E_central_up=L_rot*max(Q+G)` を計算した。新しいword上界の選び直しやkey多重度の最適化はしていない。

| Arm | 保存L_rot | 中央safe upper（概数） | 保存joint十分予算（概数） | 未評価項へ残る差（概数） | Probe判定 |
|---|---:|---:|---:|---:|---|
| B′ | 8388608 | 3.450354e-5 | 9.504670e-6 | -2.499887e-5 | THIS_CENTRAL_UPPER_EXHAUSTS_BUDGET |
| Reference | 4194304 | 1.134548e-5 | 3.517585e-5 | 2.383037e-5 | CENTRAL_SLICE_LEAVES_ROOM |

B′では、この粗い中央certificateだけで保存予算を使い切る。これはsafe upper同士の診断であり、実joint誤差が予算を超えるという下界ではない。Referenceでは、中央上界を入れた後に他項へ正の余地があるが、D／frame／products／scalar／event／CSWAP／wrapper、未資格のemitter／shell処理を0にしていない。

予算は旧d_old・PF控除・保存Nを前提に保存されたjoint値である。今回旧raw diskを再算出／再加算せず、raw geometry、半径、位相、必要shot式も起動していない。Referenceの正の余地から保存630の成功やB′に対する優越を認定しない。

## SOURCE再利用・検算・保全

`track_c_compiler_a_18round.I`と`account_pr_reference_n4c.endpoints`の変更しない関数bodyだけをAST再利用した。科学入力はSOURCE内の2886ac07c…から読み、実行SOURCEはf8e45445…のbytesと一致する。

指定されたsqrt(2)下端の正値、delta範囲、G非負、非負で順序づけられたFrobenius／spectral囲い、4/16/2件数、X/Y rotation上界一致、固定m17入力対応を検算し完了した。実処理は約0.086秒。4つの小さい式なので単一processで完了し、CPU/RAMの独自capや共有設定変更はない。

[SOURCE_INPUTS](../../../artifacts/pr_reference_post_n5b3_v0/2026-11-08/SOURCE_INPUTS.json)、[CHECKS](../../../artifacts/pr_reference_post_n5b3_v0/2026-11-08/CHECKS.json)、[ENVIRONMENT](../../../artifacts/pr_reference_post_n5b3_v0/2026-11-08/ENVIRONMENT.json)、[STOP](../../../artifacts/pr_reference_post_n5b3_v0/2026-11-08/STOP.json)を保存。再現入口は準備済みSOURCEと計画§2の固定venv・fresh出力領域である。STOP後の再実行を指示していない。

新規合成、strict guard、word生成、provider／emitter／consumer、Operator、PF build/compile、simulation／sampling、RPE再計算、T価格集計は0。旧SOURCE／raw／result／grant／ledger／marker／STOPと他Track・他ユーザーの環境は不変更。正式共通指示2026-10-25、親／worktree AGENTS、PROJECT_MAPと既存研究資料を適用。著者Codexの限定数値診断であり、独立科学レビュー・新規性判定ではない。

## 次の研究判断

**Mandatory STOP。次担当はGPT／ユーザー。** この4角で局所capとstrictモデルを維持するなら、同じword以外の構成（record再選択・新dyadic表現等）が必要になる。一方、role配分・support依存のjoint保証を選ぶ余地も残る。どちらを次の有限scopeで扱うか判断する。新規合成が必須という結論や、m17／全18roundのRPE・総T判定は出していない。V1以降を自動実行しない。
