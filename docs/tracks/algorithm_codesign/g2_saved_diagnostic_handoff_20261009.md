# Track B G2：保存表・構造・強い対照の診断結果

2026-10-09。**G2_BOUNDED_DIAGNOSTIC_COMPLETED_WITH_DECLARED_LIMITS**。
GPT [G1レビュー§13](../../research/track_b_G1_scientific_review_20261009.md)と利用者の「進めてください」に基づく、
post-hoc/developmentの技術診断。数式の独立監査と、固定保存表504 profileの診断を完了した。
**研究価値・次の実装範囲・新規性はGPT G2へ戻す。mandatory STOP。**

branch=`track-b-g2-saved-diagnostic-20261009`、基点=`4f2a08c80a293513761508421a76be2005d4c9ec`。
実装は独立した[診断script](../../../scripts/tracks/algorithm_codesign/g2_saved_diagnostic.py)。
旧science source・contract・authorization・result・consumed marker・STOPは変更していない。
shared API、Track A、rootの未commit資料も変更していない。

## 1. 対象・会計・完全性の範囲

既存R1の **2-qubit distinct-basis controlled finite P3、p=(3/4,1/4)、x={1/8,1/4}、σ=+1**。
σ=−1は保存値のcost/error同一性照合であり、独立再現ではない。
分子geometry/basis/DF rank/split L_D/PF delta窓は適用外。
7 prototypes O0/O2/P2/P3/A0/A1/A2、既存precision 1e-3/1e-4/1e-6を固定。
21 columns/xの[保存表](../../../artifacts/track_b_ra_d0_preparation/2026-10-06/candidate_table_v1.json)
SHA256=`2f86169fc301ffa6b73f3cde7c8bd49939a4e7742d422c309bac181e5f09d0b6`。
参照commitは`0ddf67756516e08f85fed1b987459a5e862676b7`。
原R1 result SHA256=`f726ad70cb2643533f0d037b518cde1b702724adb4e6571fea25e26e4bfdd61e`。

ordinary/PTSC-K0/A各precision profileは計63/x。
J1/J2/J3を含むideal 6頂点は計252/x、総504で追加候補なし。
対照にも同じprecision選択とevent-level既知IS自由度を与えた。
保存済みphase・event probability・IR identity・cost・strict error上界を照合し、
operator errorそのものの新規再計算は行わない。

各resource axisの目的は、K=Σevent c√C、s=1/200−Σevent c·2δとして **(K/s)²**。
これは固定dictionary/ideal degree equality/affine saved bias上界/他資源capなしの診断モデル。
算術は10^-60格子への外向き有理interval、整数sqrt、Decimal lnの隣接値を用いる。
minimaと比の厳密intervalをJSONに保存した。表示の丸め値は認証に使わない。

[独立数学監査](g2_independent_math_audit_20261009.md)で、return恒等式・閾値、固定線形価格、IS/線形分数化、
pure-profile補題を一般に再導出した。記載前提内はPASS。
252/xの完全性はこのideal目的に限る。元RA-D0 numeric K3、sampler丸め、固定shots、複数caps、
rangeを含むproposal最適性、arbitrary signed/新dictionaryの完全性ではない。
canonical net-costのprofile最小もcanonical混合全体の最小とは呼ばない。

## 2. 同じIS自由度の後に残った差

下表は、元3頂点のideal目的最小に対する6頂点側最小の比。
いずれも保存された保守的biasモデルの値で、旧U₃<L₂ witnessや全物理実装のlower boundではない。

| x | axis | 元3頂点の最良 | 6頂点の最良 | 6/3の比 | 追加自由度の差 | 達成性 |
|---|---|---|---|---:|---:|---|
| 1/8 | T | A | J1 | 0.9961846242 | 約0.3815% | J1はzero-T eventを含む未達成infimum |
| 1/8 | CX | PTSC-K0 | PTSC-K0 | 1.0000000000 | 追加差なし | zero-CX eventのfinite proposal未取得 |
| 1/8 | 1Q | A | J1 | 0.9964092198 | 約0.3591% | ideal real proposalで達成、sampler未構築 |
| 1/4 | T | A | J1 | 0.9847924627 | 約1.5208% | J1はzero-T eventを含む未達成infimum |
| 1/4 | CX | PTSC-K0 | PTSC-K0 | 1.0000000000 | 追加差なし | zero-CX eventのfinite proposal未取得 |
| 1/4 | 1Q | A | J1 | 0.9858219778 | 約1.4178% | ideal real proposalで達成、sampler未構築 |

T/1Qのstrict interval差はあるが、過去の5%/10%を使った科学GO分類は付けていない。
目的間でproposal・precision・winnerが違うため、単一の実装が全資源で勝ったとは言えない。
他資源も各proposalのexpected vectorとして全rowsへ保存した。

Tのprecisionは、x=1/8のAが全group 1e-4、J1がP2/P3 1e-3・A0/A1 1e-4。
x=1/4のA/J1は全active group 1e-3。
1Qはx=1/8のA/J1が全active group 1e-4、x=1/4のJ1だけP3 1e-4、残り1e-3。
CXは両xでPTSC-K0の全active group 1e-6を代表として保存（同値profileの可能性あり）。
J2/J3はこの3axisの最良にならなかったが、別条件全体で不要という証明ではない。

ISを元構成へ戻す効果は、保存profile subsetのcanonical最小との記述比較で
Tが約2.93%/2.83%、1Qが約2.90%/2.80%、CXが約7.17%/8.51%。
CXと一部Tはinfimum比較である。このcanonical subsetとの比は、canonical混合全体の最適値との比ではない。
precision自由化の影響もuniform precision対照と別保存した。
例えばx=1/8のTでJ1のuniform 1e-4とmixed精度との差は小さく、
J1-vs-Aの差をprecision選択だけの差と混同しない。

## 3. priceによる説明とbiasの役割

同precision固定のIS価格ℓ_g=norm_g·E√Cではβ>0、ζ>0を全18条件で確認。
αはx=1/8のT/1Q・1e-4、x=1/4のT/1Q・1e-3で正となり、J1の固定価格が最小。
それ以外はα<0でPTSC-K0が最小。
J2の増分μζ>0、J3の増分α+βはこの表で最良を作らない。
これは7-prototype同precisionでの線形価格であり、最終bias分母を含む判定ではない。
粗い合成ほど安いがsを削るため、biasを戻した結果、T/1QではA/J1が選ばれる。
価格表はexpected-C affine priceも別modeで保存し、E√Cとsqrt(EC)を区別する。

## 4. finite-confidenceで未確定なもの

旧R1からεaxis=1/200、αaxis=1/5280、cap=10⁹/axisを固定して、
Bernstein sufficient shotsの算術予測を保存した。
canonical 504/504 profileはこの予測でbias・shot capを通過。
uniform元構成18件のshot数は既存R1保存値と全件一致した。
ISでは各xでTの216/252、CXの135/252 profileがzero-costを含む。
そのfinite proposalは **MISSING_FINITE_PROPOSAL_ZERO_COST** とし、勝手なepsilon costを足していない。
残りはconditional analytic pass。1Qは全504でpositive-cost proposalを評価できた。

1Q最小profileのnative-1Q目的用ISをそのまま使い、Re/Imのreadout計5 gateを戻すと：

| x | 構成 | shots/axis | two-axis 1Q予測 |
|---|---|---:|---:|
| 1/8 | A | 811,506 | 465,382,325.19 |
| 1/8 | J1 | 831,338 | 463,413,930.45 |
| 1/4 | A | 1,022,850 | 468,173,605.23 |
| 1/4 | J1 | 1,115,481 | 461,948,987.57 |

これはideal実数weight/lawを条件とした予測。shotsを実行しておらず、lawのdyadic実装・mean certificateもない。
native 1Q用proposalはreadout込みのfinite-confidence最適proposalとは限らない。
全504比較に対する新しいfamilywise実測保証は主張しない。
shot capだけが成立しても、旧登録queryの他資源caps/workspace認証を代用しない。

## 5. known return/CTSの必要最小取得

既知returnのχ=5/8で、独立算術の閾値χ*は0.0016991270/0.0063835666。
Bret²/BA²は0.9809287044/0.9287211847で、レビューの値と一致。
これはnormalizationだけ。native改善を補間しない。

新しいzero-degree rotationのratioは3067/24456、763/3012。
新angle評価・合成は行っていない。両x×2 rotation labels×既存3 precisionの
**12 event実装のstrict phase-preserving cost/error/IR** が未取得。
将来の実際のprimitive synthesis key数はloweringを設計してから固定する必要がある。
label pとχへのaccess・条件付きsampling取得費用も、toyの明示情報と実問題を区別して記録する必要がある。

returnのoff-diagonal degree-2側は同じx/3 rotationとphaseを持つ既存O2 eventから
word labels不一致の4 event/precisionを抽出して再利用できる。
条件付き確率・native cost/error・IR identityを保存したので、ここには新合成を要求しない。
ただしreturned zero-degree側のMISSINGにより、returnの総resource比較は未完。

CTSの既存Pauli control結果は別context/情報accessであり、このdistinct-basis comparisonへ流用しない。
同じfinite targetに対するcollection、全Pauli coefficient/phase、dictionary construction/acquisition費用、
strict controlled cost/errorが不足している。この2-qubit toyはPauli展開可能で、I0優位の実証とは呼べない。
今回、それらを取得・構築していない。

## 6. 実行履歴・provenance・GPT G2への問い

最初の新診断processは、rotation-firstという保存label規約をword-firstと取り違えた検査で停止。
profile評価0件、input/result改変0件。初回source・scope・marker・failure・STOPをそのまま保存した。
本packetは保存値算術と技術修正を一括認可した作業で、旧scienceの別authorization one-shotではない。
検査だけを修正し、同一scopeのv2を結果前固定して初めて504 profileを評価した。
**diagnostic process invocation=2、preprofile source correction=1、completed profile pass=1**。
旧science retry=0。全packetのprocess retryを0と偽って報告しない。
初回18 testsの記録と、schema regressionを含む最終19 testsのPASSを分けて保存している。

完了process：wall 4.9173 s、CPU 4.9168 s、peak RSS 75,276,288 bytes。
wall180 s / CPU120 s / address-space512 MiB / output16 MiBの上限内。
値は完了processの計測で、初回失敗や資料整理を含む全作業時間ではない。
新registered LP、solver/backend、science/synthesis、circuit/matrix/trajectory、DF/NPZ/GPUは全て0。
旧762 pathを基点hashで照合し、文書index/overviewのappend例外をmanifestで明示する。

GPT G2には次を判断してほしい。

1. 約0.4–1.5%のideal差とknown IS/returnの比較未完を踏まえ、degree-local主claimを続ける情報価値があるか。
2. 続けるなら、J1と同条件A/PTSCのみの有限law・range/cap検査で足りるか。
   Tのzero-cost lawが未取得なので、T勝利を前提に次段階を設計しない。
3. known returnの12 event取得・strict controlled semanticsを先に閉じるべきか。
4. 本格v4 production/全LPが必要か、限定理論・mechanism noteへ縮小すべきか。
5. 同toyのCTS/IS対照を閉じた後に残る独立claimと、必要materialityをどう結果前に定義するか。

このpacketだけから全面productionの必要性は証明されない。
選んだ候補以外の拡張、元classの不可改善証明、新規性成立は主張しない。
GPT判断後の別指示まで追加検証・科学実行を停止する。

## 公開証拠の入口

- [resultと各axisのminima](../../../artifacts/track_b_g2_saved_diagnostic/2026-10-09/result_v1.json)
- [全504 profile表示CSV](../../../artifacts/track_b_g2_saved_diagnostic/2026-10-09/profile_display_v1.csv)
- [全profileの厳密interval・他資源・range・shots](../../../artifacts/track_b_g2_saved_diagnostic/2026-10-09/profile_rows_v1.json)
- [price表](../../../artifacts/track_b_g2_saved_diagnostic/2026-10-09/linear_price_rows_v1.json)
- [return/未取得事項](../../../artifacts/track_b_g2_saved_diagnostic/2026-10-09/known_return_rows_v1.json)
- [保存値identity照合](../../../artifacts/track_b_g2_saved_diagnostic/2026-10-09/saved_identity_audit_v1.json)
- [同task shot照合](../../../artifacts/track_b_g2_saved_diagnostic/2026-10-09/saved_postcheck_v1.json)
- [結果前scope v2](../../../artifacts/track_b_g2_saved_diagnostic/2026-10-09/diagnostic_scope_v2.json)
- [最終focused tests](../../../artifacts/track_b_g2_saved_diagnostic/2026-10-09/focused_test_receipt_v2.json)
- [publication/protection manifest](../../../artifacts/track_b_g2_saved_diagnostic/2026-10-09/evidence_manifest_v1.json)
