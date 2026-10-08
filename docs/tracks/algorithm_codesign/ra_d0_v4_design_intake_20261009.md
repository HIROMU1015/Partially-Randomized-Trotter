# RA-D0 v4 設計受領・次の統合監査

受領日：2026-10-09 JST。状態：`V4_DESIGN_RECEIVED_INTEGRATED_AUDIT_PENDING`。

利用者が共有した[GPT設計案](ra_d0_v4_numerical_design_20261008.md)を、次のTrack B作業の設計入力として記録する。
今回は文書の受領・整理であり、v4実装、backend導入、synthetic LP、登録最適化を実施していない。
設計案の判定 `PROCEED_TO_V4_INTEGRATED_MATHEMATICAL_AND_NUMERICAL_AUDIT` と、
独立監査の完了・source review PASS・実行authorizationを区別する。

## 設計の要点

1. 保存された非正規化degree列と有理数norm midpointにより、B2 mixtureとB3 degree matchingを構造等式で表す。
2. group総countsを先に配り、内部のprecision variantsへ配る階層largest-remainder法を使う。
3. mean、confidence、非目的resource capsの丸め損失を式から評価し、候補生成LPに事前の余裕を要求する。
4. exact feasible rational pointを取得し、丸め後lawを元のcertificateで独立に再検証する。

主案は共通のmargin付き生成器である。T0.2のordinary/O2、特定delta、特定precision pairを
productionへ固定する案ではない。単一点診断T0.3/T0.4を増やすことも、今回の次段階にしない。

候補生成は内側化される。元B2のlowerには元クラスを包含するouter relaxationを使い、
primary witnessは引き続き `U3_certified < L2_outer_certified` とする。
内側LPだけのinfeasibilityは、元B2/B3クラスのinfeasibilityでも、negative科学結果でもない。

## 固定証拠と解釈の境界

基点はT0.2 commit `d3a7cbb239487ddedf44699378f6c182c1fe5993`。
旧v3 sourceは `45cffb2aa10f9219b6cad929c3ade49fe7d36ca8`、旧authorizationは
`2daaf3b60a33db58de8fcdbbcce06f8e9ff163d9`、旧resultは
`35f8b949079f15d0348bc082b916324870da7246` のまま保持する。

| 証拠 | 保存済み状態 | この設計に持ち込める意味 |
|---|---|---|
| RA-D0 v3 | `D0_TECHNICAL_INCONCLUSIVE` | B2/B3比較に到達せず、優位・非優位とも未判定 |
| T0 | `T0_TECHNICAL_INCONCLUSIVE` | nominal membership問題とzero-mass projectionの未完了を記録 |
| T0.1 | `T01_MEMBERSHIP_REPAIRED_OTHER_CONSTRAINT_FAILED` | membership修復だけではconfidenceを満たせない保存一点 |
| T0.2 | `T02_FULL_CERT_PASS` | 指定precision移動後の保存一点の事後的certificate PASS |

T0.2の対象はsaved 2-qubit distinct-basis controlled finite P3、p=(3/4,1/4)、x=1/8、
`P1_ANCHORS:1/8:767135:minimum:T` 一件である。分子geometry、basis、DF rank、split L_D、PF delta窓は適用外。
PASSは汎用修復保証、B2 minimum、B2/B3比較、RA-RTE資源優位、新規性の証拠ではない。

固定された[既存T0.2 handoff](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3a7cbb239487ddedf44699378f6c182c1fe5993/docs/tracks/algorithm_codesign/ra_d0_t02_gpt_handoff_20261008.md)と
[verification](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3a7cbb239487ddedf44699378f6c182c1fe5993/artifacts/track_b_ra_d0_t02_exact_certificate/2026-10-08/verification_v1.json)を参照する。
旧消費済みmarkerのSHA256は `88a471e637d57c9896ffa9d3f6442c86ff1fec9859b50f3e86c4583095f7e735`。
旧runのretryやauthorization再利用はしない。

## 次の一件の監査で閉じる事項

以下は監査予定であり、今回PASSと判定した項目ではない。

| 対象 | 受理に必要な確認 |
|---|---|
| 入力adapter | saved v/c/Dの対応、normの正値性、precision間のD・内部sampling・phase identityを確認する。想定外のvariant除外をbaseline削除で処理しない |
| 構造等式 | B2からdegree matchingが従うこと、B3の自由度、inactive support、midpoint差のXiへの計上を導出する |
| 階層丸め | 固定ID tie、group/内部counts誤差、zero mass/zero counts、総和N、非負性を確認する |
| y/zとmembership | 保存モデルのYmaxを具体化し、丸め後yの境界を確認する。`Ymax rad <= 2/N` を満たす前提の下で元tauの保証を示す |
| 丸め余裕 | Gamma_xi/d/Qと共通max reserveからmean/confidence/resource/workspaceの同時保証を示す |
| exact取得 | 有理入力のround trip、exact primalの全row代入、独立dual/Farkas検証を確認する。doubleをFractionに包んだだけの解は受理しない |
| 公平性 | 新innerと原数値クラス/outerの包含方向、B2 seedのidentity付きalias合算・再量子化なし、gapの解釈を確認する |
| 実行可能性 | backend/version/build identity、refinement・elimination・certificate bytes、wall/CPU/RSS/callsの上限を結果前に固定する |

off-domain確認は設計案§14.2の全caseを一件にまとめる。confidence/cost境界、
保存dとnominal precisionの大小が逆の人工table、workspace不適格variant、
original feasibleだがinner空の反例も含む。exact backendの確認は保存R1 tableから独立した人工係数で行う。
T0.2の一点は既知regression fixtureとしてのみ扱う。

SoPlexは設計案のbackend候補であり、この環境での導入、exact入出力、性能を今回確認したわけではない。
旧2秒/LPや旧総call上限の自動継承を約束しない。数値層を主たる量子アルゴリズム新規性ともしない。

## GO/STOPと担当

次の検討単位はv4統合数学・数値監査であり、登録B2/B3比較ではない。
数学的前提、独立certificate、exact取得、計算費用に見通しが立った場合に、GPTへsource reviewと新契約の判断を戻す。
見通しが立たない場合も、単点診断の追加や結果を見たmargin/backend調整を自動で繰り返さない。

研究RQ、RA-RTE採択、新規性、強いbaseline追加、追加検証の必要性・範囲はGPT側が判断する。
Codexは具体的な承認済み監査scopeの実装・検証・provenance・報告を担当する。
登録最適化、新one-shot、合成、angle/precision追加、DF/分子/NPZ、circuit/matrix/trajectory、GPUは本受領では認可しない。

GPT設計案の「人工有理数240 cases PASS」はGPTによる自己検算の報告である。
今回Codexが再現した数値テスト、production検証、外部再現として数えない。

## 今回の保存と公開

独立branch：`track-b-ra-d0-v4-design-intake-20261009`。
rootの設計書一件だけをbytesを変えずにTrack B文書へ保存した。その他の未commit資料はcopy・stageしない。
既存のtracked source、contract、authorization、result、marker、Track Aを変更しない。
入力identityと今回の出力hashは[受領manifest](../../../artifacts/track_b_ra_d0_v4_design_intake/2026-10-09/intake_manifest_v1.json)に記録する。

今回は文書保存・Git公開・整合確認のみ。実装変更、test、LP、runner、science、合成は0。
公開後STOP。独立監査の完了や新実行承認を意味しない。
