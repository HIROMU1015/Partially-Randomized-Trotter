# M1-S Checkpoint 2後：近接先行研究の追加差分照合

2026-10-16 JST。**TECHNICAL LITERATURE AUDIT／NO NOVELTY ADJUDICATION**。

ユーザーが指定した2論文について一次資料を取得し、受理済み[T1b][T1B]と比較する。新しい還元定理・T1c・科学的新規性の最終判断は行わない。既存G0-pre/P0/P0.5調査は再実行せず、[GPT2][GPT2]§11が未確認とした差分に限定した。

## 1. 取得・版・読解範囲

| 資料 | 一次資料の取得状態 | 実際に確認した箇所 |
|---|---|---|
| Hayata Morisaki, Keisuke Fujii, *Optimized Randomized Hamiltonian Simulation via Average-Error Analysis* | [arXiv v1 PDF][MF]、[公式HTML][MFHTML]、書誌。arXiv提出2026-09-29、PDF表紙Dated 2026-09-30 | PDF pp2–4のlaw/評価、Eq(3),(16)–(29)、pp6–7 AppendixのEq(35)–(48)の関連証明。全benchmarkの再現はしていない |
| Changhao Yi, Elizabeth Crosson, *Spectral Analysis of Product Formulas for Quantum Simulation* | [arXiv:2102.12655v1 PDF][YC]（2021-02-25、表紙26日）。2022年出版書誌は[出版社検索取得情報][YCPUB]で確認 | v1 pp2–3 Lemma1/2、Corollary1、Lemma3の周辺、p6 Corollary2、pp10–13 Appendix C/D/Eの関連証明 |

Yi–Crossonの**2022年出版版本体PDFと公式supplementはこの環境で取得できなかった**。出版社のcookie redirect/非PDF応答、supplement CDNのDNS失敗を記録した。取得済みv1と2022年Version of Recordの定理番号・全証明が同一とは認定しない。出版社検索で見えた本文断片だけを全文取得扱いにしない。[取得locator][LOCATOR]にはPDF bytes/SHA256と失敗範囲を保存する。PDF全文はrepositoryへ転載しない。

## 2. Morisaki–Fujii：一次本文で確認した対象

比較の一次根拠は[PDF][MF] pp2–4,6–7（**PRIMARY TEXT CHECKED**）。

| 比較項目 | 一次本文の対象 | T1b側の対応・未照合 |
|---|---|---|
| Hamiltonian | Sum of Hermitian pieces | T1bはHD＋involution-tailとexact halves |
| Randomized law | IID choice of importance-rescaled exponentials | T1bは同位置共有±と固定rのstep |
| Branch共有 | The stated model samples a single evolution sequence | controlled liftによる包含可能性は未証明 |
| Channel／first moment | Both averaged channel and mean evolution operator are explicit | **coherent first momentというだけでは差分にならない** |
| Norm | Trace-norm mean-operator bound; Haar signal criterion | T1bはglobal HS superoperator normから固定state信号へ |
| 時間域 | Eq(35) is nonasymptotic for real time and integer step count | 長時間保証が無いとは扱わない。T1bは有限減衰C-secへの一様近似 |
| State/gap | Haar average, also a trace-norm operator estimate | 固定固有stateの共鳴floorと同じ主張ではない |
| 減衰／位相 | Bias relative to ideal complex signal | T1bは減衰参照を残し理想位相を扱う |
| Sampling | HS weights optimize the stated averaged bound | 非冗長単一Pauliではop/HS weightsが一致する点も既知 |
| 保証の型 | Variance bound, no asymptotic remainder in Eq(35) | T1bのsecular誤差と同一のboundかは未照合 |
| 直接重なる技法 | Centering, Duhamel, contraction, telescoping | これらをT1b独自の手法とは主張しない |
| 同値性 | Branch-correlated C-sec reduction is not established here | 未照合を「同等結果が存在しない」と読み替えない |

Eq(35)とAppendixを読むと、平均operatorは単なるchannelの副産物ではなく、信号bias上界の明示的な対象である。この点はabstractだけの比較より踏み込んで確認できた。Haar平均に加えてoperator trace normの評価があるため、「任意の固定stateへ何も言わない」という説明もしない。

一方、今回確認した証明をそのままT1bのΨ・Q_0・有限減衰参照・Jensen floorへ適用した厳密な還元は作成していない。Ancillaを含む生成子やHS vectorizationを通じた包含の有無は**数学的に未照合**。この直接系/個別性の評価が独立新規性を左右する。Scientific noveltyの結論はGPT/ユーザーへ残す。

## 3. Yi–Crosson：取得済みv1で確認した枠組み

以下の一次根拠は[2021年preprint v1][YC]。**PRIMARY PREPRINT CHECKED／PUBLISHED PROOF VERSION NOT VERIFIED**。

| 比較項目 | v1本文の対象 | T1b側の対応・未照合 |
|---|---|---|
| Hamiltonian | Time-independent sum of deterministic layers | T1bは乱択tailを含む平均写像 |
| Randomized law | Fixed first-order product formula | IIDやbranch共有のlawは前提にない |
| Branch共有 | No shared-branch ensemble in the stated setup | 非CP first momentの同一modelとは認定しない |
| Mean／operator | Effective Hermitian generator of a unitary product | T1bの散逸的log Ψと区別 |
| Norm | Operator/projector norms, fidelity and phase | 状態・位相をnorm距離から分ける着想は既知 |
| 長時間 | Eigenvector leakage is controlled separately from accumulated phase | T1bの全step誤差との定数・量化の同一性は未照合 |
| State/gap | Eigenstate/band and separated spectral subspace | T1bでは全H pair-sumと非直交mode射影も必要 |
| 減衰／位相 | Spectral shifts and eigenvector overlap | 乱択散逸のW/vやJensen floorを直接輸入しない |
| QPE | Precision-dependent step bounds under extra hypotheses | 本書のGate 0/RPE保証へ流用しない |
| 直接重なる技法 | Log generator, perturbation, spectral weights | 一般的mode/位相分離を新発明としない |
| T1b固有の提出対象 | 本文からそのままの一致は未確認 | 共有law、散逸、C-sec、outer/tail/mixingの係数 |
| 版・同値性 | 2022出版版の全proofは未取得 | v1の式番号を出版版のものと呼ばない |

v1のCorollary1とAppendix Cは固有basisのweightから位相とfidelityを分け、Lemma3/Appendix Eは追加条件下のenergy shiftを扱う。こうした一般的な枠組みとの重なりを保持する。T1bの具体的な三次係数や有限減衰のsecular化がどこまで直接従うかの証明は今回行わない。

## 4. 証拠区分と残る判断

| 区分 | 今回の状態 |
|---|---|
| 一次本文で確認済み | MF v1のlaw/mean信号bound/Appendix、YC v1の関連spectral proof |
| Abstract・書誌のみ／断片 | YC 2022出版版は全文proof未取得。v1と分離 |
| 既存監査から引用 | Campbell、PR-RTE、KGW、Chen、COS等のlaw/価格差。今回全文再監査なし |
| 数学的に未照合 | Controlled lift/HS liftによるMFからの厳密還元、YCとの全係数対応 |
| 科学的判断が必要 | 残る差分が独立論文の主貢献に十分か、technical noteの着地点 |

ここでの比較は**対象の定義の差と共通技法を示すもの**であり、非同値性や世界初性の証明ではない。MFの強い非漸近信号boundとYCのspectral解析を弱く記述してT1bの価値を水増ししない。逆に、一般技法が共通という事実だけで個別のC-sec結果が完全に含まれるとも断定しない。

原本や理論classificationは変更せず、新規証明・科学数値0。今回の文献追加だけで新しい全面科学レビューを開始せず、技術資料として公開してmandatory STOP。

[MF]: https://arxiv.org/pdf/2609.36694v1
[MFHTML]: https://arxiv.org/html/2609.36694v1
[YC]: https://arxiv.org/pdf/2102.12655v1
[YCPUB]: https://www.nature.com/articles/s41534-022-00548-w
[T1B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54336028edd7ef759f8a0c73bd0a96ade13f47f9/docs/tracks/algorithm_codesign/m1s_g0t1_t1b_partial_randomized_transfer_theory_20261016.md
[GPT2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/62554ee127a2bec1206367cf1754df2ff49fa000/docs/research/track_b_m1s_g0t1_checkpoint2_gpt_independent_scientific_review_20261016.md
[LOCATOR]: ../../../artifacts/track_b_m1s_post_checkpoint2_preflight/2026-10-16/v1/literature_locator.json
