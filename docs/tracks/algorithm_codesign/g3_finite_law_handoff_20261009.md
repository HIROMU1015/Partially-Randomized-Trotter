# Track B G3：有限lawと既知return比較の結果・GPT handoff

2026-10-09。状態は **`G3_PHASE_A_FINITE_LAW_DIAGNOSIS_COMPLETE`** および
**`G3_KNOWN_RETURN_COMPARATOR_COMPLETE`**。認可された技術検査を完了し、**mandatory STOP**。
次の研究採否・新規性・比較scopeはGPT G3へ戻す。全面v4・登録LP・CTS/DF移送を自動認可しない。

## 認可・source・証拠区分

利用者が添付した[GPT G2科学レビュー](../../research/track_b_G2_scientific_review_20261009.md)§8が、
保存値Phase Aと、差が残った場合の限定return取得Phase Bを指示した。
原添付SHA256 `11fcf975c577b3da69d1a9ee6e746a2c7bdda0e3be4e5cb38420dcd7b1c30396`。
CRLF→LFだけで保存し、本文の科学判断をCodexによる独立結論と扱わない。

- 独立branch/worktree：`track-b-g3-finite-law-diagnostic-20261009`、repository内`.worktrees/`同名。
- 旧G2 result base：`b260189b020ab7dfabb16bf424f49a6efff40d75`。
- Phase A clean execution source：`381449daefc7310e22729725f69656dd651b3d20`。
- Phase B clean execution source：`34fdf1c21becd96a77122a9c468b4191b9dbf2ad`。
- 各Phaseのsource/scope固定後、一回だけ起動。exclusive markerを保存。retry各0、failure各0。
- 旧R1/v3/G1/G2のsource・result・authorization・marker・STOPを保持。元のtechnical inconclusiveを変更しない。

これは**既知development入力のpost-hoc有限law診断と限定native合成**である。
独立held-out、旧RA-D0のcertified `U3<L2` witness、全B2最適性、immutable CI、実機測定ではない。

## 固定対象と比較範囲

2-qubit distinct-basis、`R=3/4 Q0+1/4 Q1`、`Q0=ZI`、`Q1=V†IZV`、`V=exp(−iπXX/16)`。
controlled finite `P3=I−ixR−x²R²/2+ix³R³/6`、σ=+1、x={1/8,1/4}。
geometry/molecule/basis/DF rank/PR split L_Dは適用外。input |00>、workspace1 beyond2 system。
Hamiltonianの新規生成・取得なし。accuracyはfinite P3に対するaxis ε=0.005、α=1/5280。
Taylor truncationからexponential/QPEへの最終総cost保証は範囲外。

固定candidate table SHA256
`2f86169fc301ffa6b73f3cde7c8bd49939a4e7742d422c309bac181e5f09d0b6`。
原R1結果SHA256
`f726ad70cb2643533f0d037b518cde1b702724adb4e6571fea25e26e4bfdd61e`。
旧21 columns/x、precision={1e-3,1e-4,1e-6}のcost/error/phaseを保持。
ordinary/PTSC-K0/Aの63 pure precision profiles/x、J1の81/x、合計288。
既存3対照とJ1へ同じproposal/precision自由度を与えた。

**重要な限定**：主表は各profile・各optimized axisで得たwinner集合（A864＋return54）を
全3座標で共通参照した有限poolの比較。6,912候補全体を別axisで再最適化した値ではなく、
任意の旧B2 representation混合・precision混合・全IS lawのglobal minimumでもない。
別表として同じoptimized axisのwinnerだけの比較も保存し、差の符号が一致することを確認した。
G2のideal pure-profile補題をfinite Bernstein/range/多資源問題の完全性証明へ拡張していない。

## 有限law・confidence・資源

[固定Phase A設計](g3_finite_law_design_20261009.md)に従い、cost-zero eventを0のまま保持した。
zero groupの正massとpositive groupのcost-IS、canonical混合を共通の有限集合から生成。
group normのmidpointをexact conditional IID shareに掛けてa_iとし、q_iを分母2^60へ丸める。
全q_i>0、Σq_i=1。補正weight `w_i=a_i/q_i` はexact rationalで、有限integer samplerを構成できる。
実samplingや量子measurementは実行していない。

`Σq_i w_i U_i=Σa_i U_i` の取消しとgroup share保存は厳密。
ideal係数でのmeanはP3に代数的に一致するが、デジタル係数と合成後operatorは誤差を持つ。
別verifierでcoefficient L1、normalized degree mean、controlled phase/IR/cost/error bindingを確認した。
選択された64 complete lawsの再照合における最大degree mean残差上界は約4.40e-58。
coefficient errorは1e-25 cap内、degree meanは1e-12 cap内。
補正weightの最大numerator/denominator bit lengthは選択law内203 bits。
60-bit proposalの累積integer表とexact rational weightを保存し、無限大weightのT infimumを達成値と呼ばない。

`m2=Σq_i w_i²`、`L=max|w_i|`、bias=coefficient L1＋`2Σa_i δ_i`。
`s=1/200−bias>0`、ellはln(10560)の外向き上界。
`n=ceil[ell(2m2/s²+(4/3)L/s)]`、n≤10⁹/axis。
独立exact arithmeticで `n s²−ell(2m2+(4/3)Ls)≥0` を確認。
T/CXは2n EC、1Qはn(2EC+5)。state preparation T/CX0、Re/Im readout+Hadamardの1Qは2/3。
各taskの共通sufficient-shot予測であり、全候補への新しい実測familywise保証ではない。
workspaceは全て1。旧queryのnonobjective capsを新設・変更せず全座標を報告するため、
**旧multicap queryへのfeasibility witnessとは呼ばない**。

## 結果：資源ごとの固定pool最小

以下は各座標の別々の最小値。単一lawが表の全最小を同時に達成したという意味ではない。
単位はfinite confidence taskの予測総T/CX/1Q count。費用の恣意的加重和・新materiality閾値なし。

| x | 座標 | 既存ordinary/PTSC/A最小 | J1最小 | known return最小 | J1対既存差 |
|---|---|---:|---:|---:|---:|
| 1/8 | T | 179,987,284.56（A） | 179,160,991.73 | 187,808,511.12 | −0.4591% |
| 1/8 | CX | 4,281,301.64（PTSC） | 4,310,150.09 | **4,245,968.98** | +0.6738% |
| 1/8 | 1Q | 465,290,176.32（A） | 463,389,609.27 | 481,297,026.22 | −0.4085% |
| 1/4 | T | 177,548,958.29（A） | 174,720,368.17 | 189,825,055.94 | −1.5931% |
| 1/4 | CX | 4,610,549.46（PTSC） | 4,723,948.73 | **4,448,986.39** | +2.4596% |
| 1/4 | 1Q | 468,173,605.23（A） | 461,712,911.04 | 495,013,282.20 | −1.3800% |

J1 T/1Q選択lawは、既知returnを含むこのpool内でnondominated。
同optimized-axisだけの公平な別集計でもT差は−0.4603%/−1.5950%、1Q差は同じ。
全B2に対するPareto membershipや最適性は未認証。差はsource数値誤差より大きいが、
**その大きさが研究として重要か、離散synthesizer notch依存かはGPTで判断する**。
1Qは今回のpost-hoc診断であり、旧RA-D0のpreregistered primaryへ昇格させない。

全座標を見るための例：

| x / law選択 | T総量 | CX総量 | 1Q総量 |
|---|---:|---:|---:|
| 1/8 Aの1Q選択 | 179,987,284.56 | 4,483,213.00 | 465,290,176.32 |
| 1/8 J1のT選択 | 179,160,991.73 | 4,915,547.85 | 481,058,984.68 |
| 1/8 J1の1Q選択 | 179,183,039.36 | 4,465,884.67 | 463,389,609.27 |
| 1/4 Aの1Q選択 | 177,548,958.29 | 5,718,904.22 | 468,173,605.23 |
| 1/4 J1のT選択 | 174,720,368.17 | 6,662,825.87 | 483,946,285.80 |
| 1/4 J1の1Q選択 | 174,820,322.72 | 5,661,819.95 | 461,712,911.04 |

Tだけの選択は他座標を悪化させ得る。1Q選択の2点は、この選ばれたA点の全3座標を改善する。
それをB2任意混合のstrict dominanceと呼ばない。

## known return：取得・同target・位相・取得費用

[結果前Phase B固定設計](g3_return_comparator_preparation_20261009.md)でχ=5/8と
ratio3067/24456・763/3012を固定した。12 keys/12 event条件だけ取得し、旧R1のpygridsynth2.0.0、
Python3.10.12、mpmath1.3.0、全package lockおよびsource .py tree一致をlaunchで確認した。
旧seed0/dps100/up_to_phase=false、ε/4 requestを保持。strict Frobenius上界でoperator errorを検査し、
global phase minimizationを行わない。全12合成はerror guardを通過。

| x | precision | label0 T/CX/1Q | label1 T/CX/1Q |
|---|---|---|---|
| 1/8 | 1e-3 | 80/2/205 | 156/6/402 |
| 1/8 | 1e-4 | 100/2/254 | 200/6/509 |
| 1/8 | 1e-6 | 140/2/357 | 276/6/706 |
| 1/4 | 1e-3 | 76/2/196 | 152/6/393 |
| 1/4 | 1e-4 | 100/2/257 | 200/6/512 |
| 1/4 | 1e-6 | 140/2/348 | 276/6/697 |

sequence全文、SHA256、T/T†、strict error、IR SHA、費用をJSONへ保存。
既存basis conjugatorとcontrolled primitive規則を共有し、word01/10のO2 IR/errorを再利用。
再照合で24 off-diagonal event identity（4 events×2 x×3 precision）の不変を確認した。
return conditional列にはlow-degree χ returnを含むため、単に旧O2係数だけへ置換していない。
登録外x=1/3の非可換small-matrix testでP3 meanとcontrolled relative phaseを検査した。
登録domainの新しいHamiltonian/finite signal/matrix oracle取得は0。

returnは同じproposal/precision/confidence規則を持つ18 profiles・162有限lawで比較。
T/1Qでは現在のJ1選択を吸収せず、CXではreturnが最良となった。
既知returnのnormalization利点が総T低下へ直結するとは限らない、という限定toyの記述的結果。

古典取得について、pは既に与えられた2項、χは2項の二乗和、off-diagonal条件付きlawは
01/10各1/2＋独立rotation pから構成する。拒否samplingを必要としない有限累積表も構成済み。
full-support sampler table最大event数はordinary10/PTSC14/A14/J1 18/return6。
60-bit RNGと有限表lookup、203bit以内の補正weight算術が必要だが、実shot毎の古典実測費用は未測定。
今回のsetup runtimeと12合成費用は記録し、量子資源へ無根拠に換算しない。
大規模DFでのχ取得・辞書取得・classical acquisition advantageは未検証。

## 実行・検査・provenance

| 項目 | Phase A | Phase B |
|---|---:|---:|
| 起動 / retry / failure | 1 / 0 / 0 | 1 / 0 / 0 |
| profiles / laws / rejected | 288 / 6,912 / 0 | 18 / 162 / 0 |
| wall | 26.862s | 0.883s |
| CPU（測定scope内） | 26.859s | 0.865s |
| peak RSS | 56.88MiB | 177.94MiB |
| new synthesis / native event IR | 0 / 0 | 12 / 12 |
| quantum measurement / LP / science input / DF / NPZ / GPU | 0 | 0 |

capはA wall600s/CPU480s/AS512MiB/15,000laws、B wall600s/CPU480s/RSS512MiB/AS1536MiB/
per-key30s wall・20s CPU/1,000laws、packet32MiB。hitなし。
Bのmatrix計算は12 synthesisのstrict primitive guard。別に10 synthetic focused testsでmatrixを使用した。
科学signal取得・wrapper trajectory・旧registered LP・新角度探索は0。
focused testsはA17/B10 PASS、full suite未実行。
read-only post-auditは64 complete laws、12 sequences、12 native count rows、24 reused eventsを再検査。

[evidence manifest](../../../artifacts/track_b_g3_finite_law/2026-10-09/evidence_manifest_v1.json)と
[publication audit](../../../artifacts/track_b_g3_finite_law/2026-10-09/publication_audit.json)に
source/contract/scope/runtime/marker identity、旧789 path照合、追記のみのindex変更を記録する。
rootの未commit研究plansやTrack A worktreeから資料を一括copy/stageしていない。

## GPT G3で判断する未決事項

1. finite T/1Qの約0.4–1.6%差を、有限比較範囲・離散合成費用・古典setup/weight overheadを踏まえてどう扱うか。
2. 任意B2混合や連続proposal最適化を閉じる情報価値が、全面v4実装に見合うか。
3. 同一toyで未取得のCTS/Pauli acquisition、他の強い既知representationと比較する必要と最小scope。
4. returnがCXで最良、J1がT/1Qで非支配という機構を独立した選択原理へできるか、それともtheory/mechanism noteへ縮小するか。
5. 継続する場合のprospective実回路条件・独立性・metric・materiality・資源上限を結果前にどう固定するか。

Codexは研究GO/STOPの科学的採否を代行しない。**現在は停止、次stage未認可**。
