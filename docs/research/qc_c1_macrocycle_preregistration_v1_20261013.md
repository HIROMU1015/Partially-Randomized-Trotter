# C1固定macrocycle最小pilot：結果前契約 v1

2026-10-13、scope `QC_C1_FIXED_MACROCYCLE_PILOT_V1_20261013`。
**準備だけが認可済み。科学実行は未認可、Mandatory STOP。**
[数学・情報・費用](qc_c1_macrocycle_technical_design_v1_20261013.md)、[機械契約](../../artifacts/qc_c1_macrocycle_preparation_v1/2026-10-13/pilot_contract.json)、
[literal入力](../../artifacts/qc_c1_macrocycle_preparation_v1/2026-10-13/inputs.json)を一組とする。

## 入力と固定比較

4件の2qubit Pauli toy、fragment A=sI−aZI−bIZ−jZZ、B=λXI+(λ/3)IX、C=μXX+κμYY。
s=1/7、caseの係数は入力JSONに全て有理数文字列で保存。geometry/basis/DF rank/rank policy/L_D/粒子sectorは適用なし。
これは分子Hamiltonianの性能resultではなく、C1の機構・比較公正性の最初の小さい判別である。
弱い結合、同程度の結合、signed C、強い結合を手書きで固定し、ground/BCH代入/PF結果を見て選んでいない。
ρ=|00⟩と悪い情報control |11⟩は同H上で扱い、独立8Hamiltonianとは数えない。

- 対称S2はreverse-isospectral quotient後の3代表。
- m=2,3,4のuniform subdivision、各代表。
- 24固定j,k,j cycle（a=1/8,1/4,1/3,3/8）。a=1/3の6件がknown uniform ordering対照。
- 各代表に五stage S4、3件。
- 3件の二点C2は保存m1/m2から導出し、追加eigensolveなし。
- strong known-compositionは同じcatalog/情報/選択規則を使用する同値性control。別の未実装既知optimizerへ勝ったと解釈しない。

Λはfragment係数L1の合計。h=1/(8Λ2^s)、s=0..5の6点。
target energy幅はmodel単位10^−5,10^−6で、PF/statistical予算を各1/2に分ける。化学精度/Hartreeに読み替えない。
36正時間unitary（single/subdivision12+macro24）とS4三つ、計39×6×4=936 PF unitary queries。
H referenceのeigh4回とunitary eig936回、**940 eigensolver calls**、上限1000。
C2は72導出record、state/価格κ3,6/targetの比較summary32件。shot sampling、最適化、SCF/FCI/compile/T合成はzero。

## 情報分離・precision・監査

係数のみからexact Pauli BCH、近似state期待値、係数由来gap/trace幅、同費用countsを求める。
empirical/joint/boxの選択と、joint PF boundによる最大h/nullを**reference前**にsealする。
参照ground vector/energy/gap、真d、finite-h残差、枝/overlapは後からの診断専用。
bad-state control、無効gap、null hも消去しない。

principal angle<π/4、full-H groundとのconnection≥1/2、principal最小quasienergyとの一致をreference側で確認する。
相違またはresidual≥1e−10、orthogonality≥1e−9、unitarity≥1e−10は未判定。
energy bias判定は1e−10 marginを加える。float eig/PF性能は区間証明でも外部再現でもない。
真値によりgrid上の最良hを選ぶfrontierはoracle診断と明示する。
coefficient-only precision選択の成否を別に記録し、事後frontierを実運用の誤差保証に置き換えない。

独立checkerは、unmerged別実装のstage registryからUを再構成し、保存全eigenpairs/H残差・完全性・phase・countsを照合する。
Djはexp/log generatorと異なるStrang再帰式から取得し、保存期待値/幅/選択/有理boundを照合する。
checkerは新しいeigensolverを使わない。同一担当のsource独立監査であり、独立研究者による科学再現ではない。

## 費用と結果の読み方

controlled RZ/CX/basis Cliffordを別記録。scalar phase、periodic single sharing、S4の負時間、C2の二推定を含める。
κは仮説価格。情報measurement zeroはproduct-state toyのみ、classical BCH/selection wallを計測する。
conditional Heisenberg energy-cost proxyと全QPE/RPE費用UNKNOWNを並記し、routing/T/confidence/分子state費用を勝手にzeroとしない。
主要係数が小さいだけでGOしない。grid上で精度/branch/overlapを満たし、strong baseline比較にも残るかを見る。

結果は少なくとも、既知compositionで説明、単純細分化で説明、真値oracleでだけ有利、情報/剰余/費用で消失、
eligible点なし、枝/数値未判定、条件付きtradeoffあり、に分ける。自動的な新規性・主研究採択はzero。
全成果がknown compositionに包含される場合、独立算法のclaimを作らず、同用途での狭い実用差の価値判断をGPTへ渡す。
全case失敗でも同型入力やaを追加しない。C4を自動起動しない。

## 実行資源・一回制限

計算資源はread-only inventoryを記録済み。future runは2spawn workers、BLAS/OpenMP各1thread。
4×4行列、Pauli closed-form exponentials、primitive cache、H独立caseの並列化を使用する。
symbolic raw/merged identityの登録外検査を先に通す。科学phaseの最適化経路・numerical referenceは実行承認後だけ検査する。
測定speedupは未取得。GPUは小行列の初期化/転送費用に対して合理的でないため使わない。

worker address-space1GiB、aggregate3GiB、primitive cache上限2000/case、output16MiB、log1MiB、calls1000。
新契約のphase/total wall-time上限は設けない。科学instructionは固定case/method/gridのみ。
起動前にruntime/source/input/plan SHA、origin HIROMU1015、CPU≥2、available memory≥1GiBを確認する。
one-shot output directoryはnumerical import前に消費し、attempt1/retry0。
例外/上限超過ではfailure記録を残し、別namespaceや再試行で回避しない。

## 起動前の次の担当

source・入力・runtime manifestとsource commitを固定したexecution planを公開し、remoteから再取得する。
通常技術準備を理由に同じGPTレビューを再要求しない。
**次はユーザーが上記940-callの非分子限定pilot一回を明示承認する段階**。
認可時はsource/manifest/plan/runtime identityと会話のapproval referenceを持つ別authorization JSONを作る。
現時点でそのJSONを発行せず、runnerは認可がなければNumPy import前に拒否する。
科学結果・独立監査・GitHub保存後STOP。重要科学結果/major不整合の解釈は別のGPT開始承認が必要。
新しい分子、より大きなinput、数値最適化、compile等をこの認可へ追加しない。
