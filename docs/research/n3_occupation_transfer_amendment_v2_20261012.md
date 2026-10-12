# N3 occupation transfer：実装前追補 v2

2026-10-12 JST。固定設計c2955ba0と完了済み科学レビューの限定修正を実装する。
元[v1](n3_occupation_transfer_design_v1_20261012.md)は変更せず保持する。
[レビュー原文](../../artifacts/n3_occupation_transfer_v1/2026-10-12/preparation/n3_occupation_transfer_design_scientific_review_2026-10-12.md)。
今回の認可は実装・synthetic検証・source/plan固定・公開まで。固定8入力の科学batchは未認可。

## 1. 科学条件の保持

実対称one-body＋非負unordered density、一外部mode、固定ν-sector。
4base、active m=6/7/8/9、ν=2/3/3/4、full modes=7/8/9/10、scale=1/1.5/.75/1.25。
各baseの同norm2方向、δ=scale/250=.004/.006/.003/.005。
geometry・化学basis・DF rank・L_D・PF/time evolution windowは適用なし。
係数packのSHA `c8df923eebfd999bf389ece2eb1728e7897c90e0950b51137785c708404489f4`
と48,201 bytesを保持し、recipe再生成・精度変更・新入力追加を行わない。
既存12入力とT3はdevelopment。新8入力のU/c/κ/採否/truthはこの準備で未評価。

共通trial U、c0/c_occ/c_star三行、NORM/FIVE/OLD_DIRECT/OCC四方法、raw96record以内を維持。
c_starは今回取得可能な三下界の最大であり、全係数下界のglobal optimumではない。
c_occ<c0になる場合もあるので、c0→c_occを必ず改善とは呼ばない。

## 2. 保存候補の継承とgain

全raw行の独自生成結果は保持する。同一input/source/U/δで、保存済み候補のκ/μ/τ/Zを固定し、
元のPSD行列を変更せず強いcでbox/affine条件を再確認する。
weak行でPSDだけ通りaffine条件に落ちた候補も保存し、結果前に決めた同じ再確認を行う。
別の数値κ探索を継承と呼ばない。

c_starのB*は全三行のNORM/FIVE/OLD_DIRECT候補をc_starへ再検証した安全なOR。
gain=raw_OCC_at_c_star AND NOT B*。同じ方法の別行からの継承、既存証拠fallback、
運用unionはgainへ数えない。弱い行への将来情報の無償逆流を避けるため、
その他の行では以前または同じ行で、元c≤検証先cの証拠だけを継承する。
元の証拠ID・origin row/method/attempt・source/input hash・検証先のU,c,δを保存する。

fallbackは名称だけではなくOCCの明示証拠を保存する。
NORM：κ=0,τ=0,Z=vvᵀ、lower=−||v||²。
旧法：同じκ、τ=μ,Z=0、lower=νμ。
κ=0のOCCはNORMと一致し、ν=1ではOCC族が旧one-body LMI族と一致する。
ν=1のOCC全体とNORMの同値性は要求しない。有限generator同士の発見集合一致も要求しない。

## 3. 有限proposal・検証・失敗の区分

各raw OLD_DIRECT/OCC searchは端点を含む128回以内、bracket幅2^-40。
best xから2^16/24/32/40のfloor/ceil＋端点、最大10有理κ候補。
PSDはexact Schur消去、checkerは元行列からのLDLと全要素再構成。zero pivot/rowを厳密確認。
数値toleranceはzero。数値solver SUCCESSや正のfloat scoreを保証にしない。

τ/Z候補はbits24/32/40×inflate係数1/4の6組以内。
外部固有値和では全6組を確認し、最大valid lowerを使う（同値なら最初）。
OCCのκ候補ではaffine条件まで満たす最初のvalid組で終了し、
全組でaffine不成立なら最大valid lowerを記録する。候補・precision予算はv1内を維持する。
PSD証拠が先に得られた場合、その後のcap/数値失敗で取得済みの有効候補を捨てない。

c≤U、下界未取得、finite探索不成功、eigensolver/nonfinite、bit cap、source/schema違反を区別。
whole-caseの有理capは12個の未判定raw行として保持し、入力を分母から除かない。
保証が得られないことを物理的縮約不能としない。

## 4. 段階別の予算予約

| stage | 最大actual numeric eig | 最大actual producer PSD |
|---|---:|---:|
| c0（5旧λ＋6 refined） | 1 | 11 |
| c_occ | 1 | 12 |
| c_dh | 1 | 12 |
| NORM、各c行 | 0 | 0 |
| FIVE、各c行 | 0 | 5 |
| OLD_DIRECT、各c行 | 128 | 10 |
| OCC、各c行 | 138（search128＋spectral10） | 120 |
| 証拠継承 | 0 | 0（同一PSD） |
| fallback埋込み（全caseのunique matrix合計） | 0 | 2（zeroとvvᵀ、旧PSDを再利用） |
| primary seal後のsecondary | 16 | 6 |
| 最大予約合計 | 817 | 448 |

case cap850 eig/512 PSDに収まる。各stageが別stageの予約を使い切らない。
cache hitはlogical requestへ記録しactual callには数えない。
failed actual callも数え、capによる未実行requestはdeniedとして別記録する。
OCC側のτ/Z pair180/case、外部下界側pair12/caseを別counterにする。
独立checkerのPSD再構成はproducer512予算に混ぜず、その検査費用を別に会計する。

trial determinant256、pair128、external diagonal256を維持。
explicit rational operands/resultsは4096bitsでguardする。Fraction内部の一時整数までは測定しない。
拒否された結果がcapを超えた値はmax-observedに含み得るが、accepted演算に許容したものではない。

## 5. 費用・性能・メモリ

case内でdeterminant diagonal、旧λ/LDL、数値小行列スペクトル、exact PSDをcacheする。
immutable matrix値とstrict flagをkeyにし、入力間・他caseからの保存結果を流用しない。
affine数値行列をsearch前に作り、候補生成はm×m小行列に限定する。
reference（cache無効）対cached経路の一致をsyntheticで検査する。

eventの依存関係・primitive key・actual/参考cost・cache hit・未判定理由を保存する。
BASE-OR単独経路は弱い下界からNORM→保存候補継承→FIVE→OLD_DIRECTの順。
OCC単独運用は同じ経路にOCC・自己継承・明示fallbackを加える。
再構成費用はその経路で必要になったproducer依存とunique primitiveを課金し、
factor-onlyの後段cacheを無料にしない。実測cold standalone時間ではないと明記する。
factor比較のwall/CPUと単独経路再構成見積りを別fieldへ保存し、speedup倍率を主張しない。

CPU32/affinity32とmemoryを確認、GPU driverに接続できないためGPU0。
最大2 case workers、BLAS/OMP/MKL/Numba/Rayon各1thread、spawn。
worker address-space4 GiB。aggregate8 GiBはrunner＋全descendants/resource trackerを含める。
50msごとのLinux RSS sampled watchdogで管理し、sample間の瞬間最大値は測定済みとしない。
parent peak・最大child peakをsampled aggregateと混同しない。
JSON shard32 MiB、全output64 MiB（通常62 MiB＋停止記録用予約2 MiB）、log8 MiB。
wall/CPU time capは置かない。tool polling timeoutは別。

## 6. 一次seal、secondary、分類、実行ゲート

全8 inputのprimary ledgerを保存・hash sealしてからconditional separatorへ進む。
secondaryはprimaryのU/c/δ/raw採否を変更しない。
旧max-margin proposal点、端点、中心の±2^-10/20/30/40という固定点で数値固有vectorを提案。
40bit dyadic vectorからrank-oneまたはzero-slope凸混合Wを構成する。
numeric eig16以内、exact PSD6以内。
W≠0、PSD、両端点traceが厳密負、同じc_starでNORM marginも厳密負の場合だけ旧族分離とする。
separator未取得は族の不可行性の否定でも、κ再探索の認可でもない。

G=0：NO_ADDITIONAL_ACCEPTANCE_OBSERVED。
G=1：SINGLE_BASE_GAIN_GATE_NOT_MET（gainゼロとは呼ばない）。
G≥2：FINITE_PIPELINE_TRANSFER_GAIN。
2 baseで適切なseparator＋NORM不成立も確認した場合だけCERTIFIED_FAMILY_TRANSFER_SEPARATION。
共通強下界の前提成立数、既存OR/OCCの採否数、whole-case未判定数、stage別未判定理由を併記する。
一般新規性、全既知算法への優位、実分子、量子総資源または中心テーマは自動採択しない。

runnerの--preflightはschema/source/input/version検査だけで、科学predicateを呼ばない。
--runは別の明示authorizationを必要とし、source/input/plan hash、attempt1/retry0、
専用execution branch、clean worktree、新しいrun1 directoryを確認して初めて始める。
この準備のplanには実行権限を付けない。
N1追加探索停止、N2保留、旧source/results/STOP、他Trackの状態を保持する。
