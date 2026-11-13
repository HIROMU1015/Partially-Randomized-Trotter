# PF有効Hamiltonianの技術基盤：人工検証のみ

2026-11-13 JST。**供給された決定論unitaryから、明示した対数分枝上のHermitian H_eff・固有位相/状態重なり・referenceとの差を診断する評価器を追加した。人工15＋既存回帰8＝23件PASS。実H4/H6の新計算・保存signal再解析・PF幅探索・N/RMSE/C/Gは0。正式な科学的バイアス評価方法には採択していない。** 公開照合後mandatory STOP。

[評価器](../../../src/trottertracks/pf_effective_hamiltonian.py)／[人工test](../../../tests/check_pf_effective_hamiltonian.py)／[TESTS](TESTS.txt)／[REGRESSION](REGRESSION.txt)／[人工packet](ARTIFICIAL_PACKET.json)／[HANDOFF](HANDOFF.md)／[未実行有限案](SCIENCE_PLAN.md)。

## 1. 調査・再利用したSOURCE

| SOURCE・処理 | 再利用・今回との差分 |
|---|---|
| A `91349e703e0d09ef702ccd108745dfaf1acc29f7`、数値fefd、`pr_rpe_signal_provider._macros` | fixed-step exact-tail controlとfinite-K corrected/rawの別伝播・scalarを確認。人工PR macroのexact-tailだけを新評価器へ入力し、R/Kに依存しない同対照を確認。実結果数値は再解析しない |
| `pf_power_stability.expm_step/power_diagnostics`（元dd24dd329949c255d5e37b90ff8993fa8df7159b） | 新しいPF builderを作らず、既存direct expm・signed順・q回power・gateを人工検証で使用。旧SOURCE・gate・STOPは無変更 |
| `h46_m5_plan.weights` | 既存S2/標準Yoshida4/M5 bestのsigned weightsを使用。新8次PFなし |
| C `86f37d25ffd5eee3daf7f25ce001b090b1f2cf33:baseline_signal_numerics.Context.pf_step/evaluate` | Full PFとB0/B0Iの実macro-step・scalarを静的確認。PF step自体はconstantを含まず、evaluateで総Tのconstant、B0Iでtail identityを一回加える。新評価器へはstep時間に対応する全scalarを含むUを渡す必要がある |
| C `c5466946ab0ef9b50511210ca1e58fb7147bb0f0:baseline_finite_signal_plan_20261113/` | Full S2/.1、B0I p7 S2/.1、Full-H K2 R448の**未実行計画**を確認。今回その科学条件・保存値の受入・39round計算を実行/変更していない |
| `pf_delta_validation._qpe_spectral_energy_distribution` | 既存のSchur・target中心branch・overlap/cluster解析を確認。ただし重み再正規化、target energyによる中心化、QPEのmoment/RMSEという旧解釈はそのまま今回のfinite-round RPE契約へ流用しない |
| `research_direction_energy_tail_pareto._matched_eigenphase_biases` | 既存のoverlap matching/nearest-reference unwrapを確認。今回reference energyをbranch oracleに使わず、minimumとstate spectral measureを区別するため、この自動matchingを呼ばない |
| `fr_revision_nonuniform._canonical_unitary_eigensystem` | 4次元専用の縮退basis処理を確認。今回一般dimensionのSchur基底と縮退group weightを返し、basis内の任意対応をground同定としない |
| sector/input SOURCE | 既存saved-input loader、`h6_df19_rpe_d2.make_context`、C `load_shared_h4`の同sector行列復元・binding/hash検査を確認。今回それらを実入力で起動せず、sectorやHamiltonianを新生成しない |

追加は数値診断moduleと人工testのみ。Hamiltonian/PF回路・RPE controller・N/risk solver・探索器は追加していない。既存A資源評価branchを91349e703から継続し、C/Bのbranch/worktreeには変更していない。

## 2. Claudeレビューとの関係

ユーザー指定のローカル資料 `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/Claude_PRリソース評価_RPE総コスト_統合科学レビュー_20261113.md` の§4.2、§7.3 T2'、付録A5を読んだ。参照版SHA256は`59002677698fb3304565dc822d1085fb63cd47f18b996fc20e6b340f2d69e648`。同ファイルは本成果へ移動・変更していない。

レビューの提案は「step spectrumからPF幅を予算へ合わせる候補手法」であり、正式採択・実分子での固有値シフト・Δ_max選択を意味しない。今回、特に次を分離した。

1. 数値的にU=exp(−iH_eff Δ)を再構成できること。
2. 選択したlog branchが物理的に比較したいbranchを表すこと。
3. minimum eigenvalueが保存state/RPEで実際に観測される成分であること。
4. 固有値シフトをenergy RMSEのtrot予算へ接続できること。

referenceに対するΔ‖H_ref‖<πと、明示したbranch中心cに対するΔ‖H_ref−cI‖<πはdiagnosticとして記録するが、PF因子・固有位相の連続追跡やground対応の証明にはしない。中心はreference energyから自動決定せず、選択energyにはcを戻すので物理energy shiftではない。人工の強い相殺を持つ非可換PFでも、このreference norm条件だけではH_eff≈Hの精度資格が出ないことを確認した。branchの有効性に必要な追加条件は科学的未決定事項として残す。

## 3. 数値評価の定義

SciPyのcomplex Schur分解 U=ZTZ†を使い、unitarity、Schur off-diagonal、固有値の単位円からのずれ、Zの直交性・固有vector residualを検査する。Schurのunitary basisという性質は[公式SciPy1.14.1資料](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.linalg.schur.html)で確認した。一般の行列logアルゴリズムを実装し直さず、unitaryの固有位相からH_effを構成する。

分枝は二つのみ。**PRINCIPAL**はE_j=−arg(μ_j)/Δという明示的な慣例で、alias解消済みと扱わない。**ENERGY_WINDOW**はcallerがSOURCE付きで指定した、一alias周期2π/Δより狭いspectral windowに一意に入る枝を選ぶ。これはRPE priorではなく、reference energyから自動生成しない。windowの物理的妥当性自体はcallerが供給すべき仮定である。

H_eff=Z diag(E_j)Z†を構成し、Hermiticityとexp(−iH_eff Δ)の再構成誤差を測る。scalarはUにstep時間で一回含める明示契約を要求する。global phaseを捨てるとenergyが変わるため、その除去・再正規化・unitary投影を行わない。人工の既知Hでは[公式logm](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.linalg.logm.html)との照合もしたが、productionで二経路を毎回呼ぶことは要求していない。

state weightsは|Z†ψ|²を**再正規化せず**返す。近接energyのgroupは明示numerical toleranceでまとめ、group weightと個別basis依存を分ける。元H_refが明示供給された場合だけ追加のHermitian eigensolveを行い、全固有basis間のoverlap行列とH_eff−H_refのoperator normを返す。reference energyはcallerの定義/SOURCEのまま保ち、このeigensolveの最小値で勝手に置き換えない。

phase cut/指定window境界の近傍、複数aliasを含むwindow、非unitary・非有限・不整合・数値residual失敗は**UNKNOWN**。finite-K corrected/raw等の非対象kindは**NOT_APPLICABLE**。failureでも観測診断値を保存し、非有限値はnullとする。成功statusは`EFFECTIVE_HAMILTONIAN_DIAGNOSTIC_AVAILABLE`であり、科学的PASSではない。

## 4. 適用範囲の区別

| operator | H_effの扱い | 元Hへの比較上の注意 |
|---|---|---|
| Full PF | scalar込み1step Uがunitaryなら対象 | 元full保存DF-Hと比較。正しいbranch・state overlap・有限RPEの統計は別証拠 |
| Truncated PF B0/B0I | 保持Hのstepがunitaryなら対象 | H_effは保持/切断H由来。referenceは元full Hのまま。B0Iのidentityと元constantを保全し、切断誤差を消さない |
| PR deterministic head | unitaryならheadだけの診断として対象 | PR全体のgeneratorやbiasではない。tail欠落を明示 |
| PR exact-tail control | 同じsplit/formula/実幅/順/scalarで構成したunitaryを対象にできる | R/K/κはexact-tail controlに現れないが、全roundの共通step検査とfinite-Kとの差・数値精度は別に必要 |
| finite-K corrected/raw | 対象外 | 一般にnonunitary、R/Kとroundで変化。rawはさらにattenuated。Hermitian共通H_effへ無条件置換しない |

全roundが同じUかつT_m=q_mΔならU^qのspectral measureでsignalを表せる。しかしΔを粗くした際のearly round・ceil-grid/末端短縮等でactualΔが異なると単一H_effは使えない。`common_step_schedule`はSOURCE付きstep descriptorとexact integer time関係の**metadata確認だけ**で、数値operator同一性や全RMSEを資格付けしない。

[METADATA_SCOPE.json](METADATA_SCOPE.json)は旧固定LD1/LD2の各13roundについてこのmetadata条件を確認したもので、新しい実operator・固有値・保存meanは評価していない。両構成が互いに同じH_effであるという主張でもない。

## 5. 人工・回帰検証

新15件は2×2/4×4のみ。既知Hと複数Δ、S2/Y4/M5 signed PFとscalar、Strangの既知δ² BCH係数、principal/window branch・cut/alias・不適格条件、最小energy成分と励起stateの不一致、mixed-stateのq=1/2/7/64 power signal、縮退group、truncated/full reference、PR exact-tailの既存macro、UNKNOWN・passive importを検証した。旧8件はdirect PFのsigned順・数値診断・旧固定SOURCE/STOPの保全を確認した。

StrangのPauli toy A=aX/B=bZでは、H_eff−H=Δ²[−ab²X/3+a²bZ/6]+O(Δ⁴)の係数を確認した。これは人工の既知展開との整合であり、実H4への係数移植ではない。最小eigenvalueにstate重なりが0の例では、最小固有値シフトが0でもRPE biasはnullのまま保持する。全spectral measureの方が必要である。[Yi–Crossonの元研究](https://arxiv.org/abs/2102.12655)も高overlapの仮定と固有vectorの摂動を扱っており、その結果を本finite-K RPE条件へ無条件に適用していない。

人工packetは`ARTIFICIAL_ONLY`、rpe_bias/十分N/global RMSE/Gはnull。数値の実測・toy理論整合・科学的資格を混同しない。CPU/RAM/workerの上限や新しい外部依存を追加していない。

## 6. 次の限定科学検証と未採択事項

最小案は、**ユーザーが指定するFull PFの一候補・一幅**について既存step生成を一回、Schurを一回、指定二時点のpower/spectral signalを照合すること。保存reference/solver energyを使う範囲と、保存signalを比較再解析するならその範囲も別途明示する。必要ならTruncated/PR exact-tailへ一候補ずつ拡張するが、自動追加しない。[SCIENCE_PLAN](SCIENCE_PLAN.md)にcall内訳・成功/STOP条件を分離した。

PF公式/Δ探索範囲・誤差予算・ground proxy採用・branch window根拠・重なりの十分条件・有限RPEへの誤差移送は今回決定していない。minimum shiftだけでMinkowski予算成立を認定せず、raw直接riskへ同じ誤差を二重加算しない。

**ユーザー**が方法採択の範囲と有限科学pilotを判断/承認する。**A/C**が許可されたstep/branch/overlap診断を供給し、**B**が全risk・N資格への接続を扱う。今回は公開・remote照合後mandatory STOPし、実分子計算・保存再解析・PF幅最適化には進まない。
