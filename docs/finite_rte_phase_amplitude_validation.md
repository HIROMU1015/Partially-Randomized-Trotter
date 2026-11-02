# finite-RTE位相・信号半径分離 FR-1検証

最終更新：2026-09-26 JST

## 結論

事前登録した2×2 fixed gridのFR-1を完了した。33条件、99状態評価、495個の適用可能な
method recordで位相上界・信号半径下界の違反は0だった。identityを含むpaired-RTE列挙、
controlled relative phase、負時間、K=4、非対称配置も固定許容誤差内で整合した。

ただし主判定状態に利用可能な下界$\underline\rho=0.8$だけを与える
`PROPOSED-AVAILABLE`は、非可換primary gridのどの点でも事前登録した有用性gate G2を通らなかった。
最小の提案/STRONG位相上界比は0.631998で、要求値0.5以下に届かない。$10^{-3}$ radの認証を
提案法だけが通る点もなかった。

一方、参照固有状態の$\rho=1$、またはdense計算で得た真の$\rho$を使う`PROPOSED-REF`では、
$q=2,4,8,16$で提案上界がSTRONG-NORMの50%以下となる点が存在した。従って事前登録規則による判定は

`GO_FR2_MECHANISM_ONLY`

である。これは位相方向と半径方向を分ける機構が数値的に成立したことだけを意味する。
利用可能情報による実用的認証は成立していないため、FR-2、H4/H12、回路compile、RPE総costへは進まない。

## 固定scope

- $H_D=0.7Z$。
- $H_R(\theta)=\cos\theta Z+\sin\theta X$。
- primary：$\theta=\pi/3$、$T=0.8$、$K=2$、$r=1$、$q=1,2,4,8,16$。
- control：可換$\theta=0$、強非可換$\theta=\pi/2$、$r=1,2,4$、負時間、非対称配置、K=0/4。
- 局所次数監査：$\Delta=0.05,0.1,0.2,0.4$。
- 状態：参照固有状態、解析的に$\underline\rho=0.8$を保証できる二分枝重ね合わせ、物理基底状態。
- complex128 dense matrix。Monte Carlo sampling、量子shot、chemistry Hamiltonian、compileは未使用。

入力とgateの正本は
[FR-1事前登録](research/finite_rte_phase_amplitude_fr1_preregistration.md)、数式の正本は
[FR-0契約](research/finite_rte_phase_amplitude_contract.md)である。

## Gate結果

| Gate | 結果 | 内容 |
|---|---|---|
| G0 semantic consistency | PASS | ordinary/controlled列挙とpaired Taylor平均が一致 |
| G1 soundness | PASS | 適用可能495 recordで位相・半径bound違反0 |
| G2 available noncommuting utility | **FAIL** | $\underline\rho=0.8$では50%比・片側$10^{-3}$ rad triggerとも0 |
| G3 conditioning/rejection | PASS | 利用可能な$\underline\rho$の妥当性とinapplicable処理が整合 |
| G4 sign/cutoff/asymmetry | PASS | 負時間、K=4、非対称配置でbound違反0 |

G2不通過は数値実行失敗ではなく、中心仮説の利用可能情報版が固定閾値へ届かなかったという研究結果である。

## Primary grid

`analytic_mixture_state`、$\underline\rho=0.8$での位相上界を示す。

| $q$ | PROPOSED-AVAILABLE | STRONG-NORM | 比 | G2 |
|---:|---:|---:|---:|---|
| 1 | 2.341293e-2 | 2.115382e-2 | 1.106794 | FAIL |
| 2 | 2.276553e-3 | 2.662405e-3 | 0.855074 | FAIL |
| 4 | 2.425161e-4 | 3.331889e-4 | 0.727864 | FAIL |
| 8 | 2.766262e-5 | 4.166172e-5 | 0.663982 | FAIL |
| 16 | 3.291551e-6 | 5.208170e-6 | 0.631998 | FAIL |

$q=1$では提案上界の方が悪く、$q$増加で改善するが、事前登録した0.5比には達しない。
$q\ge4$では両法が既に$10^{-3}$ rad以下なので、片側認証triggerにも該当しない。

## Mechanism-only診断

参照固有状態では$\kappa=0$となり、一次radial誤差が位相boundから外れる。代表値は次である。

| $q$ | PROPOSED-REF | STRONG-NORM-REF | 比 |
|---:|---:|---:|---:|
| 2 | 6.722361e-4 | 2.129923e-3 | 0.315615 |
| 4 | 4.250164e-5 | 2.665511e-4 | 0.159450 |
| 8 | 2.664068e-6 | 3.332938e-5 | 0.079932 |
| 16 | 1.666259e-7 | 4.166536e-6 | 0.039991 |

物理基底状態もこのtoyでは真の参照半径が0.999689以上で、denseの真値を使う診断では同様の改善が出た。
しかし、この真値を事前に安価に得る方法はFR-1で示していないため、実用的GO根拠に使わない。

## Semantic・次数・数値整合

- explicit paired-event count：K=0で3、K=2で30。
- ordinary平均の最大operator residual：$4.4431\times10^{-16}$。
- controlled relative-phase最大residual：$3.3379\times10^{-16}$。
- matrix local errorと解析的spectral supremumの最大差：$2.0817\times10^{-17}$。
- 最小観測信号半径：0.495063。
- 最小参照信号半径：0.809791。
- 局所radial/tangential log-log slope：
  - K=0：1.98168 / 2.99273
  - K=2：3.97543 / 4.99134
  - K=4：5.97228 / 6.99056

STRONG-NORMは有限grid最大値を使っていない。$|d_K(x)|^2$の導関数をK=0/2/4について解析し、
$|x|\le0.8$で単調非減少であることから端点をinterval supremumとした。

## 研究判断

FR-0の命題はこのfixed gridで反証されず、radial/tangential分離の機構も確認できた。しかし、
一般状態で必要な$\kappa(\underline\rho)(s+R_2)$項が利益を縮め、$\underline\rho=0.8$では
STRONG-NORMに対する事前基準を満たさなかった。

従って現段階で主張できるのは次だけである。

> 参照信号半径が1に十分近い場合、finite-RTE相対誤差の位相方向と半径方向の分離は、
> norm-only位相上界より大幅に小さい位相上界を与え得る。ただし、利用可能な粗い半径下界だけで
> その利益を保持できることは今回のFR-1では示されなかった。

次に大きな計算を行わない。研究を継続するなら、先に「高い$\rho$をoracleなしで安価に認証する条件」
または「$\kappa s$項を非可換構造から縮める新しい境界」に独立した差分があるかを再設計・先行研究監査する。
それが固定できない場合、この方向はmechanism observationとして停止する。

## Artifact・再現性

- artifact：`artifacts/finite_rte_phase_amplitude/2026-09-26/finite_rte_phase_amplitude_fr1_v1.json`
- validation fingerprint：`d96b200163f3a432652656ae97c65c837e01fe81323cd69528373dff16ce6152`
- file SHA-256：`6a81a0ba6e39ba0f5d79ba026a2a45c3c65709e071e9c6fa5606b3e99a89b0f7`
- 実行時間：0.0141秒
- 専用test：`4 passed`
- 全suite：`619 passed, 2 skipped, 4 warnings`、失敗0

artifactはdirty worktreeで生成したlocal evidenceであり、immutable CIまたは外部独立再現ではない。
実行前に凍結したFR-1事前登録（SHA-256 `bc8066d7a31a3f46f476d2c591c7f0dad5e6f49023fc261dc518b61e2ec600a3`）が
完全なgridとgateを保持する。凍結した親FR-0契約（SHA-256 `8e53763d6cd03fb10809c1a2450358834b3f9aa0dbc5677fe9a25fbc4fd45b04`）は、
後の監査でMarkdown末尾が途中終了していると判明したため監査履歴として不変保存し、現行契約側で欠落部分を修復した。
この文書上の欠落は、独立に完全凍結したFR-1事前登録に基づく計算grid・gate・判定を変更しない。
最終総cost評価は行っていない。
