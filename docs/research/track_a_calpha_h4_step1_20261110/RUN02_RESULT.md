# C-α／α0-P H4 Step 1：sector条件訂正後の二点結果

2026-11-10 JST。担当：Codex。

**判定：保証改善が必要。許可されたm0／m17の数値診断を完了し、mandatory STOP。**

同じgeneric辞書の平均・完成head・保存参照への接続を確認し、前回未取得だったΓ moment、raw平均信号、減衰参照、位相・半径の十分条件を取得した。m17のraw半径は **0.1265744404**、元energyに対する観測円周位相差は **0.01381956080 rad**。観測位相差は新C-αの0.14 rad枠内だが、現在の十分条件はhead内部誤差の累積項 **10.74909696** が支配し、正の半径下界も位相上界も得られない。

これは固定条件の信号が悪いという結果でも、C-α全体の不可能性の証明でもない。現段階では全18roundの会計へ進む根拠が不足し、保証の改善を先に判断する必要がある。数値診断を厳密上界へ認定せず、包括的な実装保証はUNKNOWNのまま保存した。

## 1. 実行範囲・保存先

ユーザーの今回の「実行してよい」に基づき、訂正後Step 1の残作業だけを一回実行した。新規の先行研究監査や理論の重要レビューは実施していない。

| 項目 | 固定条件 |
|---|---|
| 系 | Linear H4、1.00 Å、STO-3G、8 spin orbitals |
| target・入力 | 元H_phys、保存DF12、既存generic DF frame |
| 分割・状態 | generation-prefix LD=3、同じ元H保存state、Nα=Nβ=2／sector36 |
| law | 正時間partial-S2、α0-P／paired K0、fresh independent samplingに対応するcoherent平均 |
| 時間 | δ=0.02、r=4、θ=0.005。m0：q=1／T=0.02、m17：q=131072／T=2621.44 Ha^-1 |
| 予算 | 新C-α systematic 0.14 rad、stat 0.24、reserve 0.02、全体0.4。旧P1個別枠は不変更 |
| 費用条件 | 状態準備費用除外。今回shot・価格・総費用は未算出 |
| 作業場所 | branch `codex/track-a-calpha-h4-step1-20261110`、既存専用worktree `.worktrees/track-a-calpha-h4-step1-20261110` |
| 実行SOURCE | 基点 `0d5ca357ccfd8fcc850bef800c5bcadfde2a153e` と下記のlocal runner。未commit |

- [run02 raw JSON](../../../artifacts/calpha_h4_step1/2026-11-10/run02_sector_corrected/result.json)：`TWO_POINT_NUMERICAL_DIAGNOSTICS_COMPLETE`、`mandatory_stop=true`。
- [実行log](../../../artifacts/calpha_h4_step1/2026-11-10/run02_sector_corrected.log)。開始は2026-11-10 20:18 JST、runner記録時間4.2300秒。
- [実行runner](../../../scripts/run_calpha_h4_step1_mean.py)。SHA256 `05974d2ac2372e360aab0933ffbf4a8b3b562608d31f4f6c52bd726046610436`。
- raw JSON SHA256 `8ff3e10d6fb663714b68aa0e9b2affa0816bc04d92c509eede7dca8791fbfffb`。

保存した新raw信号は2点、減衰参照も2点。新ground solve、回路build、compile、sampling、shot会計、他round信号は各0。全18round、別r／LD／law／frame／sector、新SCF・積分生成・DF fit、価格MC、旧runのretryへ進んでいない。成果はlocal未commitで、immutable CIや外部再現証拠ではない。commit／pushは実施していない。

## 2. 採用資料・旧STOP・入力保全

採用した追補はrootの `track_a_calpha_sector_condition_review_20261110.md`。SHA256は `d681afee3c3ed579506f691ab0538d76341b6746509bcc4b52526988264c3108`。§3–6の平均・完成block、§8のsector外成分を含む接続、§11–12の訂正を適用した。元理論レビューの位相・半径式は引き続き使い、原本は変更していない。`research_project_instructions_unified_2026-11-09.md` は確認できた環境では未取得であり、読了とは扱わない。

[旧run01 REPORT](REPORT.md)、旧runner、旧raw JSON、旧logはbyte不変更。[保全archive](../../../artifacts/calpha_h4_step1/2026-11-10/preserved_run01.tar)にも同じ4ファイルを保持し、一致を確認した。archive SHA256は `8cf61d048e3cdd6d2ff94a5c6f75d65a95598787913ca92053f4fb4cc09ce62b`。旧status `TECHNICAL_STOP_PRIMITIVE_SPIN_SECTOR_NOT_PRESERVED`、単一Z atom46件の混合診断は当時の指示に沿った結果として残す。今回は個別atom検査を再実行せず、正しい平均の解析に必要な条件を訂正した別runである。

既存P1 loaderで元H・state・sectorのarray identityを照合し、保存DFの同じ係数projectionを使用した。新stateへの置換、state再正規化・phase変更、非保存atomのdrop、係数cutoff、sampling確率・λ・identity抽出の変更はない。

| 入力・identity | 値 |
|---|---|
| 元H_phys array | `bab58c17220230196aff8a7c7d1e1264c36e101f0fe6bf367eacba4538ae10b2` |
| 同じ元H full state array | `1a2df11b16743e68fb27a3f4b39e78e145897cb49a8cc931b080e44e7400c1db` |
| DF Hamiltonian identity | `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424` |
| state.npz file | `deb561367e216fd4c62d013c10a4adbbef5da8c3e8139c39170d71f80530a80c` |
| 保存DF12 NPZ file | `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a` |
| 元HDF5 file | `a3c88f6afa8965a994acb4f6ede4b8fcea1d074755f282b292c790e4c573a958` |

state生成SOURCEは `b168cbf93d7c319045057e8b67ea7ff014c8d7dc`。`89165cf70958e7e4676d0580e3acb7faf588772c` はその来歴・gapを読む保存記録の入口であり、生成SOURCEと混同しない。今回の再構成DF matrix hash、sector順序、projection hash、使用libraryのSHA256はraw JSONに保持した。private state・積分・HDF5本体は公開結果へ含めていない。実行後も原入力file hashは不変だった。

## 3. 同じlaw・sector解析への接続

既存 `load_private_inputs`、`mean_preparation`、`compose_step`、RTE moments、generic diagonalizer、CAR作用とvectorized exterior-minor liftを再利用した。K2固定・native buildを伴う旧driver全体は呼ばず、必要なmean-only接続を新runnerへ限定した。[準備時点の設計・再利用対応](SECTOR_CORRECTION_PREPARATION.md)を維持した。

固定signed atom和をfull Fock空間で構成し、元のλ_R・identityと結合した。物理平均 `(I−iθR_dict)/B0` は既存RTE関数で構成し、normalizationを一回だけ適用した。個別 `J†Q_lJ` の指数化やtrajectory射影は行っていない。headもgeneric basis変換・対角指数・逆変換を含む完成blockで構成した。

解析参照には同じCAR tailの明示したHermitian部分 `R_ref`、同じCAR完成head指数、`H_ref=(H_phys+H_phys†)/2` を用いた。raw targetはH_physのままで、DF model差とraw表現差を別途記録した。R_refのsector保存から、その関数であるRθ・Γθにも保存を接続する。保存float入力の0や微小差を厳密保存の証明とは扱わない。

| 接続量（operator norm、数値診断） | 値 |
|---|---:|
| `||R_dict−R_ref||` full | 8.44194e-16 |
| `||(R_dict−R_ref)J||` | 8.42142e-16 |
| R_dictのHermitian defect | 3.74784e-17 |
| R_dictのsector外column norm | 5.66232e-17 |
| R_ref、H_refのsector外column norm | 各0.0（float値） |
| 完成headのsector外column norm | one-body 0、fragment0 8.35115e-16、fragment1 3.97252e-16、fragment2 2.33405e-21 |
| `e_conn=||S_dict J−J S_CAR,S||` | **7.78310e-15** |
| 実際のfull macrostep／参照macrostepのnorm | 0.9999960922088208／0.9999960922088235 |
| `||H_DF,ref−H_ref||` | 1.02422e-14 |
| `||H_ref−H_phys||` | 5.42118e-16 |
| 同じstateのH_refに対する残差 | 2.10769e-15 Ha |

macrostep接続は256×36のcolumn matrixで評価し、sector外成分も含めた。理想参照の保存と両macrostepのcontractionを仮定する追補のtelescoping式により、`q e_conn`を一回計上した。CAR head内部の積公式差は別項であり、generic/CAR再構成差を二重計上しない。

primary raw信号はfull 256次元macrostepの反復二乗から得ており、微小なsector外成分を捨てていない。保存参照36次元との信号差はm0で3.88851e-15、m17で6.69098e-11。数値診断として接続に支障を示す差は見られなかったが、norm・contraction・unitarityを区間包含で認定したものではない。

## 4. 固定θのoperator量

同じtail・head準備とspectral分解を二点で共有した。以下は保存入力に対する数値値であり、上向きに囲んだ保証入力ではない。

| 量 | 値 |
|---|---:|
| λ_R | 0.5832704246339983 |
| 抽出identity | 0.09261516589219072 |
| 確率和／B0 | 1.0／1.000004252545811 |
| γ = ⟨Γθ⟩ | **0.0007884706336364447** |
| vΓ | **2.573204308076602e-08** |
| Γθの最小固有値 | 0.00019538994061549255 |
| `||Rθ−R_ref||` | 1.1178612011976942e-06 |
| C_split（Γを含む） | 0.04183477932655248 |
| η_forward | 4.100455076920795e-05 |
| η_reverse | 4.10045507691925e-05 |
| 保存sector gap（数値診断のみ） | 0.23263021512018667 Ha |
| 資格付きgap下界 | **未取得** |
| finite-bit／native実装誤差 | **UNKNOWN** |

Γの負固有値をclampせず、一般frameをunitary projectionして修正していない。元Hの厳密基底状態を利用する研究promiseを用いる一方、保存浮動小数点vector自体を厳密固有vectorとして数値認証したとは扱わない。

## 5. 二点の信号

位相差は保存元energyの `exp(−iE0T)` に対する円周距離。energyは診断参照にだけ使用し、branch選択や事後的な位相補正はしていない。表は見やすく丸め、完全な値はraw JSONに保持した。

| 量 | m0 | m17 |
|---|---:|---:|
| q／T (Ha^-1) | 1／0.02 | 131072／2621.44 |
| raw信号 | 0.999045750029 + 0.043313405722i | 0.072735588359 − 0.103588721173i |
| raw半径 | **0.999984230759** | **0.126574440422** |
| raw位相 (rad) | 0.043327643867 | −0.958623902039 |
| 元energyに対する観測位相差 (rad) | **1.0510543e-07** | **0.013819560800** |
| 減衰参照 `⟨e^(T(−iH_ref−Γθ))⟩` | 0.999045745435 + 0.043313510725i | 0.074155394482 − 0.102576534744i |
| 減衰参照の半径 | 0.999984230717 | 0.126573962610 |
| leading参照半径 `a=exp(−γT)` | 0.999984230712 | 0.126573962505 |
| 観測位相差≤0.14 rad | 成立（数値診断） | 成立（数値診断） |

観測位相差が枠内にあることは、誤差diskから導く保証や全RPE成功の認定とは別である。

## 6. 誤差内訳・位相／半径の十分条件

未知の実装誤差を除く条件付き式の数値評価として、次を保存した。

`ε_dyn = Tδ² C_split + T||Rθ−R_ref|| + q(η_forward+η_reverse) + Tη_model + q e_conn + expm1(T||H_ref−H_phys||)`

| 項 | m0 | m17 |
|---|---:|---:|
| split（Γを含む） | 3.34678235e-07 | 0.04386694557 |
| arctan補正 | 2.23572240e-08 | 0.002930406067 |
| **head内部** | **8.20091015e-05** | **10.74909696** |
| model差 | 2.04844456e-16 | 2.68493725e-11 |
| full辞書→保存参照の接続 | 7.78310168e-15 | 1.02014670e-09 |
| raw targetへの表現差 | 1.08423646e-17 | 1.42113041e-12 |
| **ε_dyn（未知実装項を含まない）** | **8.23661370e-05** | **10.79589431** |
| 未取得のfinite-bit／native実装項 | UNKNOWN | UNKNOWN |

資格付きgap下界がないため、主経路はgap非依存の `d=vΓ T²/2+ε_dyn`。保存gapを使う `vΓ min(T²/2,2T/Δ)+ε_dyn` は診断列へ分離した。両方ともoperator norm・momentの丸めを包含した厳密上界ではない。

| 十分条件の数値評価 | m0 | m17 |
|---|---:|---:|
| gap非依存のvariance項 | 5.14640862e-12 | 0.08841462679 |
| gap非依存 d | 8.23661422e-05 | 10.88430894 |
| gap非依存 `a−d` | **0.999901864570** | **−10.75773497** |
| gap非依存 arcsin位相上界 (rad) | **8.23674411e-05** | **取得不可：d≥a** |
| gap非依存の正半径・0.14 rad条件 | 条件付き式の数値値は成立／未認証 | 不成立 |
| 保存gap診断のvariance項 | 5.14640862e-12 | 0.0005799333245 |
| 保存gap診断 d | 8.23661422e-05 | 10.79647424 |
| 保存gap診断 `a−d` | 0.999901864570 | −10.66990028 |
| 保存gap診断の位相上界 (rad) | 8.23674411e-05 | 取得不可：d≥a |
| 実装込み保証 | UNKNOWN | UNKNOWN |

負の `a−d` は下界式が有用な正値を与えないという意味であり、物理半径が負という意味ではない。m17ではarcsinをclampせずnullとして保存した。保存gapを仮に使用してもhead支配は変わらず、gap下界の資格化だけでは今回の十分条件の不足を解消しない。

head項はforward／reverseの各半sweepと理想head指数との差を別々にnormで評価してからq倍した値である。この三角不等式による扱いは、対称な全macrostep内での誤差相殺を利用しない。観測位相が良いことと現在の上界が大きいことを分けて扱い、headを含む動力学保証の保守性を次の検討対象とする。ただし、今回新しい改善定理の成立は審査・認定していない。

## 7. 保存K2信号との比較

`ea2c0915ba2f7e11a6dac58dc01cf1af3c0d0ff5` の保存m0／m17を参照し、元H/state/DF12/LD3/δ/rの対応を既存接続で確認した。元の科学SOURCE `5ef55f923e73e61bada383d0cfe98283e7984e9d` と保存結果の閲覧commitを区別する。K2の再simulationはしていない。

| 保存K2との比較 | m0 | m17 |
|---|---:|---:|
| 保存K2 raw信号 | 0.999027516781 + 0.043312613601i | 0.006648628117 − 0.009472258083i |
| 保存K2半径 | 0.999965980312 | **0.011572723492** |
| 今回K0半径 | 0.999984230759 | **0.126574440422** |
| K0／K2の円周位相差 (rad) | 1.62102969e-09 | 0.0001688822202 |

この差は既知K0とK2のlawの違いによる観測差である。C-α理論を使ったことによって、同じ既知K0の物理信号や回路価格が改善したという結果ではない。shot・価格・総費用の比や優位性は算出していない。

## 8. 精度・技術確認・実行資源

実行前に構文・CLI helpと[非科学的mock/schema test](../../../tests/test_calpha_h4_step1_mean_contract.py) **8件PASS**を確認した。科学moduleのimport・人工行列simulationを含まず、出力scope、上書き防止、例外保存、arcsin定義域、UNKNOWN保持を確認する技術testである。科学性能のPASSとは扱わない。

80／120桁の既存手法による追加診断は、保存する36次元参照stepの累乗演算に限定した。両精度の複素信号差はm0約3.26e-81、m17約9.08e-79。binary64累乗と120桁参照との差はm17で4.71e-13、位相差1.30e-12 radだった。full辞書や基本stepの構築誤差まで高精度認証したものではなく、厳密包含の根拠には使わない。

実行前の利用可能資源はCPU affinity32、RAM約62 GiB／available約53 GiB。GPUは利用可能と確認できなかった。小さい8／36／256次元行列には既存BLAS・vectorized minors・分解の共有・反復二乗を使い、このjobのBLAS／OpenMPを1threadとした。巨大回路やq回逐次作用は展開していない。速度倍率は測定・主張していない。Pythonは既存3.11.0rc1を明示使用し、共有環境・site-packages・他ユーザーjobは変更しなかった。

実行後にSOURCE不変更、二点scope、未取得実装項、旧4ファイルとarchiveのbyte一致、原入力hash、出力JSONの整合を確認した。科学実行SOURCEとrawは以後不変更とし、報告・索引・manifestの追記だけを行った。

## 9. 判断と次の担当

今回の完了範囲は、訂正条件による同じ平均のsector接続確認と、固定二点のmoment・信号・条件付き上界の数値評価である。分類は **保証改善が必要**。m0では条件付き十分条件の数値値が成立したが、m17ではhead項が支配し、二点そろったGO候補にはならなかった。包括実装保証・全18round成功・費用優位は未認定。

**次の判断担当はユーザー／GPT**。全18roundや会計へ進む前に、headを含む動力学誤差の保証を改善する限定レビューを行うか判断する。これは提案であり、次の重要レビューや追加科学計算の許可を取得済みとは扱わない。

**Codex**は今回の記録更新まで完了し、mandatory STOP。新しいレビュー・科学実行・r/LD探索・全round・shot／価格へ自動進行しない。
