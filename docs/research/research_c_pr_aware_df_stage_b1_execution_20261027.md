# 研究C：PR-aware DF Stage B1・141-call限定科学実行

2026-10-27 JST。Repository：HIROMU1015/Partially-Randomized-Trotter。

**固定B1契約を無変更で一度実行し、141/141 COMPLETE、correctness FAIL0、中断・修正・再起動・retry/resume0。両armの全13 roundが`POTENTIALLY_FEASIBLE`だった。** 登録したideal finite-RTE／sector36数値pilotでは、ProposedはBaselineより減衰係数Aが大きく、補正RTE誤差は小さかった。PF誤差はほぼ同程度で、回路単価・shots・最終CA・総資源の優位性は未評価／未資格である。

## 固定SOURCEと実行条件

| 項目 | 固定条件 |
|---|---|
| 科学SOURCE | `7da0fdb3ff7c2e5e22c7d8658f6980077993c5d9` |
| 数値実装SOURCE | `366cb090a2c05c6d7ffeafdba2443bfc64b8a035` |
| Scope | `C_PR_AWARE_DF_STAGE_B1_MECHANISM_V1` |
| [契約](research_c_pr_aware_df_stage_b_contract_draft_20261027.json) SHA256 | `e07e6babb67cd752db0f9ce6bae90b0ddd4d3a7c91b8743f6b9138134d109163` |
| [SOURCE manifest](../../artifacts/research_c_pr_aware_df_stage_b/2026-10-27/technical/SOURCE.json) SHA256 | `767b7ffeabe8ad7a402384b664a787e4d6f236e6458baf191dd31f1908cc4171` |
| 入力と分子 | 保存直鎖H4・隣接1Å・STO-3G、元input SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| sector | Nα=Nβ2、36次元 |
| 比較 | Stage A最終Baseline／Proposed、通常number CDF8、元frame順prefix LD2 |
| Baseline final.npz SHA256 | `f3a3141059e29f79f39c3f6d3721562abfa7221ad97ce06a118331f6583e1a02` |
| Proposed final.npz SHA256 | `f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227` |
| PF／RTE | partial symmetric S2、δ=.1、s4、r16、K0、m0..12 |
| 元規約 | 元g補正κ、核scalar一度、spin／frame順を維持 |
| State／精度条件 | 各DF自身のsector ground-state oracle、状態準備除外、CA=.00159 Ha、DF reserve=.00019875 Ha |
| phase room | βtotal=π/6、βstat=π/12、既存十分条件 εPF＋εRTE≤sin(π/12)、A>0 |
| Law | ideal finite-RTE IID reference。samplingによる平均推定はしない |
| fresh output | `artifacts/research_c_pr_aware_df_stage_b/2026-10-27/B1-run01` |

Stage A公開結果`9ae5942d02572498b6ee1b5f309ca504ed58c6ef`のJ/UとDF norm＋guardを固定して再利用し、新fit・DF norm再計算・frame探索は0。旧CDF/XDFのPF operator／誤差やsampling分布を流用せず、この新しい2 armのreferenceを構成した。

ユーザーの明示許可に基づいて実行した。契約の歴史的`UNAPPROVED_PROPOSAL` metadataは維持し、承認証明書・承認原文保存・新grantは作成していない。中断時修正の追加許可を使う事態は発生せず、登録SOURCE・条件のまま完了した。

## Preflight・正しさ・ground/alias

[preflight](../../artifacts/research_c_pr_aware_df_stage_b/2026-10-27/execution_preflight.json)でSOURCE39、保存input8、元input bytes・export header・Stage A result/SOURCE/STOP、契約／manifest、package版、fresh outputを確認した。既存evaluation env（Python3.11.1、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0等）は変更していない。準備済み人工56 testsは再実行せず再利用した。

新J/Uごとにraw CAR base＋8frameを構成し、18 factorのHermiticity／general-K、両sum、両tail atom／identity、Stage A mass照合が全PASS。両tailは216 nonidentity atomsで、正規化・符号・元順を維持した。identityは決定論位相へ抽出しλ_Rに含めない。

| 指標 | Baseline | Proposed |
|---|---:|---:|
| 新lawのλ_R [Ha] | 2.2097618367788057 | 1.4424298855756241 |
| Stage A保存exact mass [Ha] | 2.2097618367788057 | 1.4424298855756243 |
| tail identity coefficient [Ha] | .4793928496324132 | .3377477315864882 |
| 保存DF norm＋guard [Ha] | 1.0297636237465247e−8 | 1.229561498918692e−8 |
| own-DF ground energy [Ha] | −2.1663874524547952 | −2.166387443451351 |
| eigenpair residual | 2.8265151640433296e−15 | 2.3552421102079465e−15 |

λ_Rは保存massと固定allowance内で一致し、κ／scalarを補正し直すことなく、Stage Aで抑えた量が既存PR lawのmassへ接続した。physical groundは−2.1663874486347625 Ha、residual1.7642254846760187e−15。3 groundのnormalization／登録alias window[-10,10] HaがPASSし、base alias marginは1.6179938779914944 radだった。これらのground値を最終energy推定精度と扱わない。

全数値referenceはspin sector36内の**完全な因子**を用いる。Gaussian gate単体のsector射影でcontrolled回路を代用したものではない。B1では実QuantumCircuitのbuild/compile・full controlled検証は0である。

## 全26 roundのPF／補正RTE／減衰

以下のoperator誤差は無次元、λ_Rと保存DF誤差はHaであり、単純加算しない。時刻h=.025、反復数n=4×2^m、総時刻t=.1×2^m。

| m | Baseline PF | Proposed PF | Baseline corrected RTE | Proposed corrected RTE | Baseline A | Proposed A |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.900319e−6 | 3.883396e−6 | 1.190213e−4 | 9.341410e−5 | .99961859 | .99983747 |
| 1 | 7.779582e−6 | 7.745791e−6 | 2.378477e−4 | 1.866724e−4 | .99923732 | .99967496 |
| 2 | 1.539242e−5 | 1.532527e−5 | 4.740882e−4 | 3.720711e−4 | .99847522 | .99935002 |
| 3 | 2.950413e−5 | 2.937293e−5 | 9.353903e−4 | 7.340480e−4 | .99695277 | .99870047 |
| 4 | 5.027433e−5 | 5.002845e−5 | .0017764294 | .0013937618 | .99391482 | .99740263 |
| 5 | 6.060251e−5 | 6.013802e−5 | .0030008294 | .0023531819 | .98786666 | .99481201 |
| 6 | 8.190995e−5 | 8.093487e−5 | .0041147715 | .0032268682 | .97588055 | .98965093 |
| 7 | .0001546709 | .0001531835 | .0054351656 | .0042493507 | .95234284 | .97940896 |
| 8 | .0002818482 | .0002814961 | .0101070448 | .0078757529 | .90695689 | .95924192 |
| 9 | .0005611409 | .0005607797 | .0194292048 | .0150903526 | .82257079 | .92014506 |
| 10 | .0011213758 | .0011207493 | .0390259627 | .0302377068 | .67662271 | .84666693 |
| 11 | .0022402667 | .0022395166 | .0790368434 | .0609542702 | .45781829 | .71684489 |
| 12 | **.0044769543** | **.0044762458** | **.1643051229** | **.1256120553** | **.20959759** | **.51386660** |

両armの13/13 roundは登録PF＋補正RTE十分条件と正のAを満たし、候補分類は両方`POTENTIALLY_FEASIBLE`。INFEASIBLE／UNRESOLVEDセルは0だった。

PFは`||U_PF(h)^n−exp(−iH_DF t)||₂`、補正RTEは`||U_corrected(h)^n−U_PF(h)^n||₂`。比較先は各armの近似DF Hamiltonianであり、物理DF誤差はStage Aの保存reserveとして別管理する。λ_R低下が新lawの減衰改善へ伝わることを観測した一方、PF誤差はほぼ同程度だった。今回の単一paired分解から、J/U・非可換性・head/tail負担を完全に因果分離したとは主張しない。

各roundのraw complex signalは既存referenceで`A <ψ_DF|U_corrected(h)^n|ψ_DF>`を評価し、[26行CSV](../../artifacts/research_c_pr_aware_df_stage_b/2026-10-27/rounds26.csv)と各arm/mのJSONへ保存した。m12はBaseline `.03234436455290321 + .21226983038598965 i`、Proposed `.07880495702931041 + .5174097923317142 i`。AはRTE normalizationの減衰係数であり、有限RTE誤差を含むこの信号の絶対値と同一視しない。量子測定値やsampling推定値ではない。

binary64のoperator数値pilotで、厳密丸めenclosure・global合成誤差・implemented-law誤差・CA達成ではない。未知の合成／実装law／cost support／shots／総費用はnullを維持した。登録十分条件のPASSからCAや資源優位を認定していない。

## call・資源・保全

input/evidence/context/physical4、fit load＋model4、raw/factor gate36、sum/law/tail/mass8、ground/alias6、kernel2、exact26、round26、classification26、arm summary2、比較1＝**141/141 COMPLETE**。正しさFAIL0、retry/resume0、修正／技術再起動0。

新fit・optimizer・DF norm再計算・sampling／calibration samples・native build・compile・shots／総資源計算は0。旧18-call STOPとStage A V2の30-callもそのまま保存し、新B1の141枠だけを使用した。

実行前はaffinity32 CPU、空きRAM約52.75 GiB、load約1.04、CPU/memory pressure avg10/60/300=0。正しさgate完了後に2 armを独立並列評価し、既存planで小行列BLASを自process内1 threadに協調した。CPU/RAM/worker/wallの人工上限追加、共有設定・venv・他job操作は0。runner wall2.06333秒、CPU2.20365秒、ru_maxrss359892 KiB。速度向上率は測定していない。

実行後SOURCE39・saved input8・元input・契約／manifestのhashは不変で、Stage Aと旧technical証拠の103ファイルも不変：[postflight](../../artifacts/research_c_pr_aware_df_stage_b/2026-10-27/execution_postflight.json)。旧H2/H4/H6、研究A/B、C4、Gaussian singletonのSOURCE／raw／grant／ledger／marker／STOPは変更していない。

- [全raw・atom/確率catalog・ground・reference・call journal・STOP](../../artifacts/research_c_pr_aware_df_stage_b/2026-10-27/B1-run01/)
- [実行summary](../../artifacts/research_c_pr_aware_df_stage_b/2026-10-27/execution_summary.json)
- [全26セルCSV](../../artifacts/research_c_pr_aware_df_stage_b/2026-10-27/rounds26.csv)

## 次の最小判断

B1では、Stage Aのtail mass削減が同じPR政策でより大きいAと小さい補正RTE誤差へ伝わる例を得た。両候補とも登録十分条件内なので、共通Gaussian／P0でのB2 native費用校正を検討する対象は得られた。ただしProposedで決定論headの費用が増えた可能性、U変化によるGaussian遷移価格、whole-trajectory単価は未測定である。

次は**研究C／ユーザー**がB2の実施価値と最小範囲を判断する。**Codex**は具体的指示後に、保存B1 reference・新J/Uと共通native部品を再利用し、build/compile・独立controlled correctness・校正sample・不確かさを整合させたB2契約を準備する。今回の許可でB2や32 samples/arm案を実行していない。

全law合成上界、finite-bit law差、費用support、shots・共通CA・期待総RZ/CX/depth、科学的新規性は未資格。Stage C・追加fit・条件探索・重要な科学レビューへ自動進行しない。成果公開・remote照合後、**mandatory STOP**する。
