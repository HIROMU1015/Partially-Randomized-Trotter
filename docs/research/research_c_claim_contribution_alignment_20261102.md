# 研究C：Claimと科学的貢献候補の対応・固定証拠照合

2026-11-02 JST。既存証拠の統合・文書整合・公開のための技術資料。
**資料Bの貢献候補A/B/Cは、資料Aの固定条件付きClaimに対応づけられる。ただし、独立新規性、収束済み既存法への優位、全law・厳密CA、他分子への一般化を支持する資格は追加しない。** 本書は二資料への対応索引であり、科学的重要性・中心claim・論文案・論文化GO/NO-GO・研究Cの継続／終了を採択しない。

## 1. 統合した原本とGit関係

| 資料 | 役割・固定版 | 原本blob SHA |
|---|---|---|
| [資料A：最新Evidence Map][A] | 科学結果・限定負結果・未資格事項。`43b304e628cc755a2640b1972f29da8efea6a435`、branch `codex/research-c-evidence-map-a-20261102` | `21b1b9d0fc4b4827589a5c743a7cfc9e068f897b` |
| [資料B：科学的貢献候補・論文構成案][B] | 先行研究との対応、候補3件・構成3案。`88afb1acc78dbc9b4af2bb5e149cd873a914b03b`、branch `codex/research-c-contribution-positioning-20261102` | `3cfdc178798d6f71e9a9b15bfd1686ddc90d4a15` |

両commitの直接の親・merge-baseはともに `46f1461d213ad73f8e187ef98e1491ef8b522aaa`。各差分は原本Markdown一件の追加だけで、共通起点には両原本ともない。追加pathは重複せず、元worktreeはcleanだった。新worktree `/tmp/prt-c-evidence-contribution-integration-20261102`、branch `codex/research-c-evidence-contribution-integration-20261102` を共通起点から作成し、`cherry-pick --ff` でAへfast-forward後、Bを競合なしで収録した。両原本の本文・blobは不変、元branchの履歴も変更していない。

統合branch内の入口は [A原本](research_c_evidence_map_updated_20261102.md)、[B原本](research_c_contribution_positioning_20261102.md)、本書。以降、**資料A/B、貢献候補A/B/C、Aの証拠記号**を区別する。例えばAの記号 `B` は別branchの資源寄与分析、Aの `P4` は構造実験であり、Codex-BやBの文献P4とは別である。

## 2. 比較条件・証拠参照の読み方

次の条件コードは資料A §2をそのまま再利用する。各表はこの条件と一組で読む。

| コード | 固定条件 |
|---|---|
| H4 | 保存直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ=2・sector36、system8/full256・ancilla込みfull512。一般signed実対称J・実直交Uの通常number CDF、元g補正κ、核scalar一回、元spin/frame順 |
| A8 | CDF8、同じqualified x0からγ0/.1各1 fit、元f*/M*・ε=1e-8 Ha。時間発展δはfit自体には非適用 |
| P4/16 | A8の最終二候補、元順LD2、partial ordered S2、s4/r16/K0、δ=.1・h=.025、m0…12 |
| P128 | 明示した候補、CDF8（CDF7行のみrank7）・元順LD2、partial ordered S2、s1/r128/K0、δ=h=.1、m0…12 |
| D1 | 明示した候補のκ＋8 frame全9因子ordered S2、s1、δ=h=.1、m0…12。PR split・r/K非適用 |
| R13 | CA=.00159 Ha・DF reserve=CA/8=.00019875 Ha、βtotal=π/6・βstat=π/12、failure=.01・axis=.01/26。Own-DF ground-state oracle、状態準備除外 |
| P0/N32 | Qiskit1.3.0/opt1/seed17、rz/sx/x/cx、approximation degree1、qubits_initially_zero=false、MPFR160 declared phase・位相fitなし。PRは各32独立finite-bit/PCG64・非paired価格cohort、Dは固定kernel。Fenced serial＋ancilla head/readout shell |

J/U表現・固定split・exact λとsmooth Mε・一般K写像の定義はB §2、各実験の科学SOURCE／結果／報告対応は[A §11.1][ASRC]を再利用する。下表の `A:記号/短縮SHA` は**科学SOURCE**（Tは技術公開、保存値整理に新科学SOURCEなし）、リンク先は結果収録commitの報告・JSONである。全SHA・SOURCE manifest・STOPへの導線はA §11.1に保持される。結果を科学SOURCEのtreeに収録したと推定しない。

特に、統合実行は科学SOURCE `980b514c81eb415fe053f558d7d3c0b8cfb5b9b4`／結果収録 `aced1647b81e8d28091058cdc7b6e3e1ee54896a`、新Dは科学SOURCE `5f27539ae0ab88181d9cfcd45badb54daf855c28`／結果収録 `38b8d0df39b8943178ecef611194819cff1c2169`。[資源寄与分析][ATTR]は `2639221a59713d431bc7c157854628bbc006a1bd` の別branch資料で、統合起点treeには存在しない。原本と同様、絶対固定リンクで参照し、本branchへ追加収録しない。

## 3. 貢献候補3件と中心主張の対応

「直接」は記載された条件・資格内の観測を支持する意味であり、新規性認定ではない。

| 貢献候補 | Bの中心主張候補 | 対応Claim | 証拠の役割 | 固定SOURCE・結果・報告 | 比較条件 | 確認済み結果 | 主張可能な範囲 | 不足証拠 | 整合性 |
|---|---|---|---|---|---|---|---|---|---|
| A：PR-tail DF目的・低減 | 物理DF予算を保ちながら、固定PR-tailの写像後smooth massを低減した | C01/C03/C04、限定対照C10。C11–C13・N09は補助／制約 | C01/C03/C04は直接。C10は固定RC2比較として直接。未収束・別終点は制約 | A:SA/e1085c95f、I/980b514c8、RC-A/150f904f6。各報告・JSONは§4 | H4/A8、追加軌跡もCDF8/LD2/γ.1。初期fitと旧PR対新点を分離 | 初期34.72%減、追加後旧PR比60.60%減。5受理・保存DF予算PASS、U一次未達 | 単一H4の有限endpointと固定目的比較。一般J/U・係数正則化は既知 | 共通一次資格・強度政策・誤差水準をそろえた既存法対照、初期値／分子移送、独立性の実質 | 一致。5更新の直接初期点・RC2の条件差を維持 |
| B：候補固有の量子資源評価 | PF/RTE・A・十分shots・W・complete価格・shellを接続し、λだけでは見えない変化を示した | C02/C05–C08、補助C04/C12/G01 | C02/C05–C08は直接。C12は別終点の補助。G01は別固定回路の保証資産 | A:OLD-C1/b1c735da3、OLD-M4/7dbabc786＋dcf7ab6da、I/980b514c8。G01は旧F2/F3 | P4/16とP128を分離、R13/P0/N32。G01はs4/r16各1trajectory | 新PRはPF増・補正RTE減・A増。W減・価格増、総RZ11.19%減 | Ideal/assumedとfinite32価格proxyの条件付き会計。既知式の候補固有評価 | 全law合成／implemented law、母平均・support、厳密CA。保証を中心に選ぶ場合のみ対応する追加証拠 | 一致。G01から最新law全体への資格移送なし |
| C：PR/PF資源tradeoff | PR内部改善は同じ新J/Uの強Dに対する主RZ優位を伴わず、CX depth交換が残る | 直接C05–C09。N01–N10は§4で役割別 | C09が同J/U比較の直接証拠。旧候補・人工・同値・技術NO-GOは一般不可能性の証拠にしない | A:I/980b514c8、D/5f27539ae、各構造証拠。N10は技術公開46f1461d2 | 同update005、D1対P128、R13/P0。N07は旧J/U、N06は人工 | PR/D総RZ1.97018542、CX depth .72499466。PR内部総RZは改善 | 固定H4・serialモデル・有限価格・宣言誤差laneでの点tradeoff | 価格悪化の因果機構、別条件での再現／crossover、hardware上の意味、母平均・全law | 一致。主RZ不利から全指標／PR一般のNO-GOを導かない |

## 4. 全24 Claimの追跡表

原本A §8のC01–C13、N01–N10、G01を各一行で保持する。「背景」はどの候補の中心主張も直接証明しない。SOURCE短縮名と結果リンクは§2の対応を使い、詳細条件・生結果をAから辿れるようにした。

| Claim | 貢献候補・証拠の役割 | 比較条件・対象 | 確認済み保存値／判定・支持範囲 | 固定SOURCEと報告／結果 | 制約・不足証拠 | 整合性 |
|---|---|---|---|---|---|---|
| C01 | A直接、B初期基盤 | H4/A8、同x0 γ0対.1 | λ2.20976183678→1.44242988558 Ha、34.7246%減、両DF予算PASS | A:SA/e1085c95f、[SA]・[SAJ] | 各1 fit、相対目的減少success≠一次収束・最適γ | 一致 |
| C02 | B直接、C歴史的補助 | A8終点、P4/16/R13/P0/N32 | Ideal shots53136→19574、条件付き総RZ83.08598%減 | A:OLD-C1/b1c735da3、[C1]。B1/B2 SOURCEはA/旧Evidence Map | 新selected/P128の改善率ではない。全law・母平均未資格 | 一致・政策差あり |
| C03 | A直接 | 旧19-call受理Uから新軌跡、CDF8/γ.1/cold J | 新5受理、10点数値適格、J inner/refinementと独立終端の成立 | A:I/980b514c8、[I]・[IJ] | 真のgradient boundなし、raw precision lossあり、同時一次未達 | 一致 |
| C04 | A直接、B入力資格の補助 | 旧PR対update005、H4/CDF8/LD2 | λ.568349854686 Ha、旧PR比60.60%減、sector＋guard8.46076622093e-9 Haで予算PASS | A:I/980b514c8、[I]・[IJ] selected.physical | 5更新の直接初期λは.725819387316。厳密DF enclosure・最適性なし | 一致・分母を区別 |
| C05 | B/C直接 | 旧新PR、P128/R13 | 全13roundで新PF増・補正RTE減・A増 | A:I/980b514c8、[PJ] rounds、[ATTR] §3 | Raw信号誤差も全改善ではない。PF/RTE数値reference≠全law | 一致 |
| C06 | B/C直接 | 同上、ideal lane | N16056→14048、W16631888→10487392。十分数と反復加重を分離 | A:OLD-M4/7dbabc786＋I/980b514c8、[WOLD]・[PJ] lanes | 実量子shots・普遍的下界・evidence保証ではない | 一致 |
| C07 | B/C直接 | P128/P0、新旧非paired32 | RZ平均14076.4375→19825.625、約40.84%増 | A:OLD-M4 native/dcf7ab6da＋I/980b514c8、[M4NJ]・[PJ] calibration | Finite-bit標本平均・descriptive SE。母平均／support／原因未認定 | 一致 |
| C08 | B/C直接 | P128/R13/P0、kernel＋shell | Ideal総RZ234117788135→207919150188、11.19%減 | 同上、[M4NJ]・[PJ] lanes、[ATTR] §6 | Evidence総費用null。因果ablationではない | 一致 |
| C09 | C直接、B対照の補助 | 同update005、D1対P128、R13/P0 | PR/D主RZ1.97018542062、CX depth .724994660275。両idealでも主RZ点差を維持 | A:D/5f27539ae、[D]・[DJ] comparisons | D数値pilotとPR反実lane。厳密CA・母平均優位・全指標支配なし | 一致 |
| C10 | A直接（限定RC2対照） | H4/A8、初期penalty寄与.1一致、full uniform L2 | 初期RC2 λ2.20976187195 Ha、DF PASS・optimizer success。旧PRより大きい固定点 | A:RC-A/150f904f6、[RCA]・[RCAJ] | 原論文solver／最適ρ／一次収束をそろえた勝敗ではない | 一致 |
| C11 | A制約 | 共通政策のRC2/PR、同元x0・CDF8 | λ8.54568712823/.954543284658、DF両PASS・一次両未達、復旧JSONも比較未資格 | A:RC-COMMON/e2f7dfd44、RC-RECは保存JSON修復、[RCREC] | 初期RC2・旧PR・update005と別。修復≠収束改善 | 一致・終点差あり |
| C12 | A/B補助（別終点） | C11終点、P128/R13/P0/N32 | PF/RTE、仮定shots、各32価格は取得済み。Kernel RZ20487/13246.375 | A:RC-B/d6b17b794、RC-SHOTS/e88fa599a、RC-NATIVE/14843a28f、[RCBJ]・[RCSJ]・[RCNJ] | 未収束。Interrogation総費用未算定で、本書でもW×価格を新計算しない | 一致・終点差あり |
| C13 | A制約・技術保全 | 別stable RC2、64受理／75点admission | F≈.02919→.01743、U₂=.0047620784で未達 | A:RC-STABLE/12e0079a2、[RSTJ] | 親PR STOPにより新終端DF評価なし。C11のDF PASSを移さない | 一致・軌跡差あり |
| N01 | C補助（限定反証） | Tail-P4対旧PR、H4/CDF8/P128 | 係数位置216→144でも総RZ275422425481.5、約17.64%増 | A:P4/4ac0e5f96、[P4J] | Projection、J/U、lawも異なる。Supportだけの因果・全疎化否定なし | 一致 |
| N02 | C補助（限定反証） | 独自fit CDF7、LD2/P128、対CDF8 | DF PASS、平均RZ17754.9375、総638880821593.5、主RZ点改善なし | A:R7/173ba2fd8、[R7J] | Rank単独効果・低rank一般の否定ではない | 一致・rank差あり |
| N03 | Cの価格機構claimへの制約 | 保存M4/P4各32、固定4-fold | 複合surrogateの登録MAE条件不成立 | A:SUR/030128649、[SURJ] | 64trajectory≠64DF。相関からη・介入・新DF性能を資格付けない | 一致 |
| N04 | Cの汎化claimへの制約 | 固定旧学習→第三DF32holdout | A-switch MAE369.787>B-overlap303.329、登録転移判定不成立 | A:THIRD/30436b142、[THIRDJ] | Aはmeanより良い。単一独立DFで汎化不存在は言えない | 一致 |
| N05 | 背景・数学的同値 | 厳密同一UのG群、一般対称J | 群内ΣJで通常CDF outer rank≤Gに帰着 | A:SHARED/T0876449d7、[SHARED] | 表現集合の同値。PF順・split・atom化・近似U共有の費用は別 | 一致。Cの直接性能証拠ではない |
| N06 | C補助（人工制約）、B実装背景 | 人工2/3軌道・既存P0 | Tail relativeでRZ748→780、1903→2039。中央局所減でもcomplete不変 | A:NATIVE/T0cb2e5ea9、[NATIVE] | 実H4全方式否定なし。Head/tail外側境界の価格未取得 | 一致・人工条件 |
| N07 | C補助（旧J/U） | 旧PR同J/U、D1対P128/R13 | Ideal/assumedとも13roundすべてD、MIXED_ROUND_DOMINATED | A:ROUND/79fa01076、[ROUNDJ] | 最新selectedの最適round配分・全指標・全政策の支配ではない | 一致・旧候補差 |
| N08 | 背景・既知法対応 | Euler/Parent/IS/return/相関の調査集合 | NO_DISTINCT_FREEDOMは追加の独立機構を未同定という分類 | A:LAW/Ta776aa84a、[LAW] | 性能negative・全sampler不存在・新規性否定の証拠ではない | 一致。Cの直接証拠にしない |
| N09 | A補助、資源claimへの制約 | C11のPR、U/head/tail tensor保持LP | λ.954543284658→.954432028127、約.0116555%減、保持gate PASS | A:LP/c40f81281、[LPJ] | 小さい肯定的な係数再分解結果。最新selected・native総資源・厳密LP保証ではない | 一致・別候補／問題 |
| N10 | 背景・技術準備判断 | 最新比較と既存価格機構証拠 | TECHNICAL_NO_GO_CURRENT_EVIDENCE、新cost-aware fit候補を選ばない | A:N/T46f1461d2、[NG]。新科学実行なし | 性能の不可能性証明・論文NO-GO・研究C終了ではない | 一致。正式判断へ昇格しない |
| G01 | B補助（旧固定保証）、最新lawは未資格 | A8各1trajectory、P4/16一h-step、full512、κH/polar Q | F2/F3-K1の対象限定Arb192 validated CAR上界PASS | A:F2/F3、3742ec10d／9b0f4d2b4、[A] §8.3/11.1 | 現update005/r128・全law・全13round・元physical入力・CAへ保証移送不可 | 一致・保証対象差 |

N05/N08/N10は背景・同値・技術保全、N09は限定的な肯定結果を含む。N01–N10を一括して「PRに改善余地がない直接証拠」と数えない。

## 5. 重要数値・候補identity・資格の照合

### 5.1 比較分母と保存値

下表は保存JSON・固定報告の転記照合で、新しい最適化・round集計・統計計算ではない。条件は§2、数値の削減率・比率は既報値を使う。

| 照合対象 | A/Bと固定結果の対応 | 判定・解釈 |
|---|---|---|
| 初期λ | [SAJ] fit_diagnosticsのbaseline/proposed exact_tail_mass_Ha＝2.2097618367788057/1.4424298855756243。exact_tail_mass_reduction=.34724644911133484 | A C01とB §5.1に一致。共通x0 γ0対.1の34.72% |
| 追加5更新 | [IJ] trajectoryの受理数5、initial_state λ=.7258193873157348。selected.id=update005、selected.physical λ=.5683498546862645 | 直接初期点は旧19-call受理点。旧PR1.44243を5更新の直接初期λとしない |
| 追加後λ比 | [IJ] references.old_PR_exact_lambda_Ha=1.4424298855756243、new_over_old_lambda=.39402251739914246 | A C04／B §4・5.1の約60.60%減と一致。通常CDF γ0比ではない |
| 物理・一次資格 | [IJ] selected.physical budget_PASS=true、rigorous_enclosure=null。selected.state J=MET/U=NOT_MET、first_order_met=false | 予算PASSと受理更新を、厳密enclosure・同時一次収束へ変更しない |
| W・価格・総費用 | [WOLD] affine.ideal_exact_native.W=16631888、[PJ] weighted_SE.kernel_reuse_weight=10487392。旧新価格14076.4375/19825.625、総RZ234117788135/207919150188 | A C06–08／B §4–5.2と一致。約36.94%減・40.84%増・11.19%減は別量 |
| Shell | 会計はWμ＋S。[WOLD]の旧S_RZ=56196。新S_RZ=49168は[ATTR] §6の既存残差照合値で、独立保存fieldではない | Shellを落とさず、総shotsをWに置換せず、r128や校正数32を再乗算しない |
| 初期資源83.09% | [C1]のs4/r16/K0、初期二候補の条件付き比較 | C02だけの値。P128の11.19%や最新D比較へ転用なし |
| 同一新J/U | [IJ] selected.physical candidate_sha256と[DJ] selected_exportはともに938340b8…6e56e | 原本A §2.2／B §4.1の同じupdate005。SHAは文字列・Git記録照合、NPZ decodeなし |
| 最新D/PR | [DJ] numerical.costと[PJ]の総費用にD105532782860・PR207919150188。comparisonsの主RZ比1.9701854206178373／CX depth比.7249946602754033 | 主RZはD低費用、CX depthはPR低値。両idealでも主RZ比は同じ |
| 旧D benchmark | [I]の新PR／旧D≈2.04191は旧J/U Dとの比較 | A §6／B §1・4で歴史的値として一致。同一新J/U比には使わない |
| RC2候補 | 初期17-call RC2、共通政策RC2/PR、stable RC2、新selectedをA §2.2の別identityとして保持 | C10/C11/C12/C13を一つの終点に統合しない。未算定RC2総費用を新規作成しない |

### 5.2 資格を変えないこと

| 区別 | 統合後の扱い |
|---|---|
| 数値PF/RTEと全law | 候補固有referenceの数値観測。有限回路correctnessを一様なlaw誤差上界へ昇格しない |
| Ideal/assumedとevidence | [PJ] evidenceはUNQUALIFIED、shots/総資源null。Idealは反実仮定、assumedは各.001Aの感度仮定。未知誤差を0にしない |
| 十分shotsと実測 | Hoeffding等の宣言モデル内十分数。実量子測定数・最小必要数としない |
| Finite32と母平均 | 非paired有限校正の平均／descriptive SE。同じ平均を全13roundで共有し、SEはW倍。新しいCI・有意差・母平均順位なし |
| CorrectnessとCA | Native/full512/adjoint/X/Y数値PASS、D level-B数値pilotと、actual_CA_achieved=null／厳密CA未確立を両立させる |
| 受理・停止と収束 | 5受理とJ一次METは保存。U一次未達・有限step不成立を最適性や改善不能に読み替えない |
| 技術NO-GOと研究判断 | N10は価格改善機構の現証拠に関する準備判断。新規性・投稿・研究終了／継続の正式採択ではない |

## 6. 先行研究との重複・方法差を証拠へ接続する

一次文献の比較は**B §1.2・2・3を再利用**し、今回網羅レビューをやり直していない。リンクはRC-DF 2212.07957v3、SCDF 2403.03502v1、PR/RTE 2503.05647v2、CDF/QFD 2104.08957v1の固定版を維持する。

| 整合対象 | Bから保持する区別 | Aの証拠との接続・未確定 |
|---|---|---|
| RC-DF一般J/U | 空間Z=J、一般対称係数と軌道回転は共通。Spin-orbital Kへの変換は別 | C01/C03は新表現classの証明ではなく、この表現上の目的・数値結果 |
| 正則化 | Entrywise L2、weighted L1二乗、写像後tail smooth massを区別。非平滑・符号制約下の条件付き同値あり | C10–C13の対照は限定的。COMPARABLE_DISTINCT_OBJECTIVEを独立新規性へ昇格しない |
| SCDF | 一般V前段と、rank-one・symmetry shift付き採用式を分ける | 現Cは一般signed J・shiftなし・元κ固定。現RC2比較をSCDF全体との公平比較と呼ばない |
| PR/RTE | DF/基底最適化、normalization、減衰とsampling・費用会計は既知 | C02/C05–C08は候補固有の追加数値情報。Shots×価格の式自体を新理論にしない |
| CDF/QFD | 一般CDF、係数inner／軌道outer、基底融合等は既知 | C03はこの実装で物理適格に前進した証拠。一般solverの独立性や収束保証ではない |
| 科学的貢献 | 方法差、実装差、科学的独立性・重要性は別判断 | 同じ要素を用いることだけで数値／資源比較の情報価値は否定しない。一方C01–C13だけでは独立新規性を正式認定できない |

## 7. 矛盾・過大主張・未解決事項

**照合した中心数値・条件・資格について、未解消のA/B間矛盾は確認しなかった。** 以下は条件差・時点差・未取得証拠の整理であり、原本の修正で隠さない。

| 対象箇所 | 問題になり得る読み方 | 統合した記述・修正提案の扱い |
|---|---|---|
| A C01/C04、B §5.1 | 34.72%と60.60%、直接5更新の開始点の混同 | §5.1の比較分母を必ず添える。原本は整合しており変更不要 |
| A §6、B §1/4と旧[I]/[ATTR] | 新PR／旧D≈2.04191を同一J/U比較に使う | 最終同J/U点比較はC09の1.97018542。歴史的報告の当時の条件は保持 |
| A C10–C13、B §5.1 | RC2比較が未実施、または最適RC-DFに勝利済みとする | 比較は実施済みだが、未収束・強度・solver等の制約が残る。「公平比較未完了」の具体的意味を保持 |
| A G01、B §4.3/5.2 | 最新selectedの全law保証が成立したとする | 別固定targetの保証資産として残し、最新evidence laneのnullを保持 |
| A §9、B §1.1と古い総論 | 「総費用未実施」を条件付き会計も存在しない意味にする | 条件付き点総費用は実施済み、厳密CAを保証した最終総資源比較は未実施と区別 |
| A/B末尾の引渡し | 相手が未完了という当時の未来形を現状態とする | 今回両原本を収録・対応済み。原本の時点記述は改訂せず、本書を現在の入口にする |

新規性・一般化・価格の因果機構・母平均・全law・厳密CAは、A/Bの**共通の未解決事項**である。矛盾を解消したことや文書検査の成功を、その科学的資格の取得と数えない。別条件の同値性や技術判断から全PRの不可能性を導く過大主張も採用しない。

## 8. 論文構成3案と支持証拠・不足証拠

B §6の案を選択せず、利用できる証拠だけ対応づける。

| Bの構成案 | 中心RQ・現在書ける範囲 | 直接／補助証拠 | 中心claimを強める場合の不足 |
|---|---|---|---|
| 案A：PR-tail CDF最適化 | 物理DF予算内で固定tailを減らせるか。方法定義・既知正則化との関係・H4有限点の実証まで | 直接C01/C03/C04/C10、制約C11–C13。C05–09は下流評価 | 共通一次資格・強度／誤差政策の近接対照、方法の実質的独立性、未使用初期値／分子への再現 |
| 案B：同一DFのPR/PF資源比較 | 同じ新J/U・政策で、RZとCX depthのどの交換があるか。固定serialモデルの比較まで | 直接C05–C09、C02/N07は別条件の補助。G01は保証範囲の背景 | 別条件／hardwareの適用域、必要なら母平均の不確かさ。全law／CAを主張する場合はその資格 |
| 案C：mass低減と総資源削減の不一致 | λ/A・Wの改善と価格／PF悪化が共存する具体例。会計的説明と限定負結果まで | 直接C05–C09。補助N01/N02/N06/N07、制約N03/N04。N05/N08/N10を直接証拠にしない | 価格を操作する因果機構、独立条件で成立する設計則・予測。現相関を新surrogate採択根拠にしない |

現在の限定記述をまとめるだけなら、すべての未解決事項を先に解消する必要はない。方法の一般性・母平均順位・実CA保証など**どの主張を強めるか**によって必要な追加確認は異なる。B §7の不足証拠表を再利用し、一括研究計画を自動採択しない。

## 9. 共有入口・保全・検証・次の担当

共有入口にはA/B/本書へのリンクだけを最小追加する：`PROJECT_MAP.md`、`docs/README.md`、`docs/research/研究概要・現状.md`。既存の状態説明は同J/U比較・技術NO-GOの射程を保持しているため、過去本文・他Trackの状況・研究判断は書き換えない。

実施した照合は、A/B公開本文とGit blob、24個のID、対応表、科学SOURCEと結果収録ref、既存JSONのscalar・qualification・候補hash、GitHub path、Markdown、差分と公開本文に限定した。既存のA §11・B §8の証拠参照体系を再利用し、新しい科学runner・検証pipeline・管理framework・承認証明を追加しない。新規科学計算・保存NPZ/QPY decode・再fit・PF/RTE/shots/native再計算・新統計解析・RC2総資源の新算定・詳細科学レビューは0。

A/Bの各原本blobを保持し、統合branchの共通起点からの変更を、原本二件追加＋本書追加＋上記共有入口三件のリンク追加に限定する。`VALIDATION_STATUS.md`、validation manifest、研究ノート、旧Evidence Map、科学SOURCE/raw/科学契約/grant/ledger/marker/STOPは不変。元branchのHEADも保持し、force push・mainへのmergeはしない。保全確認は当該Git差分の範囲であり、別担当の全worktreeを監査・変更しない。

正式共通運用指示2026-10-25版は依頼元workspaceの原本§8/8.1/8.4/10/10.1/10.3を確認した。独立資料確認を並列化し、共有環境・他jobの変更や恣意的なCPU/RAM/worker capはない。Commit/push後、remote branch SHA・commit/tree・A/B原本blob・本書と三入口の公開本文／リンクを照合して完了する。

**次はユーザー／研究Cが、中心に据える貢献候補、現証拠での主張範囲、判断を変える追加確認、論文化・現在系列の区切りを判断する。** Codexはその後に具体的な依頼があれば必要な技術準備を担当する候補であり、新科学実行や重要レビューの開始は別途の明示許可が必要。本統合作業から自動進行せず、公開照合後**mandatory STOP**。

## 固定出典

[A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/43b304e628cc755a2640b1972f29da8efea6a435/docs/research/research_c_evidence_map_updated_20261102.md
[B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/88afb1acc78dbc9b4af2bb5e149cd873a914b03b/docs/research/research_c_contribution_positioning_20261102.md
[ASRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/43b304e628cc755a2640b1972f29da8efea6a435/docs/research/research_c_evidence_map_updated_20261102.md#L239
[ATTR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2639221a59713d431bc7c157854628bbc006a1bd/docs/research/research_c_pr_resource_contribution_analysis_20261102.md
[SA]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_pr_aware_df_stage_a_x0_v2_execution_20261027.md
[SAJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_pr_aware_df/2026-10-27/run01/result.json
[C1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_pr_aware_df_stage_c1_execution_20261027.md
[I]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/docs/research/research_c_pr_integrated_execution_20261102.md
[IJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_integrated/2026-11-02/run01/result.json
[PJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_integrated/2026-11-02/run01/C/result.json
[WOLD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_pr_aware_df_M4/2026-10-28/run01/C/result.json
[M4NJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_pr_aware_df_M4_native/2026-10-28/run01/result.json
[D]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/38b8d0df39b8943178ecef611194819cff1c2169/docs/research/research_c_same_ju_strong_d_execution_20261102.md
[DJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/38b8d0df39b8943178ecef611194819cff1c2169/artifacts/research_c_same_ju_strong_d/2026-11-02/run01/result.json
[RCA]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_regularization_comparator_execution_20261101.md
[RCAJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_regularization_comparator/2026-11-01/run01/result.json
[RCREC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_common_convergence_export_repair/2026-11-01/run01/result.json
[RCBJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_rc2_pr_stage_b/2026-11-01/run01/result.json
[RCSJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_rc2_pr_conditional_shots/2026-11-01/run01/result.json
[RCNJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_rc2_pr_native/2026-11-01/run01/result.json
[RSTJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_j_u_stability/2026-11-02/execution_summary.json
[P4J]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_df_structural_pr/2026-10-28/run01/result.json
[R7J]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_cdf7_native32/2026-10-30/run01/result.json
[SURJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_df_basis_surrogate/2026-10-29/run02/result.json
[THIRDJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_df_basis_third/2026-10-29/run02/result.json
[SHARED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_shared_basis_df_design_20261029.md
[NATIVE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_pr_native_structure_optimization_design_20261030.md
[ROUNDJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_roundwise_det_pr/2026-10-30/run01/result.json
[LAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_sampling_law_freedom_survey_20261030.md
[LPJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/run01/result.json
[NG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/46f1461d213ad73f8e187ef98e1491ef8b522aaa/docs/research/research_c_pr_native_cost_improvement_feasibility_20261102.md
