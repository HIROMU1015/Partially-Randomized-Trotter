# H4 DF10 RPE接続：D1保存scalar評価

2026-11-10 JST。**T=.8の固定PF/PR各1候補について条件付き数値pilotを完了**した。採用設定はh=.1、βtotal=.4、βstat=.2、全13round、各軸α=.05/26。十分shot数はPFがX/Y各705、PRが各977。実装/law誤差と厳密ground同定は未資格で、actualのNはnull、全RPE成功と総費用Gは未評価である。[機械結果](result.json)が数値の正本。D2/D3は未開始。

## 科学条件・固定候補

H4、linear 1.00 Å、STO-3G、保存tol-only 1e−8の観測DF10、Nα=Nβ=2。targetは保存DF-Hの同sector基底状態エネルギー。保存数値stateをgroundと同定する条件付き、準備費P=0、主資源compiled RZ、測定X/Y。化学的精度は元config.CAの0.00159360010199040 Ha、αtotal=.05はPF/PRそれぞれの全RPEへの上限とする。元の有限時間αaxis=.025は流用しない。

| arm | 固定candidate ID | 構成、T=.8 |
|---|---|---|
| PF | `49709717a6c3d405c60d` | B0、S2、prefix7、qPF=8、δ=.1 |
| PR | `14bd81a375400c9b0305` | S2、prefix1、qPF=8、δ=.1、R_outer=4、K=2 |

PFのdiscardした項も保存DF-Hに対するtotal signed biasへ含めた。Full PFや新候補の選択は行っていない。qPFは分割数、qRPE=2^mはRPE倍率であり、D1で双方8になることは同じ定義を意味しない。回路価格は元symmetric_directional controlの完成cosine/sine Hadamard wrapperに対応する。Track DのStrong/Reference比較をPF/PR比較へ読み替えない。

## prior・base time・horizon

保存係数の表現H=c+dΓ(h)+Σλ_l dΓ(G_l)^2を使った。Hermitian h/GのGershgorin区間を4粒子区間へ拡張し、符号付きλごとのsquare区間を加えて下界を作った。上界は保存sectorの先頭occupation `[0,1,2,3]` の固定determinantに対する変分式で改善した。

`<dΓ(G)^2> = (Σoccupied G_pp)^2 + Σ(p occupied,q empty)|G_qp|^2`

係数binary64を正確な有理数として演算し、採用区間のfloat端点を外向きに丸めた。full Fock Hamiltonian、trial state vector、eigensolveは作っていない。E0/state energyはprior関数の引数になく、区間やdeterminantを正解エネルギーに合わせて選んでいない。この解析的係数境界の資格は保存Hermitian DFモデルに限定され、数値stateの厳密ground証明にはならない。

| 項目 | 採用値・判定 |
|---|---|
| 事前ground区間 | [−4.994249624417489, −2.0985459369977186] Ha |
| 既存principal decoder条件 | h max(|Emin|,|Emax|)+βtotal < π、energy shift=0 |
| h=.8の余裕 | −1.2538070459441983 rad：このprior/decoderでは未適格 |
| h=.1の余裕 | 2.2421676911480444 rad：適格、h=.1を採用 |
| βtotal / βstat | .4 / .2 rad |
| 既存horizon関数の結果 | 最大index M=12、**13round m=0…12**、qmax=4096 |
| horizonのenergy resolution | .0009765625 Ha（直前powerは.001953125 Ha） |
| 各round各軸のα | .05/(2×13)=.0019230769230769232、全26軸の保存和=.05 |
| D1の位置 | T=.8=h×2^3、m=3、qRPE=8 |

13roundは旧13/18roundの固定流用ではなく、今回のεE/h/βから既存`required_rpe_round_horizon`で求めた。horizonのtarget_metは解像度条件のみ。初期m=0,1,2を含む他roundのsignal条件は未評価で、全成功の意味ではない。h=.8の判定は現在の保守的priorと無shiftの既存decoderに対するもので、より鋭い正当化済みpriorや異なるshift条件でも不可能という結論ではない。

H6の[準備REPORT][H6]も確認したが、同報告のh=.8案はprior/alias未確定だった。共通化できる理論/APIは維持し、H4のalias条件に従って.1を選んだ。H6のprior/N/信号を今回評価していない。

## reference・normalization・誤差disk

保存referenceは定数項を含む同DF-Hの時間発展に対応する。保存solver energy −2.1663874486347634 Haによる`exp(−i E_state T)`との整合性を診断した。これはpriorやdecoderのbranch選択には使用しない。保存residual=1.9513566490707353e−15 Ha、T×residual=1.5610853192565883e−15を数値guardとして加えたが、ground同定や厳密固有対の証明にはしない。

δcorr = hypot(|bias_X|+u_X, |bias_Y|+u_Y) + |zref−exp(−i E_state T)| + T×residual。
既存uは両候補・両軸とも1e−9であり、経験的allowanceとして保持した。
A=exp(−log B)、zraw≈A zcorrを照合し、保存log_B_margin由来のraw余裕とclosureを加えた。Bをshotや価格へ二重乗算していない。

| 保存量・新scalar診断 | PF | PR |
|---|---:|---:|
| referenceのsolver位相からの符号付き差 rad | −3.56080e−16 | −1.28737e−15 |
| reference complex offset | 4.51828e−16 | 1.31627e−15 |
| 保存total signed bias X | 7.644963691968987e−5 | 1.058214274974878e−4 |
| 保存total signed bias Y | 1.2473899963061363e−5 | 1.7274558005930807e−5 |
| corrected disk δcorr | 7.746175194782574e−5 | 1.0722327670097748e−4 |
| log B | 0 | .16343103470600182 |
| B / A | 1 / 1 | 1.1775441427090354 / .8492250640382952 |
| raw/corrected closure | 0 | 2.248030287623391e−15 |
| normalization raw余裕 | 3.425081278668784e−11 | 8.515641510203661e−13 |
| observed corrected radius | .9999999585401248 | .9999999523107621 |
| observed raw radius | .9999999585401248 | .8492250235393971 |
| observed raw位相差 rad | −7.746059448741721e−5 | −1.0722212065345036e−4 |

両referenceの観測半径1.0000000000000002は丸めを含む保存数値であり、1より大きい物理的信号を主張しない。位相誤差の対真ground成分は未取得のため`unconditional_ground_identification_error=null`とした。

## 条件付き十分Nと資格

既存geometry/Hoeffdingを変更せず用いた。raw diskをδrawとするとρlower=A−δraw、βdet=asin(δraw/A)、εaxis=ρlower sin(βstat)/√2、Naxis=ceil[2 log(2/αaxis)/εaxis²]。条件付きideal laneはreference定義の決定論PF/finite-RTE PR、fresh IID ±1測定、exact-nativeを仮定する。量子shotやtrajectoryは実際には生成していない。

| D1の条件付き判定 | PF | PR |
|---|---:|---:|
| raw error disk δraw | 7.746178619863852e−5 | 9.10566948743476e−5 |
| raw radius下界 | .9999225382138014 | .8491340073434209 |
| βdet rad | 7.746178627610455e−5 | 1.0722327790918652e−4 |
| βdet+βstat rad | .2000774617862761 | .2001072232779092 |
| εaxis | .1404695491538685 | .11928671134447849 |
| 十分N_X / N_Y | **705 / 705** | **977 / 977** |
| ideal numerical pilot | CONDITIONAL_ROUND_PASS | CONDITIONAL_ROUND_PASS |
| actual implementation N_X / N_Y | null / null | null / null |

βtotal=.4以内なので、D1では複雑なα/β再最適化を要しない。これらは採用ルールによる十分Nであり最小Nではない。統計Hoeffdingの理論と、入力diskが数値pilotであることを区別する。実装誤差/law距離は未取得のままで、actual laneはUNRESOLVED_REFERENCE_OR_ERROR。ideal条件のゼロ誤差をactualの未知誤差へ流用していない。保存stateのground資格、u、reference/残差の厳密包含も不足しており、rigorousなエネルギー成功保証はない。全roundにαを予約したことも、未評価roundの成功資格を与えない。

## 保存compiled価格との対応

以下は[元主研究155529e2][MAIN]から準備時に保存した単価の転記。nは価格を作った古典的cohort数で、量子Nへの乗数ではない。探索/独立確認をpoolしない。

| arm/cohort | n | C_RZ,X = C_RZ,Y | 価格の資格 |
|---|---:|---:|---|
| PF deterministic | 1 | 44411 | 直接compiled決定論値 |
| PR exploration | 9 | 16857.11111111111 | 直接compiled有限標本平均 |
| PR confirmation | 33 | 16947.78787878788 | 直接compiled独立有限標本平均 |

candidate全tuple、T、Hamiltonian/state/snapshot、wrapper/compilerの保存bindingを照合した。compiler hashは`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。保存profile hash `4e32bb9ba57c38280562390100ebf916e9858218534504ead1519625a609bd75`はreference-onlyの来歴であり、今回新しいcompiler profileの実証とは扱わない。元runtime・compiler詳細は[準備SOURCE索引](../h4_df10_rpe_preparation_20261110/SOURCE_INDEX.md)のscientific/completion provenanceへ辿れる。別のP0 compiler条件へ一般化しない。

このT=.8の対応ではモデル価格Ĉを用いていない。有限PR cohort平均を全law期待価格と保証せず、新confidence/covarianceも計算していない。N×C、round G、全G、PF/PR総資源優位は計算・採択していない。長qのcoverageと次の最小案は[NEXT_EXECUTION_PLAN](NEXT_EXECUTION_PLAN.md)に分離した。

## SOURCE・テスト・実施範囲・STOP

| SOURCE/成果物 | 役割 |
|---|---|
| `311255aaac39dc1c3321108c296bde7dc93dd543` | D1科学実行SOURCE。結果公開commitと区別 |
| [h4_df10_rpe_d1.py](../../../src/trottertracks/h4_df10_rpe_d1.py)、[runner](../../../scripts/run_h4_df10_rpe_d1.py) | 係数prior・保存2pointの明示scalar評価だけ |
| [既存adapter](../../../src/trottertracks/h4_df10_rpe_connection.py) | e91021ae2の保存binding/disk/shot接続を無変更で再利用 |
| c1_stage3_accuracy / rpe_resource_accounting / rpe_target_round_horizon_validation | 既存raw geometry / Hoeffding / horizon、変更なし |
| [追加人工test](../../../tests/check_h4_df10_rpe_d1.py)・[記録](ARTIFICIAL_TEST_RESULTS.txt) | 7件PASS。旧準備15件は保存記録を再利用し重複実行せず |
| [PLAN](PLAN.md)、[result.json](result.json) | 固定scope、SOURCE/file/input hash、未資格項目とSTOP |
| e91021ae2 / [準備REPORT](../h4_df10_rpe_preparation_20261110/REPORT.md) | 固定SAVED_POINTS、元H/state/price binding |
| [shot調査][SHOT] / [価格調査][PRICE] | 旧DF12・Track D B/C・長qモデル/残差の再利用入口 |

元signal科学SOURCEは`e417ffc9f52b13c18f5493193ce510d858664259`、保存結果carrierは155529e2。今回の新scalar SOURCEに付け替えない。inputの完全hashはresult.input_bindingsへ保存した。保存NPZの係数だけを読み、state vector/full Hamiltonianの生成や再solveを行っていない。科学scalar評価はprior/horizon各1回、指定保存point各1回。新signal/trajectory/回路build・compile/価格refit/候補再最適化/RPE simulation/G/H6数値評価は0。依存libraryのimport初期化を研究用回路buildとは数えない。

人工追加接続testの初回はtoy priceのcohort欄欠測で失敗し、科学実行前にfixtureを補って7件通過した。実D1は固定SOURCEから一括完了し、科学的再試行や旧run再開はない。主担当Codex自身の確認であり、独立reviewやimmutable CIとは呼ばない。詳細科学レビューは未実施。

目的に合う既存branch `codex/resource-eval-rpe-shot-error-connection-20261110`を継続した。旧SOURCE/raw/結果/grant/ledger/marker/KEEP/STOPと他worktreeを変更せず、新しいapproval証明機構も作っていない。**本成果のGitHub公開とremote SHA/tree/本文照合を最終技術手順とし、その後mandatory STOP。D2/D3・H6・追加科学計算はユーザーの別途実行許可まで開始しない。**

[MAIN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md
[SHOT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fd76a3edaf5b6fee9cadb556ba57efc7c46ed778/docs/research/rpe_shot_error_connection_20261110/REPORT.md
[PRICE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/caa592a807c533c36e06a94d2fd4e62af4a38734/docs/research/rpe_price_reuse_survey_20261110/REPORT.md
[H6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/99ad77ef949b70192918a8700b092dd33c17af8c/docs/research/h6_df19_rpe_preparation_20261110/REPORT.md
