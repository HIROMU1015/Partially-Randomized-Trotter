# Stage B：RPE信号・統計の再利用条件

2026-11-06 JST。[共通報告・次の検証順序](REPORT.md)／[Stage C](STAGE_C.md)。**既存13-roundのlaw・operator誤差・accounting・decoderを再利用する入口はある。保存own-DF groundの信号・radius・Nを、元H厳密groundの結果として採用する接続は未完成である。** 以下はSOURCEと保存欄の静的確認であり、新しい誤差・信号・shot数・成功確率は算定していない。

## B1. 対象と既存証拠

対象候補はStage Aと同じlinear H4／隣接1.00 Å／STO-3G／Nα=Nβ=2、8 system modes、number-CDF8、元frame順LD2、ordered S2/s1/r32/K0、h=δ=.1。既存13-roundはm0..12、q=2^m=1..4096である。これは**再利用候補の保存schedule**であり、今回新しい目標精度・成功確率・horizonを採択したものではない。元Hの厳密groundを仮定し、状態準備費は除外する。

| 資料 | 確認内容／再利用範囲 |
|---|---|
| [r32下流報告][C32]、[固定契約][CONTRACT] | 科学SOURCE `4df8eca29c59ffd55e2dd898f05da16bb246ddbe`、実装 `703938aba8cc38080e29e75acde8e0edc8dce868`。同一PR endpointの13-round資料。ENTRY比較の成績は方式間比較へ転用しない |
| [PR/A/law][LAW]、[normalization][NORM] | λ_R=0.5000442435909978、identity=0.041217410629575235、216 atom、probability_sum=1。ideal finite-RTE IIDとimplemented finite-bit law誤差nullを区別 |
| [PR/B/references][REFS]、[m12][M12] | PF operator誤差、normalization-corrected RTE operator誤差、attenuationとown-DF groundのraw信号が別欄。state-independentなoperator誤差も、比較先HamiltonianはCDFである |
| [c1_stage3_reference.py][REFERENCE] `reference_kernels`／`reference_round` | ordered PF／corrected kernels、matrix powers、同じoperator norm、raw信号=A×corrected expectation。保存値閲覧のためにこの関数を起動していない |
| [c1_stage3_accuracy.py][ACCURACY] `RoundEvidence`／`assess_round`／`assess_energy` | raw信号単位への変換、正半径、phase、Hoeffding N、全round・failure・energy判定。ただしstate_modelはown-DF ground専用 |
| [c_pr_aware_df_stage_c.py][ACCOUNTING] `qualification`／`assess_round`／`energy` | evidence・ideal・assumedを分離。unknown synthesis/lawを0にしない。state_modelをown-DFへ固定するcall siteがある |
| [rpe_resource_accounting.py][RPE] | `_hoeffding_shots`、N×期待価格、全roundとscopeの整合確認。`evaluate_rpe_round_candidate`のunit-radius specializationを元H入力の証拠にはしない |
| [既存decoder][DECODER]、[測定接続資料][MEAS]・[adapter][MADAPTER] | `reconstruct_rpe_phase`は連続dyadic qに対する復元関数。measurement adapterはlegacy DF12／LD3,12／δ=.02／18-roundを固定しており、CDF8へそのまま起動できない |

ここで挙げたreference・accuracy・connection・stage_c・RPE accounting・decoderの6ファイルは、今回の基点`3dc9a61e…`とC科学SOURCE`4df8eca2…`でGit blobが一致した。これは6部品の同一性確認であり、closure全体の再認証ではない。`c_near_practical_r32.py`等は当checkoutの通常srcには無く、固定Git objectから読んだ。[固定SOURCEの接続部][R32SOURCE]は既存ground solve・sampling・compileまで呼ぶため、保存資料を読む目的でrunner全体を実行しない。

## B2. 保存された数値の資格

次の値は[C32報告][C32]と[references][REFS]の**引用だけ**である。新しい集計・移送計算はしていない。

| PRの保存欄 | 既存値・status | 元H入力への扱い |
|---|---|---|
| ideal lane総shots（X+Y） | 14738。m0は512/axis、m12は910/axis | own-DF ground＋exact native＋ideal IIDの条件付き値。今の元H用Nとして採用しない |
| 旧目標 | energy=.00159 Ha、failure=.01、αaxis=.01/26、βtotal=π/6、βstat=π/12 | 当該runの履歴。今回の新しい共通目標・配分には固定しない |
| m12 | PF error=.0868531196299406、corrected RTE error=.03377065128859485、A=.852119825515583 | 同CDF・同scheduleのoperator誤差／attenuation。元Hとの差の会計が別途必要 |
| m12 raw signal | Re=.1805377577488709、Im=.8394098697286076 | own-DF stateのexpectation。元H stateのphase／radiusとは認定しない |
| m12 ideal radius lower／βdet | .749333918887／.120918217460（報告表の桁） | own-DF reference signalを中心とする条件付き数値pilotの欄 |
| evidence lane | UNQUALIFIED_ROUNDS、shots／費用null | all-law synthesis・implemented-law誤差の欠落を維持 |
| ideal／assumed lane | CONDITIONAL_CA_PASS／未検証仮定を含むscenario | actual CA・厳密包囲・実PRNG成功保証へ昇格しない |

### 元Hのground保存物はあるが、同一視はできない

carrier `3649b75af191916daf44abb91e14d34c43b83e8f`には、`artifacts/research_c_near_practical_r32/2026-11-03/run01/`以下の保存物をGit metadataで確認できた。

| 保存path | Git blob | 今回の確認 |
|---|---|---|
| [shared/ground_physical.npz][PHYSVECTOR] | `eca169c47cebfc2f8099eeb052376ea0d16c877a` | 存在・identityのみ。配列を展開していない |
| [shared/physical_ground.json][PHYSJSON] | `0ae911c88175487685ace90e1ca8991049ad3782` | energy=−2.1663874486347625、residual=1.7642254846760187e−15を保存 |
| [PR/A/ground_candidate.npz][DFVECTOR] | `57d45f82e084ed5d61321778345131ac43076d05` | own-DF vector、上記physical vectorと別物 |
| [PR/B/sector_reference.npz][KERNELS] | `27de8fbb29076e58db866facba28d2dff1b3a7fb` | 保存PF／corrected kernel。存在確認のみ |

注意すべき実装上の点として、`shared/physical_ground.json`にも`state="own_DF_ground_state_oracle"`が保存されている。[r32 SOURCE][R32SOURCE]ではphysical Hを渡した共通ground計算とcandidate Hの計算を分離し、既存`ground_gate`の共通ラベルを出している。**文字列ラベルだけからphysical／candidateを判定しない。** 次の利用時は元H packet、physical_referenceのH、sector basis／bit ordering、保存vectorのbindingを確認する必要がある。

保存DF sector norm＋engineering guardが小さいことや両energyが近いことだけから、ground vector／全roundの信号／整数Nの一致は導けない。保存physical vectorもbinary64の数値groundであり、厳密ground oracleの数学的保証そのものではない。今回はoverlap・gap・signalを計算していない。

## B3. 元H入力へ再利用できる部分と不足

既存`assess_round`は`state_model != 'own_DF_ground_state_oracle'`を拒否する。`assess_energy`もown-DFを要求し、DF energy差を最終energy幅へ一度加える。元H向けにstate名だけをown-DFと偽って通す方法は使わない。

再利用候補は、同sectorでのPF/RTE operator誤差、lawとA、raw誤差→radius／phase→Nという既存手順である。元H厳密groundはexact `exp(-itH)`の位相を与えるが、CDF／PF／finite-RTEの信号半径を自動的に1にはしない。[研究目的の半径条件][PURPOSE]と[解析手順][METHOD]を維持する。

最小の追加評価候補は次の二経路。**選択・実装・数値評価は未実施**である。

1. **state vectorを要しないoperator誤差の接続を先に検討する。** 同一sectorを保存する元H／CDF／PF／corrected operatorに対して、元Hのexact evolutionまでの差を同じnorm・raw信号単位で扱う。保存DF norm＋guardを使う移送条件の確認、または`reference_round`のexact側を元Hへ結び直す限定評価が候補。保存CDF-relative PF誤差を、そのまま元H-relativeと呼ばない。元H-targetの信号誤差へDF差を含める方式と、own-DF energyへDF差を加える旧方式を混ぜてDFを二重計上しない。
2. 上記の十分条件だけで判断できない場合に限り、保存`ground_physical.npz`と既存kernelによる元H向けphase／radiusの数値pilotを候補とする。新ground solveやDF再fitを初期案の必須処理にしない。保存vectorとexact-ground仮定の差を扱う資格・数値精度は別途明示する。

どちらでも、元H向けのalias、共通energy精度・成功確率、全roundの正半径・phase余裕、synthesis／law／Reference補正誤差が必要。保存数値pilotは厳密上界へ格上げしない。追加予算が未確認ならN／成功保証は未確認のまま残す。

## B4. Referenceと同じNを使える条件

[既存信号対応][SIGNAL]のfresh-input一shot条件を全roundの**full ordered path**へ適用する。Vω,m=Aω,m Bω,m、Vω,m r=e^{iχω,m}r、actual A†とP(+χω,m)を保持する場合、補正coherenceは直接制御のzω,m=〈E₀|Vω,m|E₀〉に対応する。同じH→core→YのみSdg→H、bit0→+1／1→−1なら、各軸bで

\[
\Pr(Y_b=y\mid\omega,m)=\tfrac12[1+y f_b(z_{\omega,m})],\quad
f_X=\operatorname{Re},\ f_Y=\operatorname{Im}.
\]

これは既存恒等式の適用条件の転記であり、今回の全law／全round検証結果ではない。X/Yの同時測定joint lawも主張しない。

| 保存対象 | 必要な条件 |
|---|---|
| path条件付きX/Y分布 | 同じordered V、元H入力、reference ray、χ、scalar/identity、actual adjoint、同じ二値読出し。Referenceのsystem出力状態がdirectと同じであることは含まない |
| raw平均のphase／radius | 全対象pathで上記対応、同じp_m、同じnormalization／attenuation。B_KやLCU係数をsample済みoutcomeへ重ねて乗算しない |
| `{N_m,b}`と全round成功条件 | 元H向けdirectのN自体が資格を持つこと。fresh state／fresh IID trajectory per shot、各occurrenceのlaw、軸・round schedule、decoder、aliasと全誤差配分を維持すること |
| weighted／return方式 | さらにweight range、normalizer、zero-fill／採否／post-selectionとtrial数を含むjoint lawを維持。通常±1用Nを横流ししない |
| compiled実装でのN保存 | 全対象domainのReference位相補正、inverse、leakage、合成／routing等の誤差が同じ共通目標の余裕に収まること。Stage A有限32本の最大残差を全law boundにしない |

「同じN」は**方式間で同じround別・axis別の列`{N_m,b}`**を利用する意味であり、全roundへ同一の整数Nを固定する意味ではない。Referenceの条件付き同値性は、まだ元H向け資格のない保存Nを資格化しない。厳密に同一のmeasurement lawなら同じN・decoderによる成功分布も保存される道筋があるが、有限精度の場合は追加誤差の接続が先である。

## B5. shot・decoder・成功の再利用境界

既存Hoeffding式は、各軸内で独立な±1標本と正のcoordinate allowance εcoord,m,bについて

\[
N_{m,b}=\left\lceil 2\epsilon_{\mathrm{coord},m,b}^{-2}
\log(2/\alpha_{m,b})\right\rceil
\]

を用いる。radius／統計phaseからεcoordへ変換する既存手順、系統phase＋統計phaseのbranch余裕、全round・両軸のα和を保持する。round間独立はunion bound自体には不要だが、binomial productや仮想測定に用いる軸間・shot間の独立性は別条件である。価格標本n=32は量子shot数Nの倍率ではない。

同じ保存wordをN shotへ固定すると、無条件では共通latent pathを持つ混合分布になり、一般に保存raw平均の単一二項分布と異なる。32本を巡回する方法や同じ短pathをq回反復する方法は、元のfresh-occurrence lawの代用品ではない。PCG64 seed非重複だけでideal IIDやimplemented joint-law誤差を保証しない。

decoderはarg(raw X+iY)からdyadic branchを復元し、φ=−Eδの符号・alias branch・zero signal／tieの扱いを維持する。E₀をdecoderの入力にして正解branchを選ばない。旧legacy18-roundのV2成功値やm17包含結果、旧DF12のNはCDF8へ移送しない。CDF8の保存13-round accounting判定を、元H入力でのfresh-IID decoder成功実験と同一視しない。

**Stage Bの現在地：** 再利用SOURCEと条件の整理まで。元H用phase／radius／N、全law実装誤差、Referenceの全round測定対応の数値資格は未取得。次の最小評価とSTOPは[共通報告](REPORT.md)にまとめた。

[C32]: ../research_c_near_practical_r32_downstream_execution_20261103.md
[CONTRACT]: ../research_c_near_practical_r32_downstream_contract_20261103.json
[LAW]: ../../../artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/A/law.json
[NORM]: ../../../artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/A/normalization.json
[REFS]: ../../../artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/B/references.json
[M12]: ../../../artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/B/m12.json
[REFERENCE]: ../../../src/trottertracks/c1_stage3_reference.py
[ACCURACY]: ../../../src/trottertracks/c1_stage3_accuracy.py
[ACCOUNTING]: ../../../src/trottertracks/c_pr_aware_df_stage_c.py
[RPE]: ../../../src/trotterlib/rpe_resource_accounting.py
[DECODER]: ../../../src/trotterlib/rpe_four_round_phase_validation.py
[MEAS]: ../rpe_18round_measurement_connection_20261103/IMPLEMENTATION_CONNECTION_REPORT.md
[MADAPTER]: ../rpe_18round_measurement_connection_20261103/measurement_adapter.py
[R32SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4df8eca29c59ffd55e2dd898f05da16bb246ddbe/src/trottertracks/c_near_practical_r32.py
[PHYSVECTOR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3649b75af191916daf44abb91e14d34c43b83e8f/artifacts/research_c_near_practical_r32/2026-11-03/run01/shared/ground_physical.npz
[PHYSJSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3649b75af191916daf44abb91e14d34c43b83e8f/artifacts/research_c_near_practical_r32/2026-11-03/run01/shared/physical_ground.json
[DFVECTOR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3649b75af191916daf44abb91e14d34c43b83e8f/artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/A/ground_candidate.npz
[KERNELS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3649b75af191916daf44abb91e14d34c43b83e8f/artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/B/sector_reference.npz
[PURPOSE]: ../研究目的・研究課題.md
[METHOD]: ../研究方法・解析手順.md
[SIGNAL]: ../track_d_reference_control_feasibility_20261106/SIGNAL_PRESERVATION_MAP.md
