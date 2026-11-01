# P3 Stage 4：H4・5精度のnominal条件付きN/G会計

2026-11-02 JST。**保存結果だけによる条件付き会計が完了。10会計行・20軸すべてELIGIBLE_CONDITIONAL。**
[STOP](STOP.json)：`STOP_STAGE4_CONDITIONAL_ACCOUNTING_COMPLETE_NO_AUTOMATIC_FOLLOWUP`。
本実行の主結果は、DFの必要shot数が少なくても、保存標本価格との組合せでは全5精度でPauliのRZ総費用が小さいこと。
Pauli/DFのRZ費用比は約0.45273。ただし**finite-bit law/angle gap=0、compiled ρ=0、
保存B engineering上界候補がnominal振幅を覆う**という仮定の下の会計であり、実機精度保証・母平均順位・RPE全round総費用ではない。

## 条件・SOURCE・取得範囲

linear H4、隣接1Å、STO-3G、中性singlet、system8/full256、保存DF tol-only1e-8/rank10。
案A：共通Gaussian T_eff保持/V_square残差、M_B=Pauli184、M_C=DF rotated Z/ZZ360・10 basis。
T=exact binary64 .8、S2/q8/r4/K4、P=0、cosine/+Im sine full wrapper。
ε_sig=.05/.01/.005/.001/.0005、名目軸精度ε_sig/√2、軸failure budget .025。
実binary64のε_aとα_aは保守側に下向き丸め、両軸の複素精度・総failure .05を超えないことを確認した。

- 科学SOURCE `2f0d31f75ab1ecacc73d9ee5c34ddfc997396102`。人工7 PASSと入力技術照合後に公開・bytes照合し、1 batchだけ実行した。
- 承認済み[実行契約][CONTRACT]は`e4a9b1345a46f1bccccbf451f4e2d26521350fe3`。
- Stage 2：結果`8bcf5851f99f980dc08c5885ace54874b554c815`、科学SOURCE`4bfbbaad1f8178077489bf06b085377256d6fb80`。
- Stage 3 run04：結果`e44e87a306d05c8549f1a044b7fd056c2e1055a3`、科学SOURCE`1cfce5e961e3d063741a1e0656052b10c7b27392`。
- 元H4保存anchor`4591b8d75a2a6a55c74a51225f0c61648023c8ee`、生成SOURCE`0c329b30a81accb35c212be8defa66fee0dd0f12`。
- 保存JSON8件と、元科学入力binding15件の共通性・raw state/ν・時間・測定境界を確認。[SOURCE/入力来歴](source_and_inputs.json)。

新sampling・compile・Gaussian合成・reference/state/Hamiltonian生成・新signal・finite bias高精度化は0。
Stage 3 run04の各方式32 trajectoryだけを使い、旧pilotとのpoolingなし。旧科学結果・全STOPは変更していない。

## NとRZ総費用G

N_total=N_cos+N_sin。G=N_cos*Cbar_cos+N_sin*Cbar_sin。
±は価格標本だけのSEであり、N/B/biasの不確かさや実装誤差を確率化した値ではない。
Gのexact Fractionと未丸め表示値は[results.json](results.json)、[accounting.csv](accounting.csv)に保存。

| ε_sig | Pauli N_total | DF N_total | Pauli G_RZ ± 価格SE | DF G_RZ ± 価格SE | G_P/G_DF |
|---|---:|---:|---:|---:|---:|
| 0.05 | 21530427 | 16767433 | 6.4977483e+10 ± 2.02682e+08 | 1.4352451e+11 ± 2.51842e+09 | 0.452727431 |
| 0.01 | 544800599 | 424278978 | 1.6441742e+12 ± 5.12862e+09 | 3.6317087e+12 ± 6.37254e+10 | 0.452727430 |
| 0.005 | 2212935941 | 1723386932 | 6.6785024e+12 ± 2.08320e+10 | 1.4751707e+13 ± 2.58848e+11 | 0.452727414 |
| 0.001 | 63304873265 | 49300489113 | 1.9105015e+14 ± 5.95937e+11 | 4.2199832e+14 ± 7.40479e+12 | 0.452727277 |
| 0.0005 | 311123629521 | 242296579608 | 9.3895167e+14 ± 2.92884e+12 | 2.0739906e+15 ± 3.63922e+13 | 0.452727066 |

この固定条件ではPauliのBが大きく、各軸の必要shot数もDFより多い。
一方、run04のFull単回RZ価格の標本平均差を組み合わせると、Pauliの条件付きGが小さくなる。
normalizationと単回価格が逆方向に働くという会計を、同じsignal精度・failure・共通referenceで確認した。
これは元の純RTE対DF B3とは同一比較でなく、表現だけの独立因果効果・全候補最適性・一般的方式優位を示すものではない。
[comparison.csv](comparison.csv)には方式間G差・独立価格SE・費用比を保存した。方式間paired SEは使っていない。

## Bias：nominal-reference直接評価・数値不確かさの一回計上

保存nominal点zと保存physical reference区間[L,U]から、
`b_upper=max(|z-L|,|z-U|)+u_num+u_bind+u_ref_transfer+transfer`を用いた。
reference区間幅は端点距離に含み、display.axis_resolutionを追加していない。
outer＋finiteを合成する経路を使わず、共有splitの数値誤差を二重に数えていない。

| 方式 | 軸 | signed nominal-reference midpoint差 | 保守側bias候補 b_upper |
|---|---|---:|---:|
| P | cosine | 9.071241298731e-05 | 9.071316336196e-05 |
| P | sine | 1.481250508300e-05 | 1.481325545765e-05 |
| DF | cosine | 9.071241298944e-05 | 9.071328355079e-05 |
| DF | sine | 1.481250507989e-05 | 1.481337564124e-05 |

exact端点・各数値不確かさ・丸めoperandはresults.jsonのbias_inputsに保存。
Stage 2 finite bias成分の符号・大きさを新たに解像したわけではなく、UNRESOLVEDをゼロに置き換えていない。
u_remaining=0は残るfinite-bit gap/compiled ρの仮定で、solver/binding/reference不確かさは上記bias候補に含めている。
元のu_num等がengineering目安であることは変わらず、端点の正確な演算だけで厳密certificateにはならない。

## B上界候補・shot丸め・headroom

B_engineering_upper=max(B_saved+B_uncertainty, decimal50.B)と定義した。
保存β/B/lawを再生成せず、保存値に対するscalar上界候補・log/丸めだけを計算した。
Bを補正済みnominal信号へ再乗算していない。

| 方式 | B_saved | 保存B_uncertainty | B_engineering_upper（表示） | log_B_upper operand |
|---|---:|---:|---:|---:|
| P | 39.12590996575007 | 4.4918179498e-12 | 39.125909965754559 | 3.6667849065005163 |
| DF | 34.52799918748691 | 4.0881316050e-12 | 34.527999187490991 | 3.5417705655438367 |

この上界候補が名目振幅を覆うことも条件付き仮定。decimal50検算やB_uncertaintyを厳密なlaw上界保証とは呼ばない。
上向きlogに対するexpのscalar端点とmath.exp返却値を含め、より保守側のB_scalar_upperでshot不等式を照合した。
既存fixed_signal_shotsをEMPIRICALとして20軸で再利用し、
`N>=2*B_scalar_upper²*ln(2/alpha)/h²`をscalar端点・有理数演算で確認した。
全20軸で既存返却ceilの増加修正は0。既存2^52安全整数domain、h>0条件を維持し、1%guardは追加していない。

| ε_sig | 方式 | N_cos | N_sin | 保守側h_cos | 保守側h_sin |
|---|---|---:|---:|---:|---:|
| 0.05 | P | 10788358 | 10742069 | 3.526462589597e-02 | 3.534052580387e-02 |
| 0.05 | DF | 8401741 | 8365692 | 3.526462577578e-02 | 3.534052568369e-02 |
| 0.01 | P | 275346104 | 269454495 | 6.980354648504e-03 | 7.056254556408e-03 |
| 0.01 | DF | 214433618 | 209845360 | 6.980354528315e-03 | 7.056254436224e-03 |
| 0.005 | P | 1130578328 | 1082357613 | 3.444820742571e-03 | 3.520720650475e-03 |
| 0.005 | DF | 880470095 | 842916837 | 3.444820622382e-03 | 3.520720530292e-03 |
| 0.001 | P | 35311620169 | 27993253096 | 6.163936178246e-04 | 6.922935257289e-04 |
| 0.001 | DF | 27499939402 | 21800549711 | 6.163934976358e-04 | 6.922934055453e-04 |
| 0.0005 | P | 194200454999 | 116923174522 | 2.628402272313e-04 | 3.387401351356e-04 |
| 0.0005 | DF | 151239266851 | 91057312757 | 2.628401070425e-04 | 3.387400149520e-04 |

[axes.csv](axes.csv)に全20軸のstatus、headroom exact、元関数Nと最終N、ceil修正量を保存した。
今回null行はなかった。一般の欠測・h非正・domain外はnull/UNRESOLVEDを保持する人工検査を行い、0補完していない。

## 価格標本SE・補助指標

run04のcos/sin各32価格のexact平均・不偏標本分散・同trajectory共分散を使い、
`Var(Gbar)=(Nc²*s_c²+Ns²*s_s²+2*Nc*Ns*cov_cs)/32`を評価した。
同じ32本のg_i=Nc*C_i,c+Ns*C_i,sからの標本分散とexact一致することも確認した。
PとDFは独立cohortなので、G差の価格SEは両方式の分散の和から算出した。
旧pilotのpoolingやcos/sinを64独立標本として扱う処理は行っていない。

| ε_sig | 方式 | G_CX ± 価格SE | shot加重depth ± 価格SE |
|---|---|---:|---:|
| 0.05 | P | 2.4870334e+10 ± 1.03711e+08 | 3.3431371e+10 ± 1.12355e+08 |
| 0.05 | DF | 4.6477228e+10 ± 7.51179e+08 | 6.6100365e+10 ± 1.13507e+09 |
| 0.01 | P | 6.2931279e+11 ± 2.62428e+09 | 8.4593913e+11 ± 2.84300e+09 |
| 0.01 | DF | 1.1760483e+12 ± 1.90076e+10 | 1.6725873e+12 ± 2.87214e+10 |
| 0.005 | P | 2.5562176e+12 ± 1.06596e+10 | 3.4361363e+12 ± 1.15480e+10 |
| 0.005 | DF | 4.7770132e+12 ± 7.72075e+10 | 6.7939144e+12 ± 1.16664e+11 |
| 0.001 | P | 7.3125042e+13 ± 3.04937e+11 | 9.8296642e+13 ± 3.30351e+11 |
| 0.001 | DF | 1.3665479e+14 ± 2.20866e+12 | 1.9435177e+14 ± 3.33738e+12 |
| 0.0005 | P | 3.5938668e+14 ± 1.49867e+12 | 4.8309722e+14 ± 1.62357e+12 |
| 0.0005 | DF | 6.7161583e+14 ± 1.08549e+13 | 9.5517855e+14 ± 1.64022e+13 |

depth指標はshotごとのdepthを加重加算した値で、装置上の並列wall timeではない。
SEはNを固定した保存32標本からの診断で、非漸近confidence interval・母平均上界ではない。
run04でorder4が両方式未観測という制限を維持し、未観測tail寄与を0と認定していない。
`population_tail_bound=null`、実compiled/finite-bit誤差は未測定のまま。

## 実装検査・保全・STOP

人工7 tests PASS：区間biasと数値誤差の一回計上、B候補/丸め、複素精度の保守側配分、
負の軸間共分散・直接重み付き分散、ceil不等式・整数domain/null、独立方式差・欠測伝播。
既存fixed_signal_shotsとtrans_boundsを再利用し、旧1%guard/rho探索・科学runnerは呼んでいない。
純scalar moduleだけを読み、QPYや分子データの再構築・simulator/compiler importはない。
専用環境のPython3.11.0rc1で逐次実行。共有環境・他worktree・他利用者への変更なし。
科学batch1、10会計行/20軸、exit0、wall `0.189108`秒。

```json
{
  "scientific_scalar_batches": 1,
  "accounting_rows": 10,
  "axis_rows": 20,
  "new_sampling": 0,
  "new_compile": 0,
  "new_Gaussian": 0,
  "new_reference": 0,
  "new_signal": 0,
  "finite_bias_high_precision": 0,
  "automatic_retries": 0
}
```

[verification.json](verification.json)にscalar検査・入力8件/SOURCE6件保全を記録し、
[publication_verification.json](publication_verification.json)に転記・旧全STOP不変・公開対象hashを記録する。
旧Stage 1/2・Stage 3 run01〜04、過去のSOURCE/raw/marker/STOPを改変していない。

報告する3層は、(1)指定仮定下のnominal条件付きN/G、(2)価格標本SE、(3)未取得の実装・母平均tail保証。
条件付き比較の実装精度保証・RPE全round総費用への読み替え、重要claimの正式採択は行っていない。
次の担当はユーザー：この結果を条件付きcase studyとして使うか、finite-bit law・compiled誤差・価格tailのどれを追加検証するか判断する。
Codexは新たな明示指示まで追加科学解析・sampling/compile・高精度化・新条件・論文改訂を開始しない。
**今回のStage 4限定会計は完了。公開・remote一致確認後mandatory STOP。**

[CONTRACT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4a9b1345a46f1bccccbf451f4e2d26521350fe3/docs/research/h4_p3_stage4_conditional_accounting_contract_20261102.md
