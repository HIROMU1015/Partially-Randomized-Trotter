# 保存PF/PR候補 → RPE入力の対応

[REPORT](REPORT.md)／[SAVED_POINTS.json](SAVED_POINTS.json)。元研究155529e2の既存`best_configurations.json`からH4・T=.8/2.0・εsignal=.0005・formula_group=ALLのPF/PR行を選んだ。旧候補の引用であり、RPE候補の採択、新しい最適化・順位・N/G計算ではない。

## 保存された4点

全点Linear H4/1Å/STO-3G、tol-only DF10、保存full-DF-H reference、同state/sector36、symmetric-directional、state-preparation除外。P0=Qiskit1.3.0／opt1／seed17／rz,sx,x,cx／測定付き全Hadamard。原scalar・identity・signed angle・actual adjoint規約を保つ。

| candidate ID | T / q_PF / δ | formula・prefix | R_outer / K | 保存log B | 保存探索単価RZ/軸 | 価格cohort |
|---|---|---|---|---:|---:|---|
| `49709717a6c3d405c60d` | .8 / 8 / .1 | B0 S2、p7 | tailなし | 0 | 44411 | 決定論n1。二つのcarrierは同価格の別参照で、二標本としない |
| `14bd81a375400c9b0305` | .8 / 8 / .1 | PR S2、p1 | 4 / 2 | .16343103470600182 | 16857.11111111111 | 探索n9／独立確認n33を別保存 |
| `e9c904a137c2f96a5bf0` | 2 / 1 / 2 | B0I m5_best、p7 | tailなし | 0 | 60950 | 取得carrierはEXPLORATIONだが決定論n1 |
| `fb134f11827f3e240843` | 2 / 1 / 2 | PR m5_best、p3 | 32 / 6 | .10426191233050415 | 35967.666666666664 | 探索n9／独立確認n33を別保存 |

表は既存欄の転記。新しい比・差・総費用・成功率を作っていない。独立確認価格を使った新Gも計算していない。選択後の価格と探索価格の役割を入れ替えず、cohortをpoolしない。RZ/CX/三depth/sizeは各priceの`mean_metrics`にあり、意味は一shotのtopology-free全wrapper価格。logical幅はこのprojectionでは未取得のままnull。Tの違いにかかわらず同じ価格となる点があっても、一般的な時間非依存性を推定しない。

## 欄対応と欠測

| 元SOURCEの欄 | 今回の保存／接続欄 | 意味・次回の処理 |
|---|---|---|
| candidate `{id,system,method,formula,prefix,T,q,delta,R_outer,K}` | `candidate`そのまま、`q_PF`はbind時の別名 | Tは評価時刻。`binding.T=.8`は元referenceの履歴で、T2の評価時刻を.8に戻さない。R_outerはouter全体のshort-step数で、signed公式のpiece配分は元SOURCEを使う |
| scientific_provenance.inputs.H4.binding | `binding` | H/snapshot/state/rank-policy/fragment生成系列とcompiler/controlの固定identity。DF12・CDF8・別rankの値を混ぜない |
| signal.reference | `saved_signal.reference` | 保存DF full-H・保存stateのsignal。ground Eのpure phase/unit radiusとの対応は未資格。T=.8のPF/PRの末位差も原形で保持 |
| signal.corrected／raw | `saved_signal.corrected/raw` | 正normalizationでraw=corrected/B。rawが欠ける場合に自動再構成しない。correctedを±1 outcomeの平均へ代入しない |
| total_signed_bias、outer_PF_or_discard_bias、finite_RTE_bias | 各保存欄そのまま | PF .8には詳細分解の保存がなくtotalのみ。既存totalを詳細項と二重加算しない。座標biasをoperator normやrigorous boundへ格上げしない |
| signal.log_B | `saved_signal.log_B` | 全trajectory/全roundに相当する保存point全体のnormalization。q_PFやR_outerを再乗算しない。B/A新数値欄はnull、次回許可後に必要なscalar変換のみ |
| endpoint.u/log_B_margin/old_u/old_margin | `saved_allowance` | 主政策u floor1e−9と旧marginの保全。今回減らさず転記。normalization uncertaintyをraw誤差へ渡す方法はD1の明示入力で、未知を0にしない |
| endpoint.axes、N_cos/N_sin、ε | `saved_finite_time_endpoint` | **finite-time signal用**の旧N。Re/cos/X、Im/sin/Yの対応。RPEのN_X/Yはnullで、旧Nを移すことはしない |
| reused/exploration/confirmation cost | `prices[*]` | candidate全tuple・binding・compiler・scopeと出典を付与。直接有限標本／直接決定論を区別。旧価格modelはこの4点に注入していない |
| 元表にないRPE target/schedule/α/β/phase/radius | null／未登録 | ユーザー採択と実行許可後に評価。h/m未採択の現段階で勝手にroundを与えない |

SAVED_POINTSはsignal/価格原本全体の再公開ではない。公開済みscalar欄の限定projectionと出典selectorであり、private event/QPY/NPZ・trialfileを含めない。探索／確認の保存mean/SD等を再計算していない。

## D1へ渡す誤差の単位

将来の`corrected_signal_disk`は、**同じtarget Eに対する**補正済みstate-specific信号の誤差diskとして供給する。保存referenceとexp(−iET)の差を含むかを明示し、保存total bias/uだけでreference資格を省かない。原研究のuは経験的なので、本adapterの出力も数値pilotの条件付き診断に限定する。

adapterのraw誤差接続は既存規約の

`A=exp(−log B)`、`D_raw=A·disk_corrected + d_normalization,raw + d_implementation,raw + d_law,raw`。

既存geometryは`rho=A−D_raw`、`beta_det=asin(D_raw/A)`、座標余裕`rho·sin(beta_stat)/sqrt(2)`から軸別αのHoeffding十分Nを求める。これはコードの処理説明で、今回実H4へ適用していない。raw余裕でNを求めた後、価格へさらにB²を掛けない。βdet＋βstat、alias、energy prior、全round条件は別に維持する。
