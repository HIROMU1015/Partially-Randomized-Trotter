# Track C：D内部PFの改善（2026-11-08）

**固定H4では、D内部の主要ground biasをD/R混合biasと相殺する構成が有効だった。** Dだけの対称3段合成をBCHから調整すると、登録済みPRと同じr_mで、全18roundのshot総数は8598→8498、総RZは約26.2%、総CXは約30.7%減る推定となった。RTEには負時間を加えていない。一方、同じ探索機会を与えた強決定論PFはさらに安く、PR全体の優位性は成立していない。

結果は保存Hermitian-H行列・そのexact ground・理想nativeゲート・fresh独立samplingの条件付き評価である。元の連続的な積分Hamiltonianとの包含、T合成、状態準備、ノイズ、一般の分子/状態への移送は未検証。既知の合成族のstate-specificな係数選択であり、新しい一般高次PFの発明とは扱わない。研究範囲を完了しmandatory STOP。

## 固定条件・再利用した研究

基点`48209b088ca0dffd22a8d3fddcb82888eb29d3a5`。Linear H4 / 1.00 Å / STO-3G / DF12生成順、Partial LD3（D4 block・R218 component）、元Hの保存ground、8 orbital・4電子sector70。強対照はFull LD12。18round、delta0=.02、epsilon_E=1.5936e-4 Ha、alpha総和=.05、位相予算=.4 rad、状態準備費除外を維持した。

保存common.pkl SHA256は`68269aa0415f048a0220beb4054daa1bba715c6eb29fbbe9d5ba779baf0346f2`。新SCF・DF fit・physical-ground solveは0。Floquet演算子の固有分解は今回のbias/折返し診断として実施した。B2 P1の入力・compiler設定は使用していない。

Qiskit1.3.0・compiler A、rz/sx/x/cx、数値最適化・angle cutoffなし、seed17（確率的passなし）。制御はlowering前に付ける。Python3.12.3 / NumPy1.26.4 / SciPy1.14.1。共有設定・他利用者プロセスの変更なし。最大64 CPU affinityをtaskに応じ利用し、outer process並列とBLAS1 threadを組み合わせた。

| 既存結果・SOURCE | 今回の利用と未検証の自由度 |
|---|---|
| C-N / C-S / C-P、`a78054dc`系列 | 全体S2/Y4/Y8・残差randomization・processingを確認。残差は大きいshot負担、processingは改善ほぼなし。LD3の全24内部順序は未検証だった |
| C2-A `ce84fa8e…` | round別のfinite-RTE信号・shot・強Full対照を再利用。以前はS2/Y4、h=.02/.01、生成順0123 |
| C2-B `602ca4b6…` | 合成精度配分の約.514%は最適性未確定。今回T価格へ流用しない |
| compiler A18round `3258a89f…`、保証A `720416c5…` | local native区間評価、SOURCE generator、spectral/ground/alias certificateを再利用。新しいD角度と対角位相算術を追加評価した |
| 直前の混合公式、基点commit | BCH多項式log・誤差分解・D/R kernel・価格処理を再利用。2MNの作用素改善がground改善に残らないこと、a=1/4が半刻み同値であることを対照にした |

新実装は既存`track_c_h4`のspectra/P3平均、`track_c_mixed_formula`の三次log、DFPartialS2の_append_block、RTE event builder・conditional_flags、compiler A、RPE decoder、directed interval関数への小さい実験用接続である。旧pipelineのprepare/run/STOP再開は呼ばず、出力を新namespaceへ分離した。

## D内部の設計とBCH

まず、既存片側sweepを用いるpartial S2の全24順序を調べた。添字0はone-body、1..3は最初の3 DF fragmentを表し、partitionの構成要素は変更していない。0123のground H2係数`.01334023255`に対し、最小は0321の`.01230116103`（約7.79%減）。全順序の係数は正だったので、その正時間合成の三次係数の和だけではゼロにできない。

fragment1個の時間を対称に再配分する96組も調べた。各fragmentの総時間をhに保ったまま、外側aと中央(.5−a)へ配分する。全PR ground係数はaの二次式となり、別のa点で検算した。この族では実根がなく、最小例はfragment0、a=.25181840612658435、係数`.00611798241`。他fragmentの最小点も保存した。

次に、D自身の対称S2を`S_D(t)`、そのeffective Hamiltonianを`H_D+t²K_D+O(t⁴)`とする。既知の対称合成族

`D_v(t)=S_D(vt) S_D((1−2v)t) S_D(vt)`

をPRの両側に置く：`D_v(h/2) R(h) D_v(h/2)`。**Rの時間は正のh一回のまま**である。`η(v)=2v³+(1−2v)³`とすると、

`H2_PR = H2_outer + η(v) K_D/4`。

ここで`H2_outer=−α C_D−β C_R`、`α=−1/24, β=−1/12`（外側D/2–R–D/2）。ground期待値から

`η(v)=−4⟨H2_outer⟩/⟨K_D⟩`

を解く。Yoshida4はη=0としてD内部だけを消すが、今回の構成は**負のD内部係数で正の混合biasも相殺する**。係数はBCHだけで選び、長roundの位相や価格にはfitしていない。

| D構成 | D内部ground H2 | mixed ground H2 | 全体ground H2 | D指数/outer step |
|---|---:|---:|---:|---:|
| 元partial S2 0123 | .01194830121 | .00139193134 | .01334023255 | 8 |
| 順序0321 | .01052108376 | .00178007726 | .01230116103 | 8 |
| D自己対称S2 sandwich | .00298707530 | .00303236699 | .00601944229 | 14 |
| D-only Yoshida4 | 約0 | .00303236699 | .00303236699 | 38 |
| D調整0123 | −.00303236699 | +.00303236699 | −2.04e-14 | 38 |
| D調整0321 | −.00303236699 | +.00303236699 | −1.70e-15 | 38 |

D調整0123はv=1.4768778287800015、η=−1.0151625527143378。0321はv=1.490624664134615、η=−1.1528724828732642。中央D時間は負だが、全fragmentのnet時間と物理Hamiltonianは保持する。負時間・追加D指数は回路価格に計上した。

Yoshidaの対称合成族は既知である。[原論文](https://doi.org/10.1016/0375-9601(90)90092-3)。BCH対角期待値/固有位相を重視する方針にも先行研究がある。[Maxwellほか](https://arxiv.org/abs/2606.30738)、[Childsほか](https://arxiv.org/abs/1912.08854)。今回のH4固有係数選択の一般的新規性は未確定である。

groundの主要energy補正は消えても、H2演算子自体はゼロではない。調整0123/0321のH2ノルムは約.03146/.03177であり、**作用素4次公式ではない**。孤立ground branchではleading energy補正がh⁴となる機構である。人工行列で上式・時間保存・modular分割の3検査がPASS。H4のh=.02→.01では位相係数が約1/4になり、機構と整合した。

## 有限RTE・全18round・成功条件

最初の信号gridはh=.005,.01,.02,.04,.08,.16,.32,.64,1.28、resolution=1,2,4,8,16,32。元S2・半刻みS2・D2/D4・再配分4種・調整2種・partial Y4・Full S2/Y4に同じ機会を与えた。11988行（実条件10434）を保存。Fullにも最初の4 blockの同じ24順序を与え、追加7452行を保存した。13!全順序の最適化は行っていない。

各R occurrenceのevent数は`max(1,ceil(|t_R| resolution/.02))`。既存K2/P3の平均・複素信号・normalizer・打切り残差を用い、canonical lawのfinite-bit誤差も会計した。追加RTEや負のR時間を無料に扱っていない。

登録済み対照のh=.02とr_mは保持し、調整構成はh=.16（短roundではT/ceil(T/.16)）・**同じr_m**で比較した。

| m17 | h | PF位相上界rad | rho下限の概数 | shots/軸 |
|---|---:|---:|---:|---:|
| 登録済み元S2 | .02 | .01400660 | .57270 | 626 |
| 順序0321 | .02 | .01291704 | .57270 | 623 |
| D調整0123 | .16 | .00232990 | .57269 | 592 |
| D調整0321 | .16 | .00212863 | .57269 | 591 |

調整構成のm17は16384 step、D指数606209（境界結合後）、RTE4194304 event。元S2は131072 step、D指数917505、RTE event数は同じ。logB=.557390826も同じで、改善はnormalizationを無視したものではない。

m17のTを固定した±10%のmesh holdoutで、調整0123のfinite位相差は`.0015203/.0023168/.0033907` radとなりh⁴則に整合した。0321も同様。係数をholdoutに合わせ直していない。

旧spectral/ground certificateを読み取り再利用し、新D時刻のprimitive算術、実際のmodular D compiler context、新しい角度のlocal interval、RTE全supportの既存角度/basis、event phase、fresh-bit proposal誤差を会計した。初期110条件・追加36条件・登録対照用73新条件、計219 unique条件で予算が閉じた。native誤差を0扱いしていない。

各軸alpha=.05/36、`N≥4 log(1440)/(rho² sin² beta_stat)`。3×.4<πと`.4/(.02×2^17)=.000152587890625 Ha < epsilon_E`から、**固定保存行列・exact ground入力・理想nativeゲート・fresh独立bitsの条件下で成功率≥.95**となる。元の連続積分Hへの無条件保証ではない。

E0をdecoderへ渡さず、既存nearest-branch復元を適用した。登録比較の全18roundでbranchとenergy精度が通過した。調整0123の平均信号energy誤差は8.84e-7 Ha（元S2は5.34e-6）。量子shot実行・実機成功率測定は行っていない。

## Native価格と全RPE資源

使用するRTE occurrenceの**33長さ/分布条件を直接compile**し、通常/少なくとも1個order2の層に各16標本、計1056標本。層確率を掛け、共有cache・共通乱数によるpaired covarianceを保持した。D価格は410条件。compiler Aを適用したD区間・RTE occurrenceを連結する具体的なmodular回路を価格の対象とした。

この実装のRZ/CX/sizeは部品の和として厳密に加算できるため、m17へのwhole-compile fit移送は不要である。**depthの和はserial実行上界であり、全DAGの実depthではない**。18短条件のX/Y whole compileと比較したところ、RZ/CX/size差は全て0だったが、これを全長の最適whole価格の定理にはしない。

12件のsource/native状態検査でprojective差最大7.45e-13、測定差1.35e-12。別途、D spectraとsigned DF involution/event_unitaryから独立に予測した20 X/Y値は、full controlled sourceと最大1.86e-14で一致した。scalar・eventの相対位相を含む検査である。

下表は`G_j=Σ_m,b N_m,b E[C_j,m,b]`を個別に計算したもの。括弧内はMC標準誤差（1 SE）。gate指標を相互に加算していない。

| 同じ登録r_mでの構成 | 全18round shots | 総RZ | 総CX |
|---|---:|---:|---:|
| 元S2 h=.02 | 8598 | 4.7792e12 (1.2893e11) | 1.09324e12 (1.4076e10) |
| 順序0321 h=.02 | 8590 | 4.7607e12 (1.2842e11) | 1.08904e12 (1.4020e10) |
| 再配分fragment0 h=.04 | 8712 | 3.7940e12 (9.1704e10) | 7.6166e11 (9.9973e9) |
| D調整0123 h=.16 | 8498 | 3.5273e12 (6.5269e10) | 7.5776e11 (7.0997e9) |
| D調整0321 h=.16 | 8496 | 3.5295e12 (6.5191e10) | 7.5758e11 (7.0911e9) |

調整0123は元登録PR比でshots **1.163%減**、RZ **26.195%減**、CX **30.687%減**。paired差のMC SEはRZ1.2298e11、CX1.3406e10で、独立SEを単純加算していない。低標本の正規近似診断を厳密な価格信頼保証へ格上げしない。

元PRの各roundのshot数を上限にした既検証点の選択でも、8498 shots、RZ3.5268e12、CX7.5749e11となった。これは有限menuの価格推定値に基づく選択であり、global最適性・MC選択biasの除去は未証明。主たる改善証拠は、長round信号や価格にfitせずBCHで指定した固定D調整構成の比較である。

別に、両armへ同じh/r menuとsurrogate選択を与えた比較も保存した。元S2は11150 shots、RZ4.4440e12、CX1.0193e12。shotを各round増やさない新しい選択では10832 shots、RZ3.3632e12、CX7.2628e11。資源だけの最小推定値を選ぶと総shotが増えるpolicyもあり、全てのpolicyがshotを減らすとは主張しない。

## 強PFと折返し・負結果

Fullの24順序探索では、coarse hのendpoint位相が2πの整数倍付近で小さくなるpolicyも得た。例えばS2順序1320・h=1.28のm17は、dominant energy bias約.023996 Ha、累積bias約62.90 radなのにwrapped PF位相上界は約.06984 radだった。roundごとのsource/native成功条件は閉じるが、ground energy-bias低減の機構とは区別する。rawを不適格値として消去せず保存した。折返し・dominant overlapの分類自体はfloat64診断である。

全armに追加の十分条件`|δE_dominant|T + asin((1−p)/p)<.4`を同じく適用し、折返しに依存しない対照も計算した。PRの調整候補はこの条件を通過し、±10% mesh holdoutでも改善が残る。

| 折返しに頼らないnative-qualified policy | 総RZ | 総CX |
|---|---:|---:|
| 改良PR・資源優先round選択 | 3.3485e12 (4.9710e10 SE) | 7.2166e11 (5.4075e9 SE) |
| 強Full PF・同じ順序機会 | 1.2094e12 | 3.8093e11 |

Fullは決定論価格なのでprice MC SEは0。最適性は未証明だが、この実行可能な強対照が改良PRより安い。**PR内の改善は確認できても、強PFに対する総量子資源優位は確認できない。**

そのほか、D内部だけを消すYoshida4はmixed biasが残り、source-shot予算による価格診断で総RZ約6.00e12と元PRより悪化した。fragment1/2/3の再配分も主候補を上回らなかった。D自己対称S2は小さい診断改善に留まった。単一fragment再配分の実根なし、正時間の順序合成で三次相殺不能という結果も保持した。全体partial Y4は負R時間によりshot負担が増え、既存の負結果を覆していない。

## 成果物・再現・限界

[FINAL.json](../../../artifacts/track_c_d_internal/2026-11-08/FINAL.json)、[資源CSV](../../../artifacts/track_c_d_internal/2026-11-08/RESOURCE_TABLE.csv)、[登録比較](../../../artifacts/track_c_d_internal/2026-11-08/REGISTERED_COMPARISON.json)、[保証付きG](../../../artifacts/track_c_d_internal/2026-11-08/QUALIFIED_G.json)、[非共鳴対照](../../../artifacts/track_c_d_internal/2026-11-08/NONRESONANT.json)を参照。全raw・中間・区間証拠は[RAW_DATA.tar.gz](../../../artifacts/track_c_d_internal/2026-11-08/RAW_DATA.tar.gz)に元byte列で保存し、RAW_INDEX.jsonから照合する。private common.pklは公開せず、公開DF/state参照とhashを保持した。

科学SOURCE：初期`ed3902a1…`、再配分`dfc4d475…`、全round`f935da60…`、価格/強対照`cb445a40…`、初期保証`91f2ebee…`、非共鳴/測定意味`30aca97b…`、登録対照`7287bdc8…`。各完全SHAはFINAL.jsonに記録する。最大64 worker、初期価格wall153.58秒、RTE標本時点で観測したworker peak RSS最大501940 KiB。後続whole/区間処理のpeakを示す値ではなく、後続区間処理等もこの価格wallに含めない。

保存結果の読取検査は`python3 scripts/check_track_c_d_internal.py`。科学実行は専用common cache、PYTHONPATH=src、outer process/BLAS1で各`track_c_d_*.py`の段階入口を使用した。既存outputを上書きして再実行しない。テスト・検査はCodex自身によるもので、外部再現・独立した科学レビューは行っていない。

残る主要論点は、保存exact ground期待値による係数選択を実用的な近似情報で代替できるか、他状態/分子へ移せるか、T合成や状態準備を含めても利点が残るか、RTE減衰/価格の支配をどう軽減するかである。今回これらの別条件計算は実施していない。D精度だけをさらに上げても、m17の半径約.573というRTE負担は残る。

次の担当：ユーザーが、この条件付きPR改善を採択するか、係数推定の実用性・RTE側の改善等の次仮説を選択する。Codexは次の指示に基づいて新範囲を実行する。今回の物理条件・旧SOURCE/raw/grant/ledger/marker/STOPを保全し、公開照合後mandatory STOP。詳細GPT科学レビュー・別研究へ自動進行しない。
