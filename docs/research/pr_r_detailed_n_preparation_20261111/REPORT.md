# R可変PRの詳細N評価：候補・計算量・有限科学計画

2026-11-11 JST。**次の承認候補は短・中時間4ラウンドの12候補（P1）とする。最終ラウンドはH6の3候補（P2）、H4の3候補（P3）を別の未承認案として残す。** A0全24件を保持し、残る6件も未評価として保存した。Rの科学採択・順位・上限・減衰の合否閾値は設定していない。

今回は固定保存JSONとコード構造の調査、metadata計画生成、人工/保存fixtureテストだけ。normalizationを含む新しい科学数値、保存signalの再解析、実行時間benchmark、分子行列作用、N/C/G、sampling/compile、詳細科学レビューは全て0。既存科学kernel・B/C実装・過去STOPを変更していない。[EXECUTION_PLAN.json](EXECUTION_PLAN.json)／[B/C HANDOFF](HANDOFF.md)／[metadata専用生成器](../../../scripts/plan_pr_r_detailed_n.py)／[テスト](../../../tests/check_pr_r_detailed_n_plan.py)。生成器に科学execute機能はない。

## 固定条件と参照成果

Linear H4/H6・隣接1.00 Å・STO-3G、保存tol-only 1e−8 DF10/DF19、weighted-Hermitian completion、signed weight/fragment/product/application順、scalar一回会計、H/state/snapshot/sector bindingを維持する。H4はS2/δ=.1/PR prefix1/K2・sector36、H6はm5_best/δ=1.6/PR prefix7/K2・sector400。H6 m0..3の保存別構成は変更しない。PFは保存H4 B0/p7、H6 Full19を保持し、今回は新計算の対象にしない。

targetは保存DF-Hの同sector ground energy、保存数値stateの条件付き仮定、P=0。h=.1/全13round m0..12、εE=.00159360010199040 Ha、各方式αtotal=.05、各軸α=.05/26、βtotal/stat=.4/.2、無energy shift、保存prior/decoderをそのままmenuに保持した。E_stateはprior/branchの決定には使わない。厳密ground同定・actual implementation/law・rigorousな包含・全RPE成功は未資格である。

| 成果 | 固定公開commit | 再利用内容 |
|---|---|---|
| A0実結果 | `4745bb676568bb11d46552be42ffc6f4fba75385` | 24件の全normalization/attenuation/係数・精度statusを無変更copy |
| A R screening | `f98982118324bcc8402ed513b5c720b37bb03e87` | B metadata bridge、候補/law/allocation。元86候補や未探索上側を閉じない |
| A N provider | `5ca2dce3234f3e4a971f4a8c852da676f050e315` | RBatch/独立corrected-raw/exact-tail・R-aware scalar |
| B A0受入 | `d5d9823e17ecf78a82216fc02b7fcc13a21bb2cb` | A0全24受入済、C1共有はH6 m4 R16/32の2件 |
| B provider統合 | `ac3e1ce87b5ed7d9114abc316545366dd097209e` | RMenuBatch、同policy gate、context/reference共有、既存N/C interface |
| C R価格設計 | `8457c443554af5fadaa3ba36f97f6316ca3c5050` | 保存直接価格の条件、R/q外挿の区別 |
| C Phase1準備 | `c596e88d2a88fb219c2e373fb0d72c6e39cb723a` | 元8セルの予定domainだけ。以降の実行状況は主張しない |

各SOURCEはPLANのpinと保存blob hashから辿れる。新科学実行SOURCEはまだnull。Aの既存branch `codex/pr-r-dependent-n-model-20261111`を継続し、B/C worktree・processは操作していない。

## A0から判断できる範囲と選定

以下のAは保存attenuationであり、実raw信号半径や十分Nではない。exp/log、比、近似式を再計算していない。

- H4 m12：R4のA≈4.57e−37、R8でも≈6.14e−19。R8192/16384/32768は≈.960/.980/.990だがevent数は33,554,432/67,108,864/134,217,728。**R4→8だけでは、A0で観測した良好なnormalization帯を調べられない。** R8は小変更の対照として残す。
- H6 m12：R16/32はA≈.0542/.255、R2048/4096/8192は≈.979/.990/.995。R16→32は改善方向の対照になるが、より上側の帯まで評価した証拠にはならない。
- H4 m6のR128/256、H6 m8のR128/256は、A0の比較的減衰が小さい2帯を、最終ラウンドより少ないeventで調べられる。最終時間へN/位相を移送する根拠にはしない。
- H4 m0のR2/4/8とH6 m4のR11/16/32で、短時間のfinite law・最小整数allocation・R依存の比較を行う案。H6 R32はC Phase1の校正予定と一致する。

| package | system/round | 提案するR（baseline含む） | 提案しないA0候補 | 選定理由 |
|---|---|---|---|---|
| P1 | H4 m0 / T.1 / qPF1 | 2,4,8 | なし | 短時間の両側比較、既存baseline保存可 |
| P1 | H4 m6 / T6.4 / qPF64 | 4,128,256 | 512 | 小R baselineと減衰が小さい2帯 |
| P1 | H6 m4 / T1.6 / qPF1 | 11,16,32 | なし | 11piece最低整数配分、baseline、C1との共通点 |
| P1 | H6 m8 / T25.6 / qPF16 | 16,128,256 | 512 | 長時間へ進む前のR依存と同policy比較 |
| P2 | H6 m12 / T409.6 / qPF256 | 16,32,2048 | 4096,8192 | 小変更とA0上側帯の両方を有限点で区別 |
| P3 | H4 m12 / T409.6 / qPF4096 | 4,8,8192 | 16384,32768 | 小変更だけで終えず、最初のA0上側帯を留保 |

6件の保留は科学的FAIL・支配関係や価格優位を意味しない。NだけでRを決めず、Cが揃わない候補の最適性も主張しない。PLANの全24行にcanonical ID、完全request、piece別allocation、event、保存A0原recordとSOURCEを保存した。Bの広いmenuをこの18候補で置換する指示でもない。

## 既存backendの処理量・メモリ

根拠は固定SOURCEの `pr_r_dependent_n.RBatch`、`h6_df19_rpe_d2.evolve/half_step_keys`、`rte.finite_taylor_operator`、`run_h6_df19_rpe_d2.worker_resources`、B `RMenuBatch`。dはsector次元、Jはpiece数、pはprefix、qはqPF、E=qR_outerとする。

1. **準備**：保存loaderはsector内の一体行列・weighted G²・tail・full-Hを構成する。新SCF/DF/state/eigensolveは不要だが、この分子行列構成/作用も将来の科学実行範囲であり、今回ロードしていない。half-stepは(block,signed time/2)ごとにdense expmしてcacheする。固定δなのでRやroundをまたいで同じkeyを持つ。
2. **piece準備**：各J pieceでK2の3次finite polynomial行列とb、exact-tailのdense expmを作る。有限polynomialは既存のdense積を使い、初期化は固定Kで概ねO(J d³)。exact-tail指数の仕事量は行列norm・scaling等にも依存する。同じtail/T/q/weightのexact-tailは数学的にR非依存だが、現コードは各candidateで再計算する。
3. **伝播**：各outer・各pieceで2(p+1)回、d×d行列を3列へ作用し、exact-tail columnに1回作用する。さらに**r_j回のPython loop**でfinite stepをcorrected/rawの2列へ作用、rawをbで除算する。rawは独立伝播し、最後のcorrected/B代入にはしない。q全体ではE回の2列更新とE回のraw除算、2q(p+1)J回のhead作用、qJ回のexact-tail作用となる。
4. **診断**：head作用ごと・finite eventごと・最終scalar後に、全columnのfinite値とnormを検査する。checked呼出し数は2q(p+1)J+E+1。exact-tail作用直後だけの専用checkedはなく、その後のfinite checkedで含めて診断する。最終exact normとraw/corrected vector closure gateは維持する。Python loop、temporary、norm/isfiniteにも費用があり、BLAS積数だけでは時間を決められない。
5. **scalar**：保存signalのdisk・phase/radius・Hoeffdingと巨大整数ceil確認は候補当たり定数規模（整数の桁数には依存）。量子shot数Nそのものをloopする計算ではない。

伝播の主項はO([E+qJ(p+1)]d²)、診断はO([E+qJ(p+1)]d)、初期化はdense O(Jd³)等。**event列/trajectoryを展開保存しない**ので、メモリはEに線形には増えない。context/cacheと各candidateのstep/exact行列がO((rank+cache+J×同時candidate数)d²)、状態はO(d)。complex128行列1枚はH4で20,736 bytes、H6で2,560,000 bytes。並列candidateを増やすとこの行列群が増える。

| 候補 | E（更新回数） | head 3列作用回数 | finite積の複素dot項数 2Ed² |
|---|---:|---:|---:|
| H4 m6 R256 | 16,384 | 256 | 42,467,328 |
| H4 m12 R8192 | 33,554,432 | 16,384 | 86,973,087,744 |
| H6 m8 R256 | 4,096 | 2,816 | 1,310,720,000 |
| H6 m12 R2048 | 524,288 | 45,056 | 167,772,160,000 |

これはSOURCE loopの整数計数であり、科学数値再解析・実測FLOPs・wall見積りではない。H4は小行列でPython/診断 overheadが目立つ可能性、H6は400次元のdense積・指数初期化が支配する可能性があるが、新benchmarkは行っていない。eventが少ないH6をH4より軽いと断定できない。

過去の保存wallはH4全batch約11.203793秒（新11時点22候補等）、H6 m8..12 batch約88.604781秒（新5時点10候補等）。reference/PF/PR/並列環境を含むため、大R単独へ線形外挿しない。実行可能性について確定しているのは「明示event loopが存在し、event長に比例するmemory配列は不要」までで、P2/P3の所要時間は未測定である。

## 高速化・精度上の選択肢

現行の独立half-step/candidate ThreadPool、CPU affinity/load/MemAvailableと仕事数によるworker選択、プロセス内BLAS1によるnested pool回避を再利用できる。固定worker/CPU/RAM上限は新設しない。共有設定や他processは変更しない。

Bはcall内でcontext/referenceを共有する。half-step cacheもcontext上にあるが、**A RBatchは呼出しごとにkeyを計算し直して上書きする**ため、Bの複数R group間で指数評価の完全な再利用が実装済みとはいわない。同じkeyの既存値を使う小変更や、R非依存exact-tailのbinding付きcacheは次の技術改善候補。実装/科学検証は今回行っていない。

大Rには同一pieceのP_j^r_j、raw用(P_j/b_j)^r_jを**別々に**二乗法で作用する候補がある。pieceごとのhead/負時間/積順を保持すれば、exact arithmeticでは同じfinite meanの代数的再結合であり、新しいTaylor次数やlawへの近似置換は不要。piece powerの費用は概ねO(J d³ log R + qJ(p+1)d²)になり得るが、小Rや大dで必ず速いとは限らない。

ただし二乗法は浮動小数点の演算順・raw除算の位置・全event中間normの観測を変える。現行V1が要求するmaximum_intermediate_normを最終normで代用できず、単に高速化済として置き換えない。非可換人工系で旧loopとの独立比較、signed allocation、raw独立closure、normalizer丸め、適用可能なnorm根拠を固定する技術設計が先に必要。V1の意味を変えるなら別の科学条件として扱う。outer block全体の冪乗はhead順と診断にさらに影響し、今回の実行案には含めない。

大Rではτが小さくなりfinite bが1へ丸まる/underflowする問題、logBの丸め蓄積、長いcorrected/raw作用の誤差・中間norm、逆にV1 allowanceの演算数増加がある。A0精度statusが正常でも、新しいsignal/Nの精度資格にはならない。A0の1ULP感度はrigorous区間ではなく、新Nのuへ勝手に流用しない。A0のfsumとaction backendの逐次logB加算も演算順が違うため、保存A0値へのbitwise一致を強制せず、将来のactionが得た独立normalization証拠を残す。

## 保存reference・baselineと必要科学call

H4 m0/6/12は `89c91c...:run01/result.json` のscope_observed.referencesとPR diagnostics.signalに、独立2経路referenceとmaximum norm/closure/exact normが保存されている。H6 m8/12も `f4c0bd...:run01/result.json` に同様の証拠がある。PLANではfieldの存在・完全identityを確認しただけで、新しい差/phase/Nは計算していない。将来、SOURCE/H-state/Tと数値gateを再利用入口で検査し、欠落・不一致ならSTOPする。

**例外はH6 m4**。確認した `45607a...:result.json` は保存scalar評価で、独立expm_multiply/expm pairとexact-tail norm evidenceが揃わない。元signal carrierも既存normalization/closureだけから不足normを0補完できない。よってP1にT1.6の新2経路referenceとR16 baselineの新finite propagation/内部対照1を**明示的に**含めた。全履歴に別の適格pairがないと断定しているのではなく、今回確認したSOURCEに基づく有限案である。

| 未承認案 | 対象数 | 新finite mean / exact-tail内部対照 | 保存baseline scalarのみ | scalar/phase/radius/N（軸数） | 新reference action |
|---|---:|---:|---:|---:|---:|
| **P1を先行提案** | 12 | 9 / 9 | 3 | 12（24） | 2：H6 T1.6のみ |
| P2：H6最終時間 | 3 | 2 / 2 | 1 | 3（6） | 0：保存pair前提 |
| P3：H4最終時間 | 3 | 2 / 2 | 1 | 3（6） | 0：保存pair前提 |

新finite meanの各callにcorrected/raw独立伝播とexact-tail columnが含まれる。内部対照は別candidate探索ではない。P1は新R8候補＋証拠不足baseline1、残るbaseline3は保存signalのscalarだけを同policyで再評価する。P2/P3の大Rには必要なら上記技術改善を先行させる。三案の間に自動移行はない。

全Nは `R_AWARE_EMPIRICAL_ACTION_COUNT_V1` を使う。旧baseline Nを別policyの新R Nへそのまま比較しない。V1のstage数・中間norm・closure・pair差から既存u/logB marginを作り、同じD1 disk/geometry/Hoeffding、Decimal140/180整数ceil確認へ接続する。経験的allowanceはrigorous上界でない。baseline raw/corrected signalや旧Nの原本を変更せず、新しいSOURCEの結果に別recordを出す。H4整数在庫は `N_HANDOFF.json.shots` を整数のままcopyしてあり、旧rawやfloatへ戻していない。

## Cとの接続と実行判断

A0内で保存direct有限cohort価格を持つ確認範囲はH6 m4/R16。H6 m4/R32はC Phase1の新校正予定で、実価格取得済とは扱わない。他の22候補には今回確認したC Phase1の直接セルがない。H4のC1はqPF8/16・R4/8、H6はqPF1/2・R16/32である。H4 m0/q1やm6/q64、H6 m8/q16、両系m12は小qモデルの適格domainとは扱わない。固定Rの長qは未検証移送、新R/未観測allocationはさらにR domain不足を保持する。

Cの有限cohort/nominal covarianceを全law期待値やtotal CIへ昇格しない。長q・大R/希少order・actual sampler距離・cross-R/round covarianceは未資格である。Aは価格取得/fit/モデルを実装しない。将来BがN×Cを比較できる完全候補bindingを渡すことと、Cの資格が成立することを分ける。

実行前の最小技術作業は、保存referenceの明示callback（m4以外は欠測時STOP）、baseline保存signal→既存analyzeの小adapter、H6 m4の明示pair/再伝播、B request/law付きexportとscope集計の結線。科学kernelは再実装しない。現PLANは科学runnerではなく、科学SOURCE/この結線を固定してから、許可されたpackageだけを既存RBatchへ渡す。FAILは記録し、後続未実行と区別、threshold/R/precisionを変えた科学retryはしない。

技術検査は新規8件＋既存R provider人工回帰12件＝**20件PASS**。[TESTS.txt](TESTS.txt)・[回帰記録](N_PROVIDER_REGRESSION.txt)・[CHECKS.json](CHECKS.json)。作者自身の確認であり、独立科学レビューではない。

**ユーザー**が、まずP1の12候補・新reference2 action・新mean9/対照9・保存baseline scalar3・全12候補のX/Y十分Nを承認するか判断する。**Codex A**は許可されたpackageだけを実行し、**Codex B**は公開N recordを同policyで受入れ、**Codex C**は別途許可された価格domain検証を進める。P2/P3と保留6候補のN/C/Gは自動実行しない。今回成果をcommit/pushし、remote SHA/tree/主要成果物照合後mandatory STOPする。
