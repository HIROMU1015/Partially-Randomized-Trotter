# O2 B2-a / B1 保存値再解析：結果

2026-11-06 JST。**固定保存値の限定再解析を完了。Mandatory STOP。**

432候補すべてでexact finite-K normalizationが保存log Bを再現した。二次leadingと連続二乗則にはfinite-τと整数配分の異なる制限がある。固定RZ nomineeをCXで評価すると、一部の比較方向が反転した。強いPF envelope対照に対するB3の負結果は、この固定域のCX点比較でも残った。

今回の結果は、元研究の正式status、案X、MC1–MC4を採択・変更するものではない。サーバー上の独立cloneで行った保存値再解析であり、独立第三者再現・CI・実量子shot・実機runtime評価ではない。

## 固定条件とSOURCE

条件・事前判定は[preflight](PREFLIGHT.md)、全候補と価格/N/cohortの参照は[B2-a固定対象](B2A_FIXED_CANDIDATES.csv)・[B1固定比較](B1_FIXED_COMPARISONS.json)、取得98ファイルのcommit/blob/hashは[入力索引](INPUT_INDEX.json)に固定した。新しい候補選抜、shot最適化、sampling、RNG replay、回路build/compile/transpile、QPY decode、H/state/reference生成は行っていない。

DFはLinear H4/H6、隣接1.0 Å、STO-3G、保存tol-only 1e-8のDF10/DF19、元H/state/reference、H6 weighted-Hermitian completion。B2-aはB2 prefix3/5、B3 two-body prefix0（one-bodyを決定論的に保持）、S2/Yoshida4/m5_best、T0.8/2.0、q1/8、R_outer16/32/64、K2/4/6。δ=T/qは0.1/0.25/0.8/2.0の固定4値である。

B1長時間/B3は元40セル（T0.8/1.2/1.6/2.0、5精度）のRZ nomineeをそのまま使用した。各armのformula/split/q/R/Kは固定manifestに記録した。Pauliは各保存JW Hamiltonian/state/reference、Linear H2/H4/H6、1.0 Å、STO-3G、14/184/918 terms、T0.8、S2、元control/full-wrapper/P0・状態準備除外・仮想ρ0の条件。H8/P3/旧DF–Pauli比較はそれぞれ元のstate/control/partitionを保持し、表現間の因果比較へ統合しない。

Provider `c31ef424175c02379eafebe8fc6c7548c1d9aabe`、Signal `e417ffc9f52b13c18f5493193ce510d858664259`、Native `27ebabdd2943a4de787ec8b976c38e06fda5a497`、Continuation `ff323f1b09f96c0ce01955728832ff6b0f4c230f` のroleを区別した。最終会計はcarrier `155529e20a8fac39e00786c40b57830b71203c4f` のreport版、B3確認baselineのoverlayは `ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed` の最終report規則に従う。[再利用関数とhash](SOURCE_SPECS.json)の元純粋関数を使用し、科学runnerをimportしていない。

## B2-a：再現と近似の適用域

A=exact finite-K、B=元整数配分の二次leading、C=連続絶対時間二乗則とする。K0は人工テストのみ。B2/B3の保存λ_Rを別に使用した。

**432/432 EXACT_REPRODUCED**。最大絶対差はB2で6.94e-18、B3で0（使用binary64演算における値）。事前allowanceは1e-10。独立な直接power/factorial有限和による検算も全432候補で通過した。SOURCE定義のnormalization整合を支持する。

B2全216候補とB3の144候補は最大|τ|≤1の事前scalar residual boundを満たした。B3の72候補は最大|τ|>1のためAPPROXIMATION_LIMITED。INCONSISTENT/UNRESOLVEDは0。APPROXIMATION_CONSISTENTは次数整合の意味であり、全行が実用上十分なaccuracyであるという判定ではない。

以下の相対量の分母はすべてA。finite-τ誤差は(B−A)/A、配分効果は(B−C)/A、連続式誤差は(C−A)/A。各行は上記の固定域内72候補の集約である。

| family | formula | 最大τ絶対値の範囲 | finite-τ誤差 最大 | 配分効果 最大 | 連続式誤差の範囲 |
|---|---|---|---|---|---|
| B2 | 2nd | 0.0009114–0.1148 | 0.7844% | 0% | 4.615e-05%–0.7844% |
| B2 | 4th | 0.004137–0.5211 | 14.52% | 0.06445% | -0.05538%–14.46% |
| B2 | 4th(m5_best) | 0.001616–0.2545 | 2.635% | 7.9% | -7.695%–-0.3625% |
| B3 | 2nd | 0.02144–3.506 | 377.8% | 1.482e-14% | 0.02554%–377.8% |
| B3 | 4th | 0.09735–15.92 | 3593% | 2.078% | 0.4393%–3591% |
| B3 | 4th(m5_best) | 0.03803–7.773 | 814.1% | 70.36% | -6.426%–743.8% |

B2でもYoshida4の整数leadingは最大14.52%の過大評価となる（H6/DF19、prefix5、T2.0、q1、R16、K2）。したがって約10%を一律に保証できない。m5_bestは11stageの最低1配分が効き、τが小さくても連続式がexactを過小評価する。

次表はB2 H4/DF10、prefix3、T0.8、q8、K6、m5_bestの固定対照である。R以外を保持した。integer leadingが高精度でも、continuousとのずれが残る。

| R_outer | 元整数配分 | r_i=1 stage数 | finite-τ誤差 | 連続式誤差 |
|---|---|---|---|---|
| 16 | [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1] | 6 | 0.002511% | -7.695087% |
| 32 | [3, 3, 3, 3, 2, 5, 2, 3, 2, 3, 3] | 0 | 0.000485% | -1.663659% |
| 64 | [5, 6, 5, 6, 5, 10, 5, 6, 5, 6, 5] | 0 | 0.000114% | -0.483515% |

B3の大τ条件ではK依存も大きい。H6/DF19、two-body prefix0、one-body retained、T2.0、q1、R16、Yoshida4を固定したK対照を示す。最大|τ|=15.9168であり、二次式の適用域外である。

| K | A exact log B | B integer leading | finite-τ相対誤差 |
|---|---|---|---|
| 2 | 103.392565 | 3818.265612 | 3592.979% |
| 4 | 144.434235 | 3818.265612 | 2543.601% |
| 6 | 174.317154 | 3818.265612 | 2090.413% |

この結果は、元finite-K定義・signed schedule・保存λの連鎖が定量的に整合することを支持する。連続二乗則はsmall-τと整数配分補正を明示する近似として使用すべきである。二乗構造にはGüntherらv2付録A.3の直接先行があり、新理論の発見とは扱わない（[前回のSOURCE/先行研究調査](../o2_claude_review_followup_20261106/PRIOR_ART_MATRIX.md)）。

normalizationから総費用Gの原因、bias/headroom、最適shot overheadを特定していない。B2-bへの進行は行っていない。

## B1：固定NでRZからCXへ変更した結果

`G_CX=N_cos*C_cos,CX+N_sin*C_sin,CX`。元N・RZ選抜・cohortを固定した。[全結果](B1_CX_SENSITIVITY.csv)の592 armで既存RZ会計を照合し、322 armはexact一致、270 armは元保存表現の精度内で一致した。CX向けの新候補最小化は行っていない。

左側は、DF長時間=PR、B3=B3、Pauli=Hybrid、H8=PR。B3は同じ40セルを3基準と比較しており、120の独立実験ではない。探索と確認の行も独立した科学条件数として加算しない。

| study | 固定比較数 | RZで左低G | CXで左低G | 方向反転 | 不確かさ |
|---|---|---|---|---|---|
| DF_LONG_40 | 40 | 33 | 34 | 1 | 点比較 / SE UNKNOWN |
| DF_B3_EXPLORATION | 120 | 0 | 1 | 1 | 点比較 / SE UNKNOWN |
| DF_B3_CONFIRMATION | 120 | 0 | 1 | 1 | 点比較 / SE UNKNOWN |
| PAULI_LATEST_15 | 15 | 1 | 0 | 1 | paired sample SE / 記述的区間 |
| H8_STAGE_C | 1 | 1 | 1 | 0 | paired sample SE / 記述的区間 |

DF長時間では39/40の方向が維持され、RZでPF envelopeが低かったH4/T0.8/ε0.05だけがCXでPR低費用へ反転した。これはS2 PR prefix3/q1/R4/K2対S2 B0I prefix5/q1で、元探索PR n9と決定論価格n1を用いた点比較である。

B3はB2およびPF envelopeに対して、探索・確認とも40/40で高CX費用だった。一方、H4/T0.8/ε0.0005ではFull PF単独に対してCX点方向が反転した。B3 S2 prefix0/q16/R16/K4対Full PF m5 prefix10/q1である。強い決定論envelopeに対する負結果と、Full PF単独に対する例外を区別する。

Pauliは14/15の方向が維持された。H4/ε0.05のHybrid p36/q2/r8/K4（確認n48）対pure RTE p0/q8/r8/K2（確認n384）だけがHybrid低RZからpure RTE低CXへ反転した。他14行もpure RTE低CXであり、固定15行すべてでpure RTEのCX点費用が低い。H6は現行統合資料指定のHybrid p78/q8/r32/K2確認n64を使用した。旧latest15のH6 r8候補への取り違えはない。

H8 Stage C（Linear H8、1.0 Å、STO-3G、DF28、T1.6、PR m5 prefix8/q1/R32/K4確認n9、Stage Bの固定m5 B0I prefix15/q1決定論n1）ではPR/PF比がRZ0.754109からCX0.707837となり、方向が維持された。固定Nと同じ9 paired価格の記述的結果である。

### 方向が反転した全行

比は左/右。DFのCX SEはUNKNOWNであり、点方向の反転だけを主張する。

| study | system | T | ε | 右側基準 | RZ比 | CX比 | CX不確かさ |
|---|---|---|---|---|---|---|---|
| DF_LONG_40 | H4 | 0.8 | 0.05 | PF | 1.014846 | 0.907930 | UNCERTAINTY_UNRESOLVED |
| DF_B3_EXPLORATION | H4 | 0.8 | 0.0005 | Full_PF | 1.224869 | 0.823828 | UNCERTAINTY_UNRESOLVED |
| DF_B3_CONFIRMATION | H4 | 0.8 | 0.0005 | Full_PF | 1.221627 | 0.822784 | UNCERTAINTY_UNRESOLVED |
| PAULI_LATEST_15 | H4 | 0.8 | 0.05 | RTE | 0.951156 | 1.240487 | DESCRIPTIVE_INTERVAL_SEPARATES_ZERO |

Pauli H4/ε0.05のCX差（Hybrid−pure RTE）は3,670,450.55、paired sample SEは69,069.31、条件付き記述的95%区間は[3,535,074.69, 3,805,826.40] CX×shotである。これは固定選抜・元Nに関する標本評価であり、母集団tailや全候補winnerの保証ではない。

Pauli15行とH8について、元の独立取得cohort/IID仮定の下でcos/sin共分散を含めた差のsample SEを計算した。今回の記述的区間は全16行で0を跨がなかった。ただし、同じ価格標本を共有する精度行、確認後minimum、逐次延長、未観測rare order、PRNG独立性未証明を含む。多重比較に対するfamily-wise保証や新しい母集団保証へ格上げしない。H8のrare-order条件付き費用は依然UNKNOWNである。

DF長時間/B3は保存meanだけでCXを評価した。decimal meanに対するFraction加算を用いたが、元trajectoryの厳密mean・CX SEを復元した意味ではない。RZのSEを転用せず、UNKNOWNを0に置いていない。

### H4旧比較とP3

H4旧DF–Pauli比較はcentral系列の価格がある一方、GenericPauli/DF側の同policy・同case価格coverageが未確立のため、DATA_UNAVAILABLEを1行保全した。追加matched compileや異なるcohortの代入は行っていない。

P3はH4/DF10、共有Gaussian state・partition A、q8/r4/K4、T0.8、仮想ρ0・元engineering/finite-bit条件を保持した保存Stage4の10arm×precision行から、既存RZ/CX会計を[そのまま引用](P3_EXISTING_CX_QUOTED.csv)した。新しいdepth会計・価格・候補選抜・比は作っていない。既存depth値の存在は前回調査に記録されている。

## MC1–MC4・案Xへの影響と残る判断

- **MC1**：固定した主要PR/PF比較の大半でCX点方向が維持された。ただし39/40であり、全指標・全条件での優位性とは言えない。DFの不確かさを含む主張にはCX rawの取得が必要である。
- **MC2**：finite-K normalizationの実装整合は支持された。二乗近似の用途にはfinite-τと整数配分の制限を併記する必要がある。直接先行との差分は、既存DF実装・高次PF・splitの具体的実証に置くべきであり、新原理の採択は行わない。
- **MC3**：B2/PF envelopeを基準とする有限候補域のB3負結果はCX点比較でも残る。Full PF単独には例外がある。「どのPF対照にも不利」「一般的に不可能」とは主張できない。
- **MC4**：各表現内でmetric感度を確認した。Pauli H4の1行は明確なmetric依存例となる。異なるcontrol/law/state/partitionを越えた、Hamiltonian表現だけの因果効果は認定しない。
- **案X**：本結果を追加判断材料として返す。採否、MC2の位置付け、論文化方針を自動決定しない。RZとCXの両方を指定したclaimへ限定することが必要である。

今回の課題を終えるための新compile/samplingは不要。DFの不確かさまで記述する必要がある場合は、まず同候補・同cohortの既存paired CX rawの回収を検討する。normalizationからGの原因まで説明する必要がある場合は、別途B2-bの最小範囲と科学的理由をユーザーが判断する。新しい研究・追加レビューを開始していない。

## 検証・再現・原本保全

人工テスト13件、結果検査6件がPASS。後者は直接power/factorialによるnormalization検算、固定candidate/cohort/Nの全照合、mean-onlyのSE欠測維持、Pauli/H8のpaired標本数、592件のRZ参照検査を含む。新しい学習・fit・RNG利用はない。

実装差は固定資料を読むadapterとA/B/C・CX感度の集計のみ。B3確認のN参照は元precision表、RZ照合は元independent_comparisonsの確認費用へ結合した。preflightでN参照をRZ参照にも流用していた箇所を、実行時に元cohortの記録へ正しく結び直した。候補、N、cohort、許容差は変更していない。

既存checkoutのHEAD・dirty/untracked状態・5資料hash、Claudeレビュー原本、直前追補の9公開ファイルとclean HEADを照合する。共有設定・他job・旧結果・grant/ledger/marker/STOPには変更を行わない。本branchの新規成果物のみを公開し、正式な研究statusを変更しない。

再現方法：`INPUT_INDEX.json` の各固定GitHubバイトを `<inputs>/<commit>/<path>` に復元し、Python3.12.3 / NumPy1.26.4で次を実行する。`fetch_inputs.py` はそのバイト取得専用であり、科学runnerを起動しない。

```bash
python -B fetch_inputs.py --out /path/to/dedicated-inputs
python -B run_reanalysis.py --inputs /path/to/dedicated-inputs --task all
O2_PREFLIGHT_INPUT_ROOT=/path/to/dedicated-inputs python -B test_synthetic.py -v
python -B test_results.py -v
```

結果・SOURCE・未確認事項・公開照合は[VALIDATION.json](VALIDATION.json)を参照。Remoteの最終SHAはチャットの公開リンクと専用領域の照合記録で確認できる。

**Mandatory STOP。B2-b、新しい候補/shot政策/metric、sampling/compile、案Xや最終科学claimの採択へ進まない。次の研究判断をユーザーへ戻す。**
