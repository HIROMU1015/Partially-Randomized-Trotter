# PR-DF LD1：F0分布差から残差価格差への条件付き上界

2026-11-15 JST。固定SOURCEの静的調査と、許可された全support上界・保存TVとの積の計算。

- **条件付き分布差をRZ単位に変換できた範囲**：全13 round・X/Yの**26/26軸**。C−Sの直接の鋭い幅は未証明だが、CとSを別々に抑える粗いSOURCE上界を得た。F0の条件付きpath TVとの積は、m0約3.056e−10 RZからm12約49,770.254 RZ。これは上界であり、観測誤差や価格補正値ではない。
- **実PCG64へ移す際に残る仮定**：F0の独立一様53-bit格子lawと、実PCG64の固定seed又は明示したseed ensembleのjoint lawとの接続は未証明。全26軸のactual-PCG価格差欄はnullを維持した。compilerについても、指定した標準Qiskit 1.3.0 SOURCEの動作を前提とする。
- **次に最も必要な証拠1件**：元normal samplerの**PCG64・seed ensembleからF0のIID-grid joint lawへの接続根拠**。固定seed列をIIDの確率標本と見なさず、対象とするseed確率測度、乱数消費・outer stream再始動、joint lawの距離又は有効なcouplingをSOURCEと対応付ける。これがなければ、今回のRZ上界を保存Q cohortからideal-lawへ移せない。

[26軸のexact結果](RANGE_EVIDENCE.json)／[表示用CSV](AXES.csv)／[compiler根拠](COMPILER_ROUTE.json)／[導出SOURCE](DERIVE_ENVELOPES.py)／[C向けHANDOFF](HANDOFF_C.md)。必要精度・信頼水準・価格許容誤差は未採択で、PFの2%を使わない。

## 1. 固定対象・§10.5の再利用

保存linear H4/1.00 Å/STO-3G、tol-only DF10、PR-DF LD1/S2/Δ=.1/K2、元exact κ/H/state/snapshot・fragment/basis/qubit順、scalar c+s_Rの総T一回、symmetric-directional control、P0、測定付きfull controlled Hadamard X/Y wrapper、状態準備費除外。Cはnew ZZ `CONTROLLED_ZZ_CX_CRZ_CX_V1`、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、backend/coupling/layoutの追加指定なしのcompiled RZ数。Sは元Accountingのfold後の構造RZ数であり、新しい構造量へ置き換えていない。

| 固定参照 | 使用範囲 |
|---|---|
| F0最終 `d749ef92e8ac3087747481a46aeb28ec4ca04c4b`、実行SOURCE `fad0c1e4fe41c0e4c6d0690158056572295ed15a`、raw `ec78fde57e117e9bbf79a26bfaa2e1575c1f5784` | 保存path TV13件を読み取る。F0の再実行・表の再作成なし |
| E[C]計画 `950577f416c17a20d309a8053a4bb016027b724d`、`AXIS_EVIDENCE.json` | 26軸・physical ID・B/C configuration対応・receiptの入口を継承 |
| C受入 `8eef0ad7…` が指定する `c99d511e2422b8f250490b11054f754a81e536c5:…/ADMISSION.json` | 保存E_I[S]・P1/P2 fresh/replayを区別 |
| 凍結正本 `3cac53da1a4a8b2228245593c74c1ff9ec1f5212`、m4最終受入 `9edfa5d3201f22c23dd77569d102801be80dacf3` | 役割を保全。価格の再集計・再fit・m4の較正再投入なし |
| C F0受入 `259a0b2d7357d740502d6837db2cf3c2a8b75c5e:docs/research/pr_df_ld1_f0_admission_20261115/HANDOFF.md` | 新しい差分：14診断/13上界/26対応を受入。全support残差幅は未供給。受入完了を今回の依存条件にはしていない |
| Dのcompiler調査 `dd435d07e3f0490165ff49889be9c71f573bd5db:docs/research/d_pf_unobserved_round_closure_20261115/REPORT.md` §3、PF監査 `606881b042ed69f9a1e7f00311417f2b65855d7b` | 共通Qiskit経路のSOURCE調査だけを再利用・補完。PFの周期閉包、凍結価格、保存QPYの実測結果はPR-DFへ移していない |

950577f4/d749ef92の重複照合を再利用し、旧残差調査、P2、mean-price準備、構造kernel、builder size bound、関連C/D/Rの最新refを限定照合した。今回のPR-DF全support数値上界の既存完成品は確認範囲で0件。Dの共通compiler調査を再利用し、不足するPR-DF alphabet・展開係数・TV積だけを追加した。関連local実行名に同等range jobなし、出力先は今回の未追跡草稿だけだった。アクセス不能な保存先まで不存在とは主張しない。ref一覧・固定SOURCEとgit blobはRANGE_EVIDENCEに保存した。

## 2. TVから期待残差差への条件

round m、軸aを固定し、Ωを保存ideal-source path law μと条件付きgrid path law νの**supportの和集合**とする。ωはouter順を含む全event label列で、各labelはorder0/2と、rotation選択index・順序付きproduct indexを含む。idealで正・gridで0になったlabelをΩから捨てない。

f(ω)=C(ω)−S(ω)がΩ上で同じ有限実数関数として定義され、μ,νがこのΩ上の確率測度で、L≤f(ω)≤UがΩ全体に成立するなら、

```text
TV(μ,ν) = 1/2 sum_ω |μ(ω)−ν(ω)|
|E_μ[f]−E_ν[f]| ≤ TV(μ,ν) (sup_Ω f−inf_Ω f) ≤ δ_m (U−L).
```

f−LをU−Lで割った[0,1]値の関数へTVの双対評価を適用する。余分な係数2は不要。fが負になり得るため、上側の絶対値だけを片側rangeへ流用しない。compiler/buildが未定義又は失敗するlabelを価格0へ補完してはいけず、その場合は価格関数の対象scope自体の解決が必要である。

同じlabelが同じ回路/構造量へ写る根拠は次の固定経路に限る。

1. provider `c31ef424…:rte.py#_make_event`：component選択の最初がrotation、残りがproduct。回路作用順はproducts→rotation、phaseは `(-1)^(order//2)`、angleは保存τから `atan(τ/(order+1))`。componentのsign/basis/support/順序も固定する。
2. [Compose](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/scripts/h46_long_time_native.py) と [Schedule](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/src/trotterlib/h46_m5_plan.py)：S2のpieceは1、outerは元q/R/T。元requestを同順にcomposeし、head→tail→reverse、scalar総T一回。seed/確率質量はword生成へ再乗算しない。
3. [LegacyBuilder](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c31ef424175c02379eafebe8fc6c7548c1d9aabe/src/trotterlib/df_rte_qiskit.py) / [NewZZ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39ddd59edf54a3a901e5302207b2765ddbd03fbb/src/trottertracks/efficient_controlled_zz.py)：旧providerのbasis・product・relative phaseを保ち、制御ZZ rotationのみCX–CRZ–CXへ置換。μ/νの両方で**同じ新ZZ**を使う。旧legacy lowering価格と混ぜない。
4. [FastPath](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39ddd59edf54a3a901e5302207b2765ddbd03fbb/src/trottertracks/compiled_price_fast.py)：同じrequest/event列をSのAccountingとCのcompose/wrapperへ渡す。X/Yは同じpathの別関数f_X/f_Yで、Yのsdgやwrapper/scalarも含む。
5. [CompilerCall](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c31ef424175c02379eafebe8fc6c7548c1d9aabe/src/trotterlib/rte_compiled_cost.py) と [Context](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7/src/trottertracks/r_price_phase1_science.py)：backend/coupling/layoutの追加指定なし、basis/opt/seed/versionを固定し、元compilerが返したcompiled回路のrzを数える。同一のcompiler/runtime条件を価格関数の定義に含める。

これは固定word生成経路の対応であり、全入力の量子作用検証を追加した主張ではない。保存full-wrapper statevector検証=false等の旧gate範囲を保持する。別preparation・別順序・S′・別compiler/backend/coupling、旧ZZ、frozen予測値を用いた関数には自動移送しない。


## 3. 証拠の分類と全supportの構造量上界

- **C−Sに直接使える全support幅**：未証明。direct_residual_lower/upperは全26軸でnull。
- **CとSを別々に制限する幅**：今回のSOURCE条件下で全26軸に得られた。以下の粗いenvelopeを使う。
- **有限pathの観測範囲だけ**：P1/P2、m3、m4のreceiptとモデル検証。これらを全support幅として使用しない。m0/m3/m6は較正、m4は独立holdoutを維持する。m4の−0.544502%は各軸4 pathの平均に対する予測差で、未観測label・長round・law差の保証ではない。

元builderのuntranspiled instruction capとcompiled RZは単位が違う。特に旧tail estimatorのrotation=1命令を、new ZZのCX/CRZ/CX=3命令の上界として転用しない。新しいS boundはAccountingとGaussian/head/tailの文法から別途導いた。

8-mode Gaussian basisは最大28 Givens、初期Pは最大8、各GivensはXXPlusYYと2P。保存preparationのhead4訪問・tail basis9件を読み、wire0..7、alphabet={p,xx_plus_yy}、XX数≤28、P数≤64を確認した。AccountingのXXPlusYYリテラル展開はRZ換算10。従って1 basis当たりB_S≤8+28(10+2)=344。逆basisは同じgate型・同じ個数上限である。

headはone-body＋LD1 fragmentの2 blockをuncontrolled/directionalで各1回訪問し、basis8本、対角RZは8+36+9+37で、H_S≤2842/outer。K2 eventはproduct最大2＋rotation1、各applicationにbasis対を割り当てるとtail≤6B_S+2×8+2=2082/event。identityはbasisを増やさずphaseに集約される。relative phaseはtail requestにつきP最大1、S2はouterにつき1 request。wrapper H2個・scalar1個でXは+5、Yのsdgで+1。basis相殺・Z-foldを無視するのは上側に安全であり、

```text
0 ≤ S(m,a,ω) ≤ U_S(m,a) = q(2843 + 2082 R) + 5 + 1[a=Y].
```

これはE[S]の再計算でもpathの生成でもなく、固定のq/Rに閉形式整数boundを代入したもの。

## 4. 全support compiled上界：実際のcompiler経路

**RZ数の単調減少は仮定しない。全basis gate数でCを抑える。** 固定compiler呼出しにbackend/coupling/layout/error propertiesはない。installed Pythonの入口と、Qiskit1.3.0 tagのcommit `896d1c8b72888a52a46081a6d1f4de8c56db8aea` のactive Rust SOURCEを読む。Dの共通compiler調査に加え、BasisTranslatorのcost規則を確認した。

1. initはinverse cancellationのみ、layout/routingなし。UnitarySynthesisの対象UnitaryGateやcustom HLS objectは今回のalphabetにない。HLSはequivalence libraryに登録された標準gateを展開せずtranslatorへ渡す（high_level_synthesis.py:639/813）。
2. [登録済みequivalence](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/circuit/library/standard_gates/equivalence_library.py) ではU3→RZ/SX/RZ/SX/RZ。P→U1→RZ、H→U2→U3、S/Sdg→U1、SXdg→RX→R→U3、RY→R→U3から、必要な非basis 1q gateの分解は一律≤5 gateで抑えられる。CRZ≤4、CZ→H/CX/H≤11、RZZ≤3。XXPlusYYの登録済み14命令は6RZ＋1SX＋2CX＋S/Sdg/SXdg/RY/RYなので≤34。角度の特殊値で短くなることには依存しない。
3. [basis_search.rs:96–153](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/basis/basis_translator/basis_search.rs#L96) はtarget basisの各gateをcost1とし、ruleのcostを置換先の総costとしてDijkstra探索する。従って選ぶgate別変換の総gate数は、上の有効な登録済み分解のcostを超えない。制約付きtargetへの局所変換やroutingは今回含まれない。
4. opt1のactive [Optimize1qGatesDecomposition](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/euler_one_qubit_decomposer.rs#L1199) はbasis内runを、長さが減る場合等に置換する。targetなしではerror tupleは(length,length)、error指定なしtargetなら(0,length)。outside_basis=false、旧誤差も新誤差も0の分岐は長さ増加を許さない。InverseCancellationは削除のみ。出力も同じbasis内なので条件付き再translationは空。fixed-point loopを繰り返しても**全basis gate数は増えない**。RZだけは1q再合成で増える可能性があるため、その単調性は使わない。

以上から、C≤最終全basis gate数≤最初のtranslation全gate数。同じ有限alphabetのすべてのlegal wordに適用でき、観測compileから外挿していない。

| 項目 | 全basis gate数の上界 |
|---|---:|
| Gaussian basis B_C | 8×5＋28(34＋2×5)＝1272 |
| head/outer | 8B_C＋(8＋8＋28×3)＋(8×3＋1)＋(8×3＋28×5＋1)＝10466 |
| K2 tail/event | 6B_C＋2 product×2 CZ×11＋new ZZ(2CX＋CRZ4)＝7682 |
| tail relative phase/outer | P≤5 |
| wrapper＋scalar＋measure | X≤16、Y≤21 |

```text
0 ≤ C(m,a,ω) ≤ U_C(m,a) = q(10471 + 7682 R) + 16 + 5·1[a=Y].
```

measureを含めた全命令数で抑えるため、RZ上界としてさらに緩い。compilerが受理する角度の全域で使い、identity、basis共有、ゼロ角、compiler相殺を最大側へ無視した。通常composeのcapはhead bound＋実tail長、wrapperは実evolution長＋4なので、旧tail estimatorのrotation=1を使わない。全wordの機械実行成功・OOMなしを観測したという主張ではない。生成/compileの失敗を0価格として扱わず、数学的価格関数は指定SOURCEが正常完了するものとして定義する。正常完了はsupport和集合全体での定義条件であり、成功pathだけへの条件付け・再正規化はしていない。実資源下の全support取得可能性を保証せず、定義条件を満たさないruntimeへはこの積を移さない。

**適用条件**：元preparation、元word生成、新ZZ、標準equivalence/default plugins、backend/coupling/error指定なし、opt1の当該pass列、SOURCEに従うaccelerated routine。installed Rust binaryを過去receiptとbit単位で同一だと証明したものではなく、指定した標準release algorithmのSOURCEに基づく上界である。custom pass・別compiler・別loweringへ移さない。補助Python関数だけをactive pathと見なす誤りを避け、実際に呼ばれるRust routineを確認した。詳細なpath/行番号/installed Python hashはCOMPILER_ROUTE。

## 5. 保存TVとの積：13 round・26軸

```text
-U_S ≤ f=C−S ≤ U_C
osc_union(f) ≤ Ω = U_C+U_S = q(13314+9764R)+21+6·1[a=Y]
B(m,a) = saved_F0_path_TV_upper(m) × Ω(m,a) [RZ].
```

両分布で正のlabelの和集合を覆い、ideal正/gridゼロの32 componentも除かない。directなC−S幅のtightさは得ていない。下表は表示用丸め値で、exact分子・分母はRANGE_EVIDENCEの各行に保存する。元F0表の再計算はしていない。

| m | q / R | Ω_X / Ω_Y | B_X (RZ) | B_Y (RZ) |
|---:|---:|---:|---:|---:|
| 0 | 1 / 2 | 32863 / 32869 | 3.05589899e-10 | 3.05645692e-10 |
| 1 | 2 / 4 | 104761 / 104767 | 3.86904977e-09 | 3.86927136e-09 |
| 2 | 4 / 8 | 365725 / 365731 | 5.43294909e-08 | 5.43303823e-08 |
| 3 | 8 / 16 | 1356325 / 1356331 | 7.97200761e-07 | 7.97204288e-07 |
| 4 | 16 / 32 | 5212213 / 5212219 | 1.22525238e-05 | 1.22525379e-05 |
| 5 | 32 / 64 | 20422741 / 20422747 | 0.000191590023 | 0.000191590079 |
| 6 | 64 / 128 | 80838805 / 80838811 | 0.00303665429 | 0.00303665452 |
| 7 | 128 / 255 | 320401173 / 320401179 | 0.0482819069 | 0.0482819078 |
| 8 | 256 / 510 | 1278196245 / 1278196251 | 0.764181618 | 0.764181622 |
| 9 | 512 / 1020 | 5105968149 / 5105968155 | 12.1195101 | 12.1195102 |
| 10 | 1024 / 2040 | 20410238997 / 20410239003 | 195.792867 | 195.792867 |
| 11 | 2048 / 4080 | 81613688853 / 81613688859 | 3147.96585 | 3147.96585 |
| 12 | 4096 / 8160 | 326400221205 / 326400221211 | 49770.2535 | 49770.2535 |

m8まではこの条件付き上界が1 RZ未満、m9以降は12.12→195.79→3147.97→49770.25 RZへ広がる。1 RZは説明用の単位比較で、採択した合否基準ではない。許容G誤差・必要価格精度・Nによる重みが未採択/未接続なので、どのroundまで総資源比較に十分かは未判定。長roundでもTV自体が小さいことだけでは価格差が十分小さいとは結論しない。

## 6. 実sampler・有限標本・ideal-lawの境界

F0のμは保存SOURCEの重み比によるlaw、νは独立一様53-bit格子等の仮定を満たすpath lawである。helper内Qはchoice入力分布、F/GはCDF連続/格子分布であり、Gは総量子資源Gではない。今回のBはこのμ/ν間の**期待残差差の絶対値上界**で、符号付きの補正や正式E[C]の値ではない。

PCG64の固定seedなら軌道は決定的であり、その4 path平均を確率期待値と同一視しない。seed ensembleを対象とするならその測度を明示し、同じword生成SOURCEのjoint lawとνの関係を別に示す必要がある。周辺53-bit格子の正確さだけでは、同一stream内・outer再始動間の独立性を証明できない。actual_PCG64_joint_distanceとactual_PCG64_RZ_shift_upperは全26軸でnull。解析的実数重み→保存値の差もF0/今回では確定しない。ideal_finite_RTE_lawという資源比較対象は維持し、保存μとその全定義の一致を未確認のまま仮定しない。

この接続が解決しても、有限cohortの平均推定誤差、未観測rare labelの寄与の推定、m0/m3/m6から長roundへの残差モデル移送、凍結モデルの誤差は残る。m4の記述的検証は指定したQの4 path平均との比較だけで、これらの上界ではない。保存E_I[S]・m4・凍結モデルを変更せず、CI・正式E[C]・N/G/winnerへ進めていない。

## 7. 次の1作業・検査・STOP

推奨するのは、追加compileより先の**元PCG64 samplerのjoint-law接続調査1件**。SOURCEの乱数消費（choiceのorder/component順、K2、外側seed再始動）を固定seed生成規則と対応付け、対象seed確率測度とIID-gridとの距離/couplingに有効な既存結果があるかを限定調査する。対象seed ensemble自体が未定なら、その研究上の選択が必要であり、便宜的なseed数やIID仮定を追加しない。有用な距離が得られなければnullを維持して停止する。希少event探索や追加path取得はこの案に含めない。ここからの実行は別指示。

今回の検査はSOURCE/metadata/ASTとexact算術に限定した。保存basis metadata、AccountingのXX展開、標準equivalenceのXX展開、26 physical/axis対応、event_count=qR、保存TV locator、exact積、X/Yの別記録、旧証拠の不変を確認する。新H4 gate、full-wrapper statevector、量子作用の全support検証は実施していない。過去のgateの未実施範囲も変更しない。[SOURCE_CHECKS](SOURCE_CHECKS.json)／[整合性検査](CHECKS.json)。

成果は共通SOURCE上界の導出1件、26軸envelope/積、再利用path TV13件（26参照）。今回の追加価格receipt・新path・H4 build/compile・F0再実行・rare探索・fit・CI・正式E[C]/Q3/N/Gは0。数値nullは「失敗0件」への補完ではなく、actual-PCGの全26軸等の未確定を表す。ロードマップ本体は未編集。全support range不足は条件付きの粗いSOURCE上界まで進み、次の主要不足はactual samplerのlaw接続と有限標本/長roundモデルの資格である。

指定D2 branchへcommit・pushしremote成果物を照合後、**mandatory STOP**。元F0、m4独立holdout、凍結正本、保存E_I[S]、旧raw/SCOPE_DEVIATION/STOPは保全する。
