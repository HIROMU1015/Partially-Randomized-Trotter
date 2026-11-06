# Track D Stage C4：全RPE期待回路価格・長qモデルの技術準備

2026-11-06 JST。**保存価格と再利用部品はあるが、whole-path Referenceの全law期待価格・長q価格coverageは未取得である。** 最小の次作業は、C3の強い最適化条件を保ったrequest入口とfull-wrapper価格記録を共通化し、q2の条件差を解消したうえで、短qのpaired校正・独立holdoutを行うことである。旧fenced kernel会計へStage Aのfull-wrapper価格を渡す方法は採用できない。

本書は静的な技術調査と未実施の実行計画である。新しい価格集計・fit、sampling、科学module import、回路生成・compile・postcheck、N・成功率算定、全13-round費用評価、アルゴリズム実装、詳細科学レビューは行っていない。保存数値は既存表・JSON欄の引用に限る。以下の提案は採択・実行許可ではない。公開・remote照合後mandatory STOP。

## 1. 固定条件・資料の取得範囲

linear H4、隣接1.00 Å、STO-3G、Nα=Nβ=2、system8 modes、number-CDF8、元frame順LD2、ordered partial S2/s1/r32/K0、各h=δ=.1。元Hamiltonian Hの厳密基底状態を仮定し、状態準備費を除外する。保存own-DF groundの信号・Nを元Hへ転用しない。m=0..12／q=1..4096は保存評価の再利用候補であり、新しいenergy精度・成功確率・採用round数ではない。

共通・repositoryのAGENTS.md、PROJECT_MAP、研究概要・現状、研究目的、関連validation status／manifest、正式共通指示2026-10-25版を実際に確認した。正式共通指示は適用原本を読み、GitHubに収録されていない文書への架空の公開リンクは作らない。C4専用branch `codex/track-d-c4-price-model-preparation-20261106` はC3完了commitから分離した。B4-1のbranch・worktree・SOURCE・環境・process・出力・STOPには介入していない。

| 固定資料 | 今回取得・静的参照した内容 | 証拠の意味 |
|---|---|---|
| [Stage A][A] `3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0` | RESULT、SOURCE metadata、finite_cohort_summary、STOP、adapter/runner/checker | q1保存32 path。科学SOURCE `2dd012dc77dcc877d5918c7d212819eb79b97557` |
| [C2][C2] `c8fe5091c42ec786688b6931105ad64d10c87d18` | RESULT、価格比較、SOURCE metadata、STOP、compile/postcheck入口 | 保存00→01のq2一fixture。科学SOURCE `e0befa619a14b956ad64349a86ec2f261962883c` |
| [B1/C1][B1] `d3913866e6ade4ab97ac02645fed91b2d5fcfa75` | RESULT、B1 resultのstate/law/N資格、SOURCE metadata、STOP | 元H operator十分条件は条件付き、N未算定。科学SOURCE `8c6eeb02e89f6cf14870789719493e75482f62f8` |
| [C3完了][C3] `cc554cf6daea08df6c20ea8e1e92405259ce49b5` | 最終COMPLETION_RESULT、元PLAN、run02 PLAN、中断記録、補完PLAN、SOURCE metadata、resource_comparison、旧/最終STOP、archive索引、公開範囲 | q4/q8の固定fixtureと全8 postcheck PASS。今回checkoutの基点 |
| [B3統計接続準備][B3] `89a13fe6841c1017f0feba724ed71dfc79f9e81b` | 固定Git objectからREPORTの元H接続・N共通化条件・費用会計・STOPを取得 | 静的準備であり、B4-1の進行・結果の確認ではない |
| [既存CDF8契約][CONTRACT]・[law metadata][LAW] | 固定policy/sampling/statistics/qualifications/STOP、lawの保存欄 | 旧PR対ENTRY、own-DF target、旧RZ主指標・Welch95%等は今回のReference比較へ自動採用しない |

元h/g/scalar packet SHA256は `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、CDF J/U endpointは `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。cohortは `near_practical_PR_r32_fixed32_v1`。SOURCE metadata上、元science `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7` の143 SOURCE・16入力・追加law、carrier `3649b75af191916daf44abb91e14d34c43b83e8f` の96保存bindingに接続する。今回、全closureの再監査、raw NPZ展開、QPY decode・再検証はしていない。

既存CDF8 downstreamのscience SOURCE `4df8eca29c59ffd55e2dd898f05da16bb246ddbe`、上記ba0 SOURCE、C3基点の間で、`c1_stage3_connection.py`、`c_pr_aware_df_m3.py`、`c1_stage3_accuracy.py`、`rpe_resource_accounting.py` の4 fileのGit blobがそれぞれ一致することを確認した。これは選んだ再利用部品の版照合であり、全SOURCEの同一性や新しい科学的資格ではない。

### C3の成功・中断・補完の区別

旧SOURCE `f18adc46dffcb69a6b1028d5c38793f9b4c0b948` のrun01はcontract参照先誤りでraw構築前STOP。修正SOURCE `52b3cab50d5c5edcbd7e3249637168e9978be535` のrun02は両raw PASS・8 compile・7 postcheck後に中断し、終了原因未確定、最終result/STOP未生成である。補完SOURCE `88f0496349c056d0736f08487889f942c1aa4d37` は保存回路の最後のq8 Reference Y postcheckと最終paired/χ照合だけを別出力へ実施し、新compile0で完了した。成功資格はcc554の最終記録にあり、旧runを成功へ書き換えない。[中断記録][INTERRUPTED]／[補完PLAN][COMPPLAN]／[最終STOP][C3STOP]

## 2. 保存full-wrapper価格と比較条件

全表は状態準備なしのH→core→YのみSdg→H→ancilla測定を含む。P0はQiskit1.3.0、optimization_level1、seed17、rz/sx/x/cx、approximation_degree1、qubits_initially_zero=false、topology-free、位相保持。既存native compiler・declared-phase蓄積を再利用している。MPFR160は位相蓄積規約であり、全数値検証の精度やRZのT合成精度ではない。

以下は[Stage A][A]・[C2][C2]・[C3][C3]の既存値の転記で、新しい平均・差・fitを計算していない。q1は各軸32 pathの有限cohort単純平均、q2/q4/q8は各軸1固定wrapperの実compile価格であり、同種の母平均系列ではない。

| q・方式 | RZ count | CX count | RZ depth | CX depth | Total depth | qubits | size |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 symmetric：保存32平均 | 5531.09375 | 2089 | 1322.75 | 1008.71875 | 3071.71875 | 9 | 11247.65625 |
| 1 Reference：保存32平均 | 5814.15625 | 2042.9375 | 739.15625 | 493.5625 | 1649.53125 | 17 | 11623.40625 |
| 2 symmetric：保存00→01 | 10773 | 4100 | 2624 | 1996 | 6096 | 9 | 21970 |
| 2 Reference：保存00→01 | 11114 | 3860 | 1357 | 860 | 3014 | 17 | 22263 |
| 4 symmetric：保存00..03 | 19867 | 7686 | 4746 | 3744 | 11116 | 9 | 40572 |
| 4 Reference：保存00..03 | 20092 | 6998 | 2516 | 1570 | 5583 | 17 | 40301 |
| 8 symmetric：保存00..07 | 39610 | 15360 | 9492 | 7515 | 22259 | 9 | 81015 |
| 8 Reference：保存00..07 | 39602 | 13776 | 4531 | 2870 | 10093 | 17 | 79615 |

q1の保存summaryが持つpaired差Reference−symmetricの観測範囲は、RZ +202..+425、CX −72..−4、RZ depth −823..−395、CX depth −639..−421、total depth −1939..−1011、qubits +8、size +210..+659。各方式の個別min/maxはこのsummaryに保存されておらず、今回CSVから再集計していない。全個別価格は[保存CSV][ACSV]を参照する。これらの観測範囲をlaw supportとしない。q2/q4/q8は一fixtureしかなく、path価格の分散・範囲・SE・CIは未取得である。未取得値を0にしない。

| q | cut・融合・標本単位 | 比較に使える範囲 |
|---|---|---|
| 1 | 各pathのcomplete event16後cut。head relative fusion、tail同basis共有。cutが同basis共有を分断するpathは保存14/32 | 32 path上の方式間paired観測。X/Yの7資源は各pathで同じだが64独立pathではない |
| 2 | complete step1後cut。保存強いcore連結、C3の追加手動cross-step融合・全path位相集約は未適用 | 一fixture・同P0内の両方式差。C3と同じ最適化条件の校正点にはそのまま使えない |
| 4 | complete step2後cut。symmetricは3境界、Reference内部は2境界を融合 | ascending固定fixture。law母平均標本でも未見holdoutでもない |
| 8 | complete step4後cut。symmetricは7境界、Reference内部は6境界を融合 | 同上。保存q4をprefixとして含み、q間でも独立価格標本ではない |

比較可能なのは共通P0・資源単位・full-wrapperの包含範囲と、各固定request内のpaired差である。q、path、cut種別、融合、位相処理、有限cohort平均／単一価格が同時に異なるため、q1/2/4/8から同一条件のq依存則をfitしない。q8のRZ差−8もそのfixtureの観測であり、一般のRZ削減やq単独の因果効果としない。

## 3. 強い対照・whole-pathの最適化条件

[StrongBuilder][STRONG]のuncontrolled Gaussian basis、解析的位相補正付きrelative transition、diagonal fsum、同basis run groupingを保持する。対照は初回forward headがuncontrolled、tailがordinary controlled、reverse headがdirectionalである。全制御Gaussianの弱い対照へ戻さない。

[C3 SOURCE][BUILDER]の`head_recipe`／`mixed_head`は、directional half＋uncontrolled halfの同block0境界をordinary controlled fullへ融合する。directional fullではない。元の[repeated builder][REPEATED]のhalf-sweep融合・位相集約規則と対応するが、legacy DFPartialS2Preparationとnative CDFモデルの型を同一視してそのbuilder全体を差し替えない。

Referenceはwhole V=A B、Bが前半、Aが後半で、一つのgadgetを使う。[reference_gadget][ADAPTER]はAのactual inverseをsystem lane、Bをreference laneへ各1回置き、前後2 CSWAP cascadeとpath全体χ補正を加える。AとA†を両方走らせる二重価格にも、短いgadgetをq回並べる価格にも置き換えない。符号、scalar/identity、event順序、Gaussian位相、normalization/attenuationを保持する。

symmetricはq−1境界、ReferenceはA/B内部のq−2境界を融合し、global cutで1境界を失う。q≥2のcomplete半path cutとq1のevent cutを分ける。両方式ともtail同basis共有・head relative fusionを維持し、C3ではhead-tail横断融合を追加していない。別の合法最適化を将来導入するなら両方式へ同等の機会を与え、新SOURCE・correctnessの対象にする。別実装の節約値を加減算して架空の最適回路を作らない。

全domainに対する測定分布保存を主張するには、ordered path、AB=V、vacuum ray、actual A†、χ、scalar/identity、X/Y conventionが同じで、compile後誤差のcoverageが必要である。C3のcomponent abs1e−10、observable Frobenius/signal2e−10、rtol0、phase fitなしの全256入力列照合は固定fixtureのlocal binary64証拠であり、全law実装誤差boundではない。未来の長qで失敗した後に閾値を緩和しない。

## 4. 再利用SOURCEと費用境界

以下のSOURCEの版はC3基点cc554のもの。表の「再利用」は将来の許可後の候補で、今回関数を実行した意味ではない。

| SOURCE・入口 | 再利用できる部分 | 必要な最小差分・適用できない部分 |
|---|---|---|
| [C3 whole-path][BUILDER] `prepare/physical_segment/mixed_head` | 強い境界融合、A/B分配、phase集約、Bundleと固定条件記録 | `FIXTURES={4,8}`、保存ID・cut固定のため一般requestを受けない。固定entryを残し、明示ordered requestと事前cutを受けるopt-in共通入口を検討 |
| [Stage A adapter][PAIRED] `prepare_common/saved_record/path_phase`、[H4 adapter][ADAPTER] | 同CDF8 context、保存event復元、path χ、actual inverse、gadget/shell | 保存path入口をfresh lawの生成器と誤認しない。新sampling時は既存canonical law samplerを別の許可scopeで接続 |
| [StrongBuilder][STRONG]、[repeated builder][REPEATED] | 位相付きrelative Gaussian、half-sweep融合・位相会計の既存規則 | 制御mode混在はC3を再利用。別型のDF12係数や過去runの契約を移送しない |
| [C2 runner][C2RUN]・[H4 runner][H4RUN]・[correctness harness][CHECKS] | P0 `compile_one/verify_one`、CAR/observable/state-action、metrics | 過去の固定runnerを再起動しない。新SOURCE/出力で必要な部品を呼ぶ。sampling/fit/compileの自動連鎖を追加しない |
| [c1_stage3_connection][CONNECTION] `compose_serial_cost/serial_native_sequence/readout_segments` | target/scope検査、fresh occurrenceの定義、head/readout分離 | policyは`native_kernel_repeat_with_fences`＋`separate_native_readout`。countは連結和、depthはscheduled serial sum。n+1幅・測定なしkernel用で、17-qubit full-wrapperの接続先ではない |
| [c_pr_aware_df_m3][M3] `serial_cost/calibration_uncertainty` | complete h-step kernelのmultiplicityはq、shared calibrationを独立roundへ水増ししない規則 | `substeps=1`。r32や価格標本32を追加乗算しない。stage全体の固定32標本・旧target・旧比較を新providerとして流用しない |
| [c1_stage3_accuracy][ACCURACY] `bounded_cost_intervals/aggregate_totals` | support不明→区間null、m×X/Y coverage、target/scope/long-domain guard、6資源会計 | aggregateは各roundの単一`shots`をX/Yへ共用。軸別Nには最小拡張が必要。qubitsと共分散を含まない。scope名や`long_round_coverage`を変えるだけでは資格を得ない |
| [rpe_resource_accounting][RPE] `RPERoundCostRequest/RPERoundCompiledCost` | axis別期待価格/SE、compiler/scope/fingerprint、provider protocol、coverage/union検査の設計 | preparationは既存DF型、全candidate評価はeffective PF eigenstate/unit-radiusモデル。元HのB入力を偽装して通さない。N外部入力の薄い接続を検討 |
| [affine proxy][PROXY] fit/validation、[validated provider][PROVIDER] | calibration/holdout分離、設定fingerprint、失敗model拒否、直接検証したq集合だけを許すguard | native CDF/Reference用dataset入口がない。slope/intercept共分散・round間共分散はモデル型にない。providerのSE=Noneは0ではない。既存係数・旧DF/r/K/compiler条件は移送しない |

### 混合しない費用境界

| 価格の種類 | 含むもの・depth | C4の接続方針 |
|---|---|---|
| C3 whole-path一体compile | 1 gadget/core＋H/readout/測定、全step融合、cut損失、追加register/CSWAP/χ。DAG上のcompiled depth | 今回の比較対象。full-wrapper期待価格として受け取り、旧shellを再加算しない |
| kernel単体 | 既存complete h-stepの制御時間発展。r32 eventsを既に含む | Stage A full-wrapperをkernelと名付け直して入れない |
| 別head/readout shell | 時間発展の外側、別compileによってH/H相殺を防ぐ | 分離政策でだけ加算。一体compileに同じshell価格を足さない |
| fenced serial | compile済kernelをbarrierで区切る指定schedule。scheduled depth和 | 異なる実装政策。C3の価格や強いwhole-path対照の代用にしない |
| 将来の局所/boundaryモデル | 指定した局所primitiveと補正の和＋compiler残差 | モデル候補。whole-wrapperとのbridge検証なしで実測価格と混ぜない |

M3の旧shared calibrationは、軸同N・kernel政策の重みWに対して`SE_total=W*SE_kernel`とする。whole-path Referenceに同じWやkernel係数を与えない。[B3 §5][B3]も同じ境界を明示している。最大round indexの型上の許容量と、価格の科学的valid q domainは別である。

実際のr32経路も、[固定CDF8 SOURCEの入口][R32ENTRY]の`c_near_practical_r32.execute`から`c_update064_r32_downstream.references/calibrate`へ進み、`retuning.assess_lanes`、`m3.serial_cost`、`m3.calibration_uncertainty`を呼ぶことを静的に確認した。入口はC3 checkoutに未収録のため固定Git objectから読んだ。汎用旧stageの`substeps=4`をr32経路へ移さず、元H state/Nの接続をこの旧own-DF経路で済ませない。

## 5. q依存モデルの候補と校正条件

以下はSOURCE構造から検査対象を整理した設計案であり、係数算定・新しい価格予測・モデル採択を行っていない。

| 候補 | 方式ごとの構造・必要情報 | 校正／適用範囲／破綻条件 |
|---|---|---|
| 同law whole-path直接価格標本 | 両方式を同じωでcompileし、6資源と幅を別記。短qの参照平均/paired差を得る | そのq・law・compiler・cut政策だけ。長qで重ければ自動的に全roundへ実行しない |
| 方式・axis・資源別affine候補 | μ(q)=a q+b。symmetricのq本体・q−1境界、Referenceのq本体・q−2内部境界・固定gadget overheadが動機 | 同じ最適化条件の複数qで校正し、未使用qの独立law標本でholdout。countもcompiled後のaffine性は未証明、depthは特に非線形になり得る。保存4点からfitしない |
| native構造＋境界補正＋compiler残差 | head/tail、同basis run/relative transition、合法なstep境界、global cut、χ/readoutを明示featureへ分ける。既存局所/境界検証の方法を再利用 | native countの加法性と一体P0の加法性を区別。長run、符号/角度相殺、跨境界compiler最適化、非局所routingで破綻し得る。境界2/3等の旧係数は使わない |
| Reference lane依存モデル | A†とBのgate数は両lane分、中央depthは別registerの並列機会とlaneごとの深さ・偏りを追う。CSWAP/χ/shellはwhole gadgetに1組 | max型のlane bottleneckとfull-wrapper残差を検証する候補。E[max(D_A,D_B)]をmax(E[D_A],E[D_B])で置換しない。joint分布・covarianceとcutを保持し、単純なq/2倍則を採用しない |

native gadgetのA†とBは別registerに配置されるが、CSWAP cascadeは同じancillaを共有する。16 CSWAPを無償の同時操作としない。global cutは演算数の分配だけでなく融合を失わせ、actual inverseは順序・位相を含む。scalar/identityは各step一回分を集約し、χのmoduloやcompilerのgate消去も固定overheadを厳密定数と断定できない理由になる。

q1のevent cutはq≥2のcomplete step cutと別の条件として扱う。最小のaffine候補なら校正q2/q4を同じC3政策へそろえ、q1は別cutの診断値として保持する。q8の保存一fixtureは既に観測済みなので、未見の統計holdoutに数えない。

既存validated providerは**holdoutとして直接検証されたq集合だけ**を受理する。q8 PASSをq≤8全域、q16 PASSをq≤4096へ拡大しない。長qを資格付けする選択肢は、追加で必要なdomainのholdoutを得るか、native構造・compiler bridge・残差包絡が全対象domainへ適用できる証拠を得ることである。後者も新しい実装/検証scopeとする。少数holdoutだけで未測定域の一様誤差を証明したとは扱わない。

最大qの直接compileを自動必須にはしない。構造的な包絡や適用domainが得られなければ、そのroundの価格を未資格として全費用を出さない。fenceを追加して価格の加法性を得る案は別政策であり、strong対照を弱めてC3比較を成立させる代用案ではない。採用する場合も別の明示判断・信号対応・同scope比較が必要である。

## 6. 全sampling law期待価格と不確かさの設計

求める量は、方式a、資源g、axis bについて

\[
\mu_{a,g,m,b}=\mathbb E_{\omega\sim p_m}
[C^{\mathrm{full,no\text{-}prep}}_{a,g,m,b}(\omega)]
\]

である。同じr32/K0/hとfresh IID occurrenceを採る場合、ωはq_m本のordered h-step列であり、p_mはそのwhole-path lawである。idealモデルでは各h-step内の32 eventも既存canonical lawに従う。eventは局所構成要素、occurrenceはcomplete h-step、**価格標本の単位は1 full-wrapperを決めるwhole trajectory**である。q個のoccurrence、32 event、X/Y、classical標本数を独立なwhole-path標本数や量子shot乗数にしない。

保存law metadataはλ_R=`.5000442435909978`、216 atoms、ideal finite-RTE IIDを記録し、implemented finite-bit law誤差はnullである。signed atom/scalar/identityとnormalizationは保存定義を保つ。旧PCG64 cohortは固定selectionの保存32本で、今回新しいideal母平均標本として再認定しない。実装samplerの価格標本とideal law期待値を結ぶにはlaw bridgeが別に必要である。law差の価格影響を評価するなら、正当化した全law support等と結ぶ条件が要る。

### 標本の選び方と必要数の決め方（提案のみ）

| 用途 | 対象law・selection・単位 | 数の提案と限界 |
|---|---|---|
| 保存資産の再利用 | 32 event列と固定00..連結fixture。保存順・cutは維持 | correctness回帰・構造/価格範囲の診断に再利用。巡回連結、bootstrap、同pathのq乗で全law標本を補わない |
| 短q校正pilot | 同じendpoint/K0/r32のwhole-path law、q2とq4。結果非依存のseed domainとsample IDを実行前固定 | 初期案は各q **16 independent whole paths**。これは分散・実装負荷を知るpilot案であり、必要十分なnでも母平均保証でもない。実行許可時に数と停止規則を採択する |
| 独立短q holdout | q8の同law、校正とdisjointなseed domain。モデル/閾値固定後にだけ開封 | 初期案16 whole paths。既存q8一fixtureを代用しない。正式なholdout必要数はpilotと採択した価格誤差/信頼度で設計し、数不足なら未資格とする |
| 確認的な母平均/paired差 | 各q・方式・axis・資源のscopeを固定。1 pathを生成し両方式へ渡す | 必要nは現時点で未決。母分散・価格support・誤差許容量が未資格のため数値n_reqを算定しない。pilotと確認用標本を区別する |

このpilot案を一括実行する承認はない。科学callの予算は必要なstageごとに定める。各pathで両方式×X/Yを比較するなら最大4 wrapperであり、X/Y価格省略は全対象域の同価格を保証する別の証拠が得られた場合だけ検討する。既存32本で同値だったことだけでは省略を正当化しない。

`trajectory_seed`はarm/axis等を引数に取る。paired価格では方式ごとに別seedを引かず、共通の科学PR-roleとpath IDで一度生成した同じordered requestを両方式へ渡す。校正/holdout/量子shotは別domainにし、価格の良否を見てpathやcutを選び直さない。cross-qでprefix共有を使う追加設計を採るなら、その依存をcovarianceへ記録する。独立性の自動仮定はしない。

必要数の設計では、まず価格誤差の絶対/相対幅とclassical信頼度・多重性をユーザー／GPTが採択する。全law support [L,U]が正当化できる場合に限り、既存`bounded_cost_intervals`の条件付きHoeffding幅からnを決める方法を再利用できる。support不明なら点推定/SE/仮定付き区間と資格を分ける。pilotの名目的SE・t区間や観測min/maxを厳密保証に格上げしない。任意停止で点wise区間のcoverageを保持したとも主張せず、固定した確認標本または別途採択した逐次推論法を用いる。

### shared calibrationと長qの誤差

1. 同じpathの両方式差を保存し、方式間covarianceを保つ。X/Y、q、6資源間のjoint情報を必要に応じて保持する。
2. model parameter θを共有するなら、parameter covarianceと全roundへの依存を保存する。既存affine型にはこの共分散がなく、providerのSE=Noneを0として埋めない。
3. classical path sampling誤差、calibration parameter誤差、compiler/model residual、長q移送bias、ideal/implemented law差を別項として記録する。独立と確認していない項を二乗和で合成しない。
4. holdoutはfitやmodel/cut選択へ使わない。失敗後に同じholdoutへ再fit・閾値緩和してPASSにせず、新たな判断へ戻す。独立holdout平均のSEだけを受理幅へ足す既存proxy判定は、校正不確かさや全law・長q一様保証の完成形ではない。

将来Bが固定したNに対して、期待価格ベクトルのcovarianceをΣ、重みをw=Nとすれば、総費用推定の分散は条件付きにwᵀΣwで扱う設計が必要である。paired方式差にもjoint covarianceを使う。同じ校正をroundごとに独立として誤差を縮めない。長q model biasはこの分散だけでは覆えず、別の残差包絡・適用域が必要である。今回、Σ・fit係数・総費用・信頼区間は計算していない。

## 7. 次の最小実装・科学検証案と優先順位

以下は未実施。実装・科学実行・保存値の新解析それぞれを次の明示scopeへ含める。新ground solve・DF再fitは初期案に不要であり、新しい精度目標・成功確率は本書で固定しない。

| 優先・担当 | 解消する不足／最小作業 | 再利用・成果・実行境界 |
|---|---|---|
| P0 ユーザー／GPT | 比較するfull-wrapper政策、ideal/implemented claim、価格誤差幅/信頼度、候補round domain、許可する科学callを選ぶ | 主資源指標を一つへ勝手に固定しない。B4-1の終了を待たずCのscope判断は可能 |
| P1 Codex：実装許可後 | C3の固定entryを維持したまま、明示ordered request/cutを受けるwhole-path共通部と、方式/axis/資源別価格記録の薄いadapterを追加 | `prepare_common`、`mixed_head`、`physical_segment`、gadget、P0/checkerを再利用。旧固定契約・SOURCE・結果を変更しない新revision。新しいcompiler/sampler/state simulatorを作らない |
| P1a Codex：同実装scope | 境界の両control branch、非零scalar、actual inverse、phase、複製保全、未知law/support/domain、scope混在、未検証qの拒否を人工入力で検査 | 既存C3人工test、repeatedのtrajectory/scalar検査、proxyのholdout/未検証q拒否、M3のsupport/shared calibration、resource guardを選択再利用。今回test実行0 |
| P2 Codex：限定科学許可後 | q2保存00→01をC3の強いcross-step政策へそろえる。raw correctness後に必要なP0 wrapperを比較 | 初期上限4 wrapper。Reference raw/compiled bytesを完全再利用できる場合だけ重複compileを省く。C2結果は書き換えない。q1/q4/q8の既存compileを理由なく再実行しない |
| P3 Codex：sampling/compile/解析許可後 | §6の短q pilotでpath依存性とcalibration分散を取得し、同じwhole-path政策で候補モデルを校正 | q2/q4を選ぶ理由は最小の2校正点と境界条件差の解消。q1は別cut診断、q8保存fixtureは回帰証拠。law/precision/fairness一致と必要なcorrectness検査を先に固定 |
| P4 Codex：独立holdout許可後 | 凍結したモデルをfresh q8 holdoutで検査。各資源・paired差の点誤差、校正/標本不確かさと残差を照合 | 許容abs/relative幅とclassical失敗配分は実行前採択。単にCIが0を跨がないことを価格モデルの精度PASSとしない。失敗ならdependentな長q価格をSTOP |
| P5 ユーザー／GPT→Codex：必要時の別許可 | 要求domainがq8を越える場合に限り、境界/critical-path構造とcompiler残差の適用域を検討。最初の追加q候補はq16 | q16は最初の倍長でcompiler/蓄積/分配の破綻を調べる理由があるが、今回未実行で自動必須でもない。q16だけから4096を資格付けしない。構造的coverageと限定holdoutで足りるか、未資格域を残すか判断 |
| P6 Codex：統合計算の別許可後 | 採択された全m×X/Yについてqualified Nと同scope期待価格を結合 | 欠落round/未qualified law/モデル域外/未知誤差なら総費用null。最大qの直接compileや全13-round和は本書から自動開始しない |

新規実装が必要な理由は、既存C3入口の固定q/保存ID限定、旧kernel政策との費用境界差、native CDF identityと既存provider型の差、共分散/幅の不足である。同じ回路処理を別実装として重複させる理由ではない。

価格モデルのholdout PASSはcorrectnessの代用ではない。将来のpath/long-q正確性の必要範囲はAB/adjoint/vacuum/χ、full complex coherence、X/Y、raw/compiled一致とreference leakageを含めて定める。既存固定fixtureの数値誤差を全path実装boundとして使わない。構造対応や一様誤差coverageを得られない場合は、ideal条件付きモデルと実装の未資格を区別してBへ返す。

将来のheavyな処理は、独立path・方式/axisのcompileや独立検証を分割し、共有入力/context・Gaussian/相対遷移cache・既存phase compiler/state-actionを再利用する。実空きCPU/RAMとtaskごとの必要メモリを観測して並列度を決め、恣意的な低いworker/thread capは置かない。共有venv・OS・scheduler・他jobを変更しない。SOURCEを固定し、完了済taskを保全できる出力と接続寿命に依存しない起動を計画するが、旧runのretry条件を変更しない。

## 8. Stage Bから受け取る入力と最小interface

B4-1の新しいSOURCE・実装・結果は今回取得・評価していない。以下は[B3][B3]と保存B1から作る将来の受渡し仕様であり、現在Nや成功率が確定している意味ではない。

| B→Cの入力 | 必要な資格・単位 |
|---|---|
| 元H/CDF packet、sector/encoding/bit順、frame/LD2、SOURCE、scalar/identity | 同じ科学identity。own-DF state名やlegacy DF preparation hashへ偽装しない |
| 採択したm/q/t/h/s/r/K・round set | 13-roundは候補。時間とqはcomplete h-step単位、rは1 h-step内のmicro-events |
| N_m,X／N_m,Yとその資格 | sufficient Nか経験評価か、exact-H仮定、ideal/implemented law、未知誤差、alias/decoder条件、energy/phase/alpha配分、全round成功条件のcoverage |
| Referenceと対照で同じNを使える条件 | 全対象lawの測定信号対応、finite-precision追加誤差を共通予算で覆うこと。固定fixture PASSだけでは共通N全域を認定しない |
| 誤差と成功確率 | quantum failureとclassical価格区間の信頼度を分ける。状態準備除外、reference初期化/維持等のscopeを明示 |

C→Bは、`method / m / q / axis / endpoint+law+ordered-request policy / cut+optimization policy / P0+precision+routing / full-wrapper scope / depth_semantics / estimate_kind / valid_q_domain / mean[6資源] / SE・covariance・校正ID / interval資格 / width・そのcoverage / SOURCE・sample selection` を渡す最小recordを候補とする。`estimate_kind`はfinite_fixture、finite_cohort、law_mean_estimate、validated_model等を区別する。未知量はnull・未資格のまま保持する。

既存`RPERoundCompiledCost`のaxis期待価格・compiler/scope/fingerprintの設計と、native `aggregate_totals`のcoverage/identity検査を再利用する。全体の候補評価をそのまま呼んで元H成功資格を得たことにはしない。新しいnative whole-path providerは明示的な対象型を受け、既存state/target/domain guardを緩めず、外部から資格付けされたNと結ぶ薄いopt-in接続を検討する。軸別N・共分散・widthは必要な差分だけ追加する。

将来加算する6資源gは、同scopeでのみ

\[
G_{a,g}=\sum_{m\in\mathcal M}\sum_{b\in\{X,Y\}}
N_{a,m,b}\,\mu_{a,g,m,b}
\]

と会計する。これは記号上の接続設計で、今回数値和を計算していない。RZ/CX count、各depth、total depth、sizeを別々に保持し、RZをTへ変換しない。depthの和は採択した1-shot depthに対する逐次shot負荷指標であり、実時間やT-depthではない。qubitsは別のpeak width制約で、q/Nを掛けず、同時shot実行を選ぶ場合は別の同時幅を定義する。観測9/17をrouting/workspace込み全domainの幅保証に格上げしない。

追加register、CSWAP、actual inverse、χ、cut損失はfull-wrapper価格に含む。routing、任意のfault-tolerant合成精度、reference register維持/reset等は保存P0価格で未評価であり、未測定費用を0と認定しない。全RPE総費用の優位性は、同じenergy精度・成功確率と価格coverageを両方満たして初めて別の評価対象になる。

## 9. STOP条件・今回の終了

| 条件 | 止める作業・次担当 |
|---|---|
| SOURCE/input/endpoint/law、符号/順序、scalar/identity、cut、phase/adjoint不一致 | 依存する科学実行・価格会計をSTOPし保全。Codexが技術原因を記録、許可範囲を越える修正/再実行はユーザーへ戻す |
| correctness不一致、長q数値精度不足、fairなstrong対照を維持できない | 弱い対照・閾値緩和・結果依存cut変更へ進まない。ユーザー／GPTが次の科学範囲を判断 |
| model holdout失敗、calibration/holdout混入、期待law/SE/support/domain未資格 | 該当model・依存する長q/母平均/総費用claimをSTOP。有限fixture観測や点推定の報告と分ける |
| N、alias/全round成功条件、m×axis価格coverage、scope/depth/幅条件が未接続 | 全RPE総費用は未評価/null。同じNの無条件流用や未取得費用0への補完をしない |
| 新sampling/fit/compile/保存値再解析、B2/B4-2、B4-1介入、詳細レビューが必要 | 今回の許可外。実行せず、対象・SOURCE・入力・call範囲・担当を提案して明示許可へ戻す |
| 実OOM/自process異常/入力破損等の継続不能 | 自分のjobだけで保全・必要な停止。一時的競合や無関係worktreeの通常更新を理由に他jobへ介入しない |
| 今回の静的調査・文書公開・remote照合完了 | mandatory STOP。追加q、新sampling、長q compile、価格fit、全RPE評価、詳細科学レビューを開始しない |

今回変更するのは本書とPROJECT_MAP/docs索引の必要最小限のMarkdownだけである。科学結果・研究方針の採択は変えていないため、科学コード・入力・raw/result・SOURCE・契約・grant/ledger/marker・既存STOP、科学manifest/VALIDATION_STATUS/過去の研究概要は変更しない。内部CPU/RAM観測・内部環境path・非公開raw・資格情報を公開しない。

**次担当：ユーザー／GPT**がP0の科学条件とP1/P2以降の最小許可範囲を採択する。**Codex**はその明示許可後に、request入口/価格recordの最小実装と、承認されたcorrectness・compile・classical価格標本/解析だけを担当する。B担当は別途許可された元H N・成功条件の資格を返し、CはB4-1へ介入しない。承認証明・新grant/承認ledgerは追加しない。

専用branchへcommit/pushし、remote SHAと全公開Markdownの内容・bytesを照合して終了する。公開照合の実測SHAは最終チャットで報告する。**公開後mandatory STOP。**

[A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0/docs/research/track_d_h4_paired32_reference_20261106/RESULT.md
[C2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8fe5091c42ec786688b6931105ad64d10c87d18/docs/research/track_d_c2_q2_compile_20261106/RESULT.md
[C3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc554cf6daea08df6c20ea8e1e92405259ce49b5/docs/research/track_d_c3_q4_q8_20261106/COMPLETION_RESULT.md
[B1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3913866e6ade4ab97ac02645fed91b2d5fcfa75/docs/research/track_d_b1_c1_validation_20261106/RESULT.md
[B3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/89a13fe6841c1017f0feba724ed71dfc79f9e81b/docs/research/track_d_b3_rpe_statistics_preparation_20261106/REPORT.md
[CONTRACT]: ../research_c_near_practical_r32_downstream_contract_20261103.json
[LAW]: ../../../artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/A/law.json
[ACSV]: ../../../artifacts/track_d_h4_paired32_reference/2026-11-06/run01/paired_resources.csv
[INTERRUPTED]: ../track_d_c3_q4_q8_20261106/RUN02_RESULT.md
[COMPPLAN]: ../track_d_c3_q4_q8_20261106/COMPLETION_PLAN.md
[C3STOP]: ../../../artifacts/track_d_c3_q4_q8/2026-11-06/run03_completion/STOP.json
[BUILDER]: ../../../src/trottertracks/track_d_c3_whole_path.py
[PAIRED]: ../../../src/trottertracks/track_d_h4_paired_reference.py
[ADAPTER]: ../../../src/trottertracks/track_d_h4_reference.py
[STRONG]: ../../../src/trottertracks/c1_h4_strong_baseline.py
[REPEATED]: ../../../src/trotterlib/df_partial_s2_repeated.py
[C2RUN]: ../../../scripts/run_track_d_c2_q2.py
[H4RUN]: ../../../scripts/run_track_d_h4_reference.py
[CHECKS]: ../../../tests/track_d_h4_reference_checks.py
[CONNECTION]: ../../../src/trottertracks/c1_stage3_connection.py
[M3]: ../../../src/trottertracks/c_pr_aware_df_m3.py
[ACCURACY]: ../../../src/trottertracks/c1_stage3_accuracy.py
[RPE]: ../../../src/trotterlib/rpe_resource_accounting.py
[PROXY]: ../../../src/trotterlib/rpe_hadamard_compiled_cost_proxy.py
[PROVIDER]: ../../../src/trotterlib/rpe_hadamard_validated_proxy_provider.py
[R32ENTRY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4df8eca29c59ffd55e2dd898f05da16bb246ddbe/src/trottertracks/c_near_practical_r32.py
