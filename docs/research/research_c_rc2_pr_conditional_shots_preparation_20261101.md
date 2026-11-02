# 研究C：RC2／PR-tailの条件付きshots・coherent反復重みの技術準備

2026-11-01 JST。Repository：HIROMU1015/Partially-Randomized-Trotter。
起点：e33ab6e03344c8179ff9b378dc446bf4cfc55e10。

**保存Stage Bの26セルを既存C1 accuracy／energy／Hoeffdingと結ぶ専用entryを準備した。人工・回帰125件PASS、SOURCE101／入力14のpassive preflight PASS。科学実行案は2候補×3 lane×13round、最大105 logical units。実H4保存値を使うshots／W／energyの新規算定は0で、実行直前で停止する。**

[未承認105-call契約](research_c_rc2_pr_conditional_shots_contract_unapproved_20261101.json)／[SOURCE101](../../artifacts/research_c_rc2_pr_conditional_shots/2026-11-01/technical/SOURCE.json)／[人工125 PASS](../../artifacts/research_c_rc2_pr_conditional_shots/2026-11-01/technical/tests.txt)／[passive preflight](../../artifacts/research_c_rc2_pr_conditional_shots/2026-11-01/technical/passive_preflight.json)。実装commit：b4e466af6b3502b7fe777c4c160bda65d0b9a1ea。共通運用指示2026-10-25版§8.1/8.2/8.4/10/10.1/10.3をworkspace原本で確認した。承認証明・承認原文保存・新grantは作成しない。重要GPT科学レビューは行っていない。

## 1. 目的と引継ぎ

問いは「**同じCA／RPE条件で、保存PR-tailの減衰改善が、条件付き十分shotsとcoherent kernel重みWの低下につながるか**」。[Stage B結果](research_c_rc2_pr_stage_b_execution_20261101.md)では、両候補とも13/13roundがPOTENTIALLY_FEASIBLEだった。PRは全roundでAが大きいが、PF誤差も大きく、補正RTE誤差の改善はm6..12に限られた。

同報告からの引用として、m12のAはRC2=8.424701129970533e-6、PR=.8643440178248293。PFは.0584255430/.0670810994、補正RTEは.0521750245/.0300545902である。これらから新しいshotsやWを計算していない。POTENTIALLY_FEASIBLEを測定回数の優位・CA達成と扱わない。

対象は**直近の共通収束政策2-fitから得た固定保存終点**。同じ元qualified x0、初期寄与を合わせたα=γ=.1、共通solver・関数評価上限で、両DF予算PASS／両optimizer一次収束資格未達。十分に最適化されたRC-DF一般、最適強度、大域最適性の比較ではない。旧γ0／旧RC2／旧PR終点、旧M3／M4の数値やnative価格へ置き換えない。

GitHubから起点commitのStage B報告・result.json・accuracy／C1 SOURCEを実際に取得し、local bytesと一致確認した。保存値の引用・schema／identity確認に限定し、新しい科学的な再集計や順位付けは行っていない。

## 2. 固定SOURCE・入力・科学条件

| 固定物 | Identity |
|---|---|
| 保存結果公開commit／今回起点 | e33ab6e03344c8179ff9b378dc446bf4cfc55e10 |
| Stage B科学SOURCE | d6b17b79461a39bd8ba2e6b7389a1f4fad4d74e6 |
| 共通2-fit科学SOURCE | e2f7dfd44357510d54ab9c6c9c5032e0072ab376 |
| JSON復旧公開commit | f6bb336b41cddface5647c6b1acf23c330f32637 |
| RC2 final J/U SHA256 | d6cb5d56dfc6a83090a6ad1bc1207fdaee62a14cf120542fcb5a7d116644a798 |
| PR-tail final J/U SHA256 | 3fdb0cb77afc770294cbe7aebf47d8a4529e3567ef9e886895495f4490b7ee21 |
| 元h/g/constant snapshot SHA256 | 1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db |
| 新SOURCE manifest SHA256 | dc3caba2e3cf797f742f3b1ae4a026f353bd2d0a87d24565b529314ddb81a0a4 |
| 新未承認契約SHA256 | 10701a6a0f2c92ac2df0bd20b693fe0bb613c754b1969d985e381fdb44cd724e |

数値入力は `artifacts/research_c_rc2_pr_stage_b/2026-11-01/run01/result.json` の保存scalar。対応CSV `rounds26.csv` もhash固定するが、数値計算はJSONを一次入力とし、二重に評価しない。保存STOP／SOURCE／契約／両ground/alias記録、候補exportのopaque hash等を合わせた全14入力を契約に固定した。J/Uやh/g配列をdecodeするentryはない。

| 科学条件 | 固定値 |
|---|---|
| 分子／basis | 直鎖H4・隣接1.00 Å・STO-3G |
| Sector／DF | Nα=Nβ=2、sector36、number CDF rank8・一般対称J |
| 分割／PF | 元frame順LD2、partial ordered symmetric S2 |
| 時間／RTE | δ=h=.1、s1／r128／K0、m0..12、t=.1×2^m |
| 精度／位相 | CA=.00159 Ha、DF reserve=.00019875 Ha、βtotal=π/6、βstat=π/12 |
| Failure配分 | Candidate .01、axis .01/26、13round×X/Yで一度ずつ計上 |
| State | Own-DF ground-state oracle、状態準備費用除外 |

Failure上界は**各candidate・各lane**の既存配分。2候補／複数仮定laneを同時に測定したjoint confidenceへ拡張せず、新配分もしない。元g補正κ、scalar、DF誤差・guard、ground/alias、保存PF/RTE／Aは再計算しない。

## 3. 既存実装の再利用と新しい接続

新規ファイルは[adapter／runner本体](../../src/trottertracks/c_rc2_pr_conditional_shots.py)、[CLI](../../scripts/run_research_c_rc2_pr_conditional_shots.py)、[人工テスト](../../tests/management/test_research_c_rc2_pr_conditional_shots.py)。既存library／旧runner／SOURCEは変更していない。

| 既存資産 | 今回の利用 |
|---|---|
| c1_stage3_accuracy.AccuracyTarget／RoundEvidence／assess_round | 同じraw complex-signalの誤差・半径・位相・Hoeffding接続 |
| assess_energy、既存target horizon | 保存DF norm＋guardをHaで一回だけ計上し、13round・failure・energy十分条件を評価 |
| c_pr_aware_df_stage_c.qualification／assess_round／energy | Evidence／ideal／assumedの数値規約と資格。旧B2価格proxyの仮定だけ除外 |
| rpe_resource_accounting._hoeffding_shots | 通常数値域の十分shots。新しい推定器や近似式は作らない |
| c1_stage3_shots_screen._count | 既存の大整数range拡張：binary64定数の有理数積・integer ceil |
| M4 shots_summary／affine_accounting | X/Y・2^mを一度ずつ計上する既存会計を調査。旧M3比較やshell価格に固定された関数は直接起動しない |
| 新Stage B保存schema／identity | RC2／PR、元fit・export hash・26セル／ground／STOPを接続 |
| 既存Calls、save_json | Logical call／partial STOP、NumPy scalar正規化、atomic/exclusive出力 |

新adapterが必要なのは、旧C1／M4 runnerが旧Baseline／Proposed、旧native32校正、旧M3のshots差等に固定されているため。純粋な評価関数を呼び、今回のrole・保存scalar・fresh outputへつなぐ。新しいHamiltonian builder、CAR、PF/RTE reference、RPE estimator、shot法は作らない。Native費用を必要とする旧集計entryは呼ばない。

SOURCE101は、既存Stage B・C1・shots-screenの依存集合と新3ファイルのunionで、現在の起点版のbytesを固定した。C1の旧manifestには現在の起点との既存4差分があり、新manifestのhistorical_to_baseへ記録した。過去版とのbyte同一性を装わず、旧manifestと履歴をそのまま保全する。今回old libraryへの差分は0。

## 4. 3 laneと誤差会計

既存C1の次の式を維持する。

\[
d=A(\epsilon_{PF}+\epsilon_{RTE})+\epsilon_{syn}+\epsilon_{law},\quad
\rho=A-d,\quad \beta_{det}=\arcsin(d/A),\quad
\epsilon_{coord}=\rho\sin(\beta_{stat})/\sqrt2.
\]

βdet＋βstat≤βtotal、正の半径等を既存関数で確認し、既存Hoeffding十分shots `ceil(2 log(2/α_axis)/ε_coord²)` を使う。DF誤差[Ha]をdへ加えず、既存energy評価で一度だけ扱う。

| Lane | 扱い | 得られる資格 |
|---|---|---|
| evidence | Synthesis／implemented-law誤差をNoneのまま保持 | EVIDENCE_UNQUALIFIED。Shots／W／CA資格を作らない |
| ideal_exact_native | Exact synthesis、ideal finite-RTE IIDを明示的に仮定 | 反実条件付き十分shots。実装保証ではない |
| assumed_raw_signal_budget | 各roundでepsSyn≤.001A、epsLaw≤.001Aを仮定 | 旧C1と同じ感度条件。実測上界・全law保証へ昇格しない |

Assumed laneでは当該仮定の上限値を誤差会計へ入れる。Actual synthesis／law誤差は全laneでNone、all_law_bound_qualified=falseのまま。旧C1 scenarioに含まれた「旧B2 sample meanを価格proxyにする」という仮定はshots-onlyでは不要なので転用しない。

## 5. 数値域と正確な整数

通常域はC1.assess_roundのraw-signal geometryをそのまま使用する。Shot countがbinary64の整数解像度未満なら、元Hoeffding helperと同じ整数を返すことを人工検証した。

**2^53以上のshots**は、旧float ceilでは1単位が分解できない。既存shots_screen._countを再利用し、保存binary64のcoordinateとlog定数をFractionへ移し、正確な有理数積とPython整数ceilで算定する。人工例では旧float ceilとの差も検査し、geometryを変えずに丸めだけを補正することを確認した。これは**binary64定数で定義した式の整数算術の正確さ**であり、log／sin／πや保存operator誤差を囲う厳密enclosureではない。

**極小Aで旧helperがunderflow／overflowとなる場合、またはAがsubnormalの場合**は、既存assess_roundをA=1の同じ正規化誤差で呼び、同じgeometryを取得する。Aとの積とshots算定には既存range拡張を使う。原本のphase基準・failure配分は変更しない。正のexact積がbinary64で0へ丸まるdiagnosticは、黙って0にせずnullとし、正規化値と `*_fraction_binary64_constants` の有理数文字列を保存する。

Zero A／負の半径／phase不適格は登録条件の不適格、非finite・型不一致等はTECHNICAL_FAILUREとして分ける。大きなshotsやWそのものを不適格・実行不可能としない。任意のshot上限・clip・float/int64への縮約は設けない。

## 6. 出力schemaと会計

予定出力は78 round assessment、156 X/Y axis行、6 arm/lane集計、3 lane比較。各roundにA、PF/RTE誤差、raw-signal誤差、半径、deterministic phase、固定βstat、coordinate、status／理由、算術methodと資格を記録する。旧関数の不適格recordにないd／rhoのdiagnosticのみ、上記raw-signal式で補い、判定やshot式は変更しない。未知または定義できない値はnull。

- `shots`：既存Python APIとの互換性用の任意精度整数。Per-axisの十分数。
- `shots_decimal`／`axes.X/Y.shots_decimal`：**正確な十進文字列が正本**。CSVにも同じ文字列を出す。
- `N_total_decimal`：Σ_m(N_X＋N_Y)。実際に測定した回数ではない。
- `W_decimal`：Σ_m2^m(N_X＋N_Y)。s1なので2^mがkernel occurrence。**r128を追加で掛けない**。
- `axis_total_decimal`：X/Yの合計を別々に保存し、それぞれ一度だけ計上する。
- Groups：m0..9、m10..12、m12。最後はdiagnostic subsetであり、全groupを足すと重複する。
- `PR_minus_RC2_decimal`：両候補で算定できたlaneだけ、整数の差を正確な符号付き文字列で保存する。

JSON parserが任意精度整数を扱えるPythonではnumeric shotsも正確。JavaScript Number／Excel等での自動変換は精度を失う可能性があるため、consumerはcanonical decimal-string列を文字列として扱う。量子測定／samplingは実行せず、quantum_shots_drawn=0。

Energyは既存assess_energyへ13round planを渡し、保存guard込みDF bound、既存horizon・failureを使う。Counterfactual laneのconditional energy十分条件とactual_CA_achieved=nullを区別する。

## 7. 未知のnative価格

将来の各arm・指標kには既存serial会計 `C_arm,k=W_arm μ_arm,k＋S_arm,k` を使える。今回求めるのはNとWまで。μは当該armのcontrolled kernel価格、Sは当該armのhead／X/Y shellのshots加重費用であり、**両方とも未測定／null**。

J/Uが異なるRC2とPRへ同じ価格を強制せず、旧C1／M4のkernel／shell価格・32標本平均・literal/shared効果を移送しない。価格0の総RZ、未測定shellを省いた総費用、保証付き資源順位を作らない。Native32、whole-circuit compile、cost support資格は契約外。

## 8. 人工・回帰検証とpassive preflight

新39＋既存C1／M4／shots-screen／新Stage Bの86＝**125 PASS、10.09秒**。実分子の保存値をテストに使用していない。

| 検証 | 範囲 |
|---|---|
| Role／identity／SOURCE／coverage | 新RC2/PR、元export hash、13round×2、重複・欠落・rank・alias・target不一致 |
| 既存評価との一致 | 通常域のC1 assess_round、既存assess_energy、DFの一回計上 |
| Laneの資格 | Evidence null、ideal／assumedの区別、旧価格proxyなし |
| 数値域 | 極小／subnormal A、zero A、非finite、phase不適格、2^53超のceil、int64超のcounts |
| 整数会計 | X/Y一度ずつ、NとW、r非乗算、group subset、符号付き大整数差 |
| JSON／CSV | 巨大Python整数・NumPy scalar、canonical decimal文字列、156行とJSONの一致 |
| End-to-end | 105単位の成功／感度lane不適格／技術FAIL、partial STOP、fresh出力、旧入力hash不変 |

Actual saved-input preflightでは、assess_round／assess_energy／count entryをprocess内で呼出し禁止にして成功することも確認した。SOURCE101／入力14／Stage B schema／依存版を照合し、numpy／scipy／qiskit／ffsim／jaxすらimportしていない。実H4のshots／W／energy新規算定0、NPZ decode0、future run01未作成。

環境は[既存evaluation pins](../../artifacts/research_c_rc2_pr_conditional_shots/2026-11-01/technical/evaluation-pins.txt)：Python3.11.1、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0等。既存venvを読み取り専用で使用し、Python symlinkをresolveせず起動した。ffsim fit環境は今回使わない。

今回の将来計算は78件の小さいscalar評価と、登録条件内で数百桁程度の整数算術で、重いmatrix/GPU workloadではない。既存helperの呼出しとI/Oが中心のため、process/threadを増やさず順次に評価し、最初のcorrectness FAILで即座に止める。CPU/RAM/worker/wall/shotの人工上限はない。空き資源・quotaのcapacity probeをpreflightに保存し、共有venv／cache／設定／他jobは変更しない。

## 9. 最小科学実行案

| 内容 | 最大logical units |
|---|---:|
| 保存bundle読込／identity・26セルcoverage／target-horizon | 3 |
| 両armのscalar attachment | 2 |
| 2 arm×3 laneのqualification | 6 |
| 2×3×13 round accuracy | 78 |
| 6 conditional energy評価 | 6 |
| 6 shots／W集計 | 6 |
| 3 lane比較 | 3 |
| 最終result | 1 |
| 合計 | **105** |

既存164／98-call等を機械的に再利用せず、必要な処理から積み上げた。CSV出力は評価結果の転記で追加のaccuracy／shots算定を行わない。Range拡張も当該round単位内の既存geometry／ceil接続であり、追加候補や新科学条件の探索ではない。

Future outputは `artifacts/research_c_rc2_pr_conditional_shots/2026-11-01/run01`、retry／resume0。Old Stage B runner／消費済み枠を再利用しない。

| 科学実行後の分類 | 意味 |
|---|---|
| CONDITIONAL_SHOTS_AVAILABLE | 当該仮定laneで両候補のenergy条件・十分shots・Wを取得できた |
| ONE_OR_BOTH_INFEASIBLE | 当該laneの登録phase／radius／energy十分条件が未達。方法一般の不可能性ではない |
| EVIDENCE_UNQUALIFIED | 誤差資格が欠ける、または必要数値証拠が未解決。未知値を0で補わない |
| TECHNICAL_FAILURE | SOURCE／入力／数値型・処理等の不一致。Partial evidenceとfailed STOPを保存 |

候補laneの不適格では他の登録lane／roundを続け、correctness FAILでは依存する後続callを停止する。追加fit、PF/RTE／A／ground再評価、sampling、native build/compile、価格校正、総資源、重要科学レビューは契約外。正常完了・FAILいずれも公開照合後mandatory STOP。

既定CLIはpassiveのみ。

~~~bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
 /tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python -B \
 scripts/run_research_c_rc2_pr_conditional_shots.py \
 --contract docs/research/research_c_rc2_pr_conditional_shots_contract_unapproved_20261101.json
# 別途科学許可後のみ、--execute --output artifacts/research_c_rc2_pr_conditional_shots/2026-11-01/run01 を追加
~~~

## 10. 科学実行後の問いと残る判断

Q1〜Q5（候補別shots、長round減衰、PF悪化を含めたshots／W差、ideal対assumed）は、今回準備した105-call実行で初めて数値回答を得る。**今回の技術準備だけで値や順位を作らない。**

Q6のevidence laneには全law synthesis／implemented-law資格が欠けたまま残る。Q7／Q9のnative価格取得の情報価値・必要候補／controlled kernelとhead/X-Y shell scopeは、shots/Wの結果と誤差資格を見てユーザー／研究Cが判断する。Q8では両optimizer未収束という制限を保ち、固定候補対・固定政策の条件付き結果に限定する。Q10の研究方向・最終GOをここで採択しない。

現時点で人工範囲の技術blockerはないが、実H4のshots／energy適格性は未評価。厳密な丸めenclosure、全law誤差資格、最適RC-DF対照、新規性、実機CA／総資源の認定も未確立である。

[保全postflight](../../artifacts/research_c_rc2_pr_conditional_shots/2026-11-01/technical/postflight.json)で新SOURCE／入力、旧Stage B raw69／CSV／SOURCE74、旧fit／JSON修復原本とSTOPを照合する。Commit/push後はremote SHA/tree・本文・SOURCEの一致を確認する。

**次の担当：ユーザー／研究Cが、固定SOURCE・入力・105-call契約について科学実行を別途明示許可するか判断する。Codexは許可後だけfresh runを一度実行する。公開照合後mandatory STOPし、実H4 shots／W／energy、native価格、総資源、追加fit、重要レビューへ自動進行しない。**
