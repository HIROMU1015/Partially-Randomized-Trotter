# RPE総コストに基づく比較ベースラインの設計

2026-11-13 JST。**既存4方式API・候補生成・signal・価格会計を再利用できる。新しい探索frameworkは不要。** 今回はFull PF・Truncated PF・Full-H pure RTEの探索自由度、公平性、有限な評価順を整理した。11案menuは初期の参照集合で、最適候補群ではない。実科学計算・保存値の新解析・新候補の数値最適化・正式選定は0。

[探索計画](BASELINE_OPTIMIZATION_PLAN.md)／[公平比較HANDOFF](FAIR_COMPARISON_HANDOFF.md)／[次の有限科学案](NEXT_SCIENCE_PLAN.md)／[SOURCE索引](SOURCE_INDEX.json)／[技術回帰](TESTS.txt)。Cの既存branchを継続し、A/B/Dの作業領域・共通実装へ変更しない。新production codeは追加せず、既存機能で足りる部分を重複開発しなかった。

## 共通の目的と条件

最小化対象は状態準備を除く全RPE費用 `G=Σ_(m=0..M,a=X,Y) N_(m,a) E[C_(m,a)]`。Cはcontrolled時間発展・scalar一回・X/Y readout・measurementを含むfull Hadamard wrapperのcompiled RZ/shot。量子shot数N、古典価格標本数n、normalization Bを区別し、nやBを価格へ再乗算しない。全roundを評価し、単一時間やm12の価格最小化を代用しない。

参照タスクはLinear H4/1.00 Å/STO-3G、保存tol-only DF10/weighted-Hermitian completion、sector(2,2)、保存full DF-Hの基底エネルギー、同H/state/snapshot・fragment順、P=0。既存案のεE=.00159360010199040 Ha、h=.1/M12・元prior/decoderとRMSE budgetを初期shellとして参照する。数値stateを厳密groundへ昇格せず、条件付き観測・population risk・認証globalを分ける。DF前Hamiltonianへtargetを変更しない。

各方式のNは同じ精度/loss/claimに対して評価する。旧有限時間signal精度やHP型N、PR nominal Nを他方式へ流用しない。h/Mや誤差・shot配分まで探索する場合は全方式に同じ許容範囲とalias/decoder条件を与え、同じh/Mを固定した比較と別に表示する。現在のB selectはhを共通条件としており、h横断探索を実装済みと誤称しない。

## 調査した固定SOURCEと再利用

|SOURCE|実際に確認した内容|今回の再利用・制限|
|---|---|---|
|`51500b724a75f9535a2c831d50bcfe09eabb9326`|4方式REPORT/menu/evidence matrix/NEXT_STEPS|11 schedule案、保存coverage、元target、役割分担。集合を正式採択せず探索のseedとする|
|`256caad3414a80de02fb68eca89c13c957ed6354`|h4_four_method_screen/mean、SCREEN_PLAN/HANDOFF、関連test SOURCE|候補・保存locator・参照時刻・局所meanの既存入口。固定20セルrunnerは任意の新候補を扱う汎用探索器ではない|
|`e7dca5f7301c8633b17a59b6a7161331303aef2d`|full_baseline_v2_requests、pr_rpe_rmse_controller、Full PF receipt REPORT/HANDOFF_C|Full PF3公式とFull-H明示q/Rの物理→B接続、73直接価格/136候補の既存判定。receiptを再監査・価格再計数していない|
|`b72ea6b30e48e6a2f5d805903b7bd750b965cf16`|PF power診断REPORT/HANDOFF/RESTART案|PF signalは直接expmで元macro-stepを構成する方針。代表1セルの数値改善を全候補の精度PASSへ拡張しない|
|`cdf6a80671b2d0bd58983fbbbaf7334f58dc8422`|価格REPORT/PRICE_VALIDATION_PLAN|全domain、同時価格区間、N依存誤差、fit/holdout/rare/lawの会計をそのまま利用|
|Track A：h46_m5_plan、h46_long_time_signal、pf_common/M5資料|formula weights、candidate/cid、prefix、FULL/B0/B0I、signed scalar、旧有限時間格子|formula/構築手順・候補情報を再利用。Track A最良endpointを全RPE最良へ転用しない|
|`9b83b96b83f62142c9d87858dc24a94db7ac5d61`|h46_full_h_pure_rte.prepare_full_h/context|one-body固有basisのZ成分を全DF tailへ追加、全identity scalar抽出、head0。DF B3/prefix0とは別方式|
|`7a8bc7f132e166390c943e6b32c92cc6abe1baf4`|h46_full_h_t20_opt.evaluate/layout/validation SOURCE|理想meanのs=qR共有とpriced_equivalence_claim=false、K0/2/4/6の既存探索。旧proxy_Gはcompiled価格/下界でもRPE全Gでもない|
|Cの既存law/feature成果|d72dc4e6、37cbc331のREPORT/HANDOFF|ideal/CDF/PCG64の区別、構造期待値の条件。PRの324-component kernel/係数をFull-Hへ移さない|

Track Aの`h46_m5_plan.SYSTEMS['H4']`には旧rank12が残る。今回のDF10でpilot/screenのsystem定数をそのまま使用しない。純粋なweights/signed順と、実snapshot由来rankを分離する。M5係数の原SOURCEは同梱referenceの20d64c2へ戻り、名称だけのM5や標準Yoshida4と混同しない。

## 方式ごとの最適化変数

|方式|初手で探索する自由度|後続の拡張と注意|
|---|---|---|
|Full PF|S2、標準Yoshida4（3 signed piece）、M5 best（11 piece）、外側macro-step幅Δ又は整数q_m|fragment順/Hは固定。B現APIはglobal formula/Δからq_m=ceil(T_m/Δ)。round別q/formulaの最適化は別の明示schedule拡張で、現在自動対応とはしない|
|Truncated PF|保持prefix p、B0/B0I、同3公式、Δ/q_m|one-bodyを保持。B0Iは切り捨てtailのidentity scalarを元規則で一回追加。round別prefixや別並べ替えは別科学scope。p7固定だけで方式の限界を決めない|
|Full-H pure RTE|非負偶数K、総event s_mと明示q_m/R_m grouping、必要ならSOURCE付きκ|旧研究にK0/2/4/6があり、K2/K4や旧R係数448に固定しない。pure RTEにPF次数や決定論headを勝手に追加しない。現在B specはglobal K、K_m拡張は担当Bの版管理が必要|

各候補の全round raw signal、決定論bias、半径・attenuation、decoder/riskを通じてNが変わる。細かいPF・大R・大KはCを増やす一方でbias/normalization/Nを改善し得る。公式次数、gate/event数、単一roundの位相誤差だけで優劣を決めない。joint risk制約があるため、各roundを独立に最小化してscheduleを合成する方法も条件付きである。

## Full PFの評価と価格

signalは元signed順・control/scalar規則を保つ直接expmのblock因子→macro-step→既存matrix_powerで評価する。共通因子cacheを使い、保存mean/referenceは同じphysical identityと数値資格がある場合に再利用する。数値gate失敗をphysical FAILにせず、閾値緩和・投影・再正規化で救済しない。追加expm検証を独立研究として増やさない。

価格の候補は `C_a(q; fixed macro-step, profile)=a_a+b_a q`。固定Δ・公式・prefix・native/control/lowering・wrapper/axis・compilerという条件が必要で、a/bの既存数値移植はしない。短roundのq=1や端数ceilにより実幅T/qが変われば、同じtemplateとは限らない。角度のliteral zero/Clifford、scalar、compilerの跨step最適化でaffineが破れる可能性も保持する。

保存直接価格で校正・検証可能なら重複compileせず、未知部分だけ必要に応じて同domainの独立qを取得する。2係数ならrank2の校正は識別条件に過ぎず、使用最大qと境界regimeを独立にchallengeする。最大q一つだけが全domainを保証するわけではない。直接価格・MODEL・未検証外挿・欠測を分け、未資格modelを厳密pruningに使わない。

## Truncated PFのtargetとscalar

Hを保存full DF-H、H_pを共通constant・one-body＋保持fragment、c_pを元SOURCEで抽出されたdiscarded identity scalarとする。B0はH_p、B0IはH_p+c_p Iの近似発展だが、**推定targetはどちらもE0(H)**。B0Iで切断誤差が全て消えるわけではない。PF次数/qを増やしても残る切断biasがあり、Nを増やせば解消すると仮定しない。

誤差台帳ではcutoff/PF/RPE統計・分解能・数値/stateを分ける。一方、直接同full-H referenceへのraw mean/riskを評価したなら、含まれているcutoffやPF biasを再加算しない。signedなcutoff/PF誤差が相殺する場合と、norm上界を足す保守的条件を区別する。`||H−(H_p+c_p I)||`によるWeyl/作用上界は十分条件に使えるが、その上界が予算を超えただけでは候補の不成立を証明しない。

PRと同prefix比較では同じone-body・retained block・tail scalar/source順を照合し、PRがdiscarded非identity tailを近似する効果を分離できる。ただしこれは説明用の対応比較。方式ごとの最小G比較ではtruncatedとPRを同じprefix/公式/qへ拘束しない。

## Full-H RTEのsignal・layout・下界

理想厳密算術では、同じpositive event時間t=T/s、finite Kとfull-H preparationならmean polynomialとBはs=qRで共有できる。元実装は `(T/q)/R` のbinary64演算、pieceごとのseed再生成、boundary/scalar/native構成を持つため、共有にはlaw/数値契約の照合が必要。価格はq/Rの別identityで扱い、sだけでcacheや価格を共有しない。

normalization、finite-K bias、raw半径を同時評価し、同じRPE riskを満たすNとの積でs/Kを探索する。Bは価格の倍率ではない。normalizerの丸め/underflowはUNKNOWNとして保全し、巨大attenuationを便宜的閾値だけで方法不成立にしない。DF B3のtail λやPR固定κをfull-Hへ流用しない。

現SOURCEから、event数・S・局所gate数に比例する**compiled RZの非自明な厳密下界は確認できなかった**。跨event相殺・gate synthesis・角度特殊値を含む証明が必要。常に安全なC≥0だけでは候補を除外できない。有効なoracle/query下界やN下界も、同target/prior/risk/推定器クラス・lawに対する条件とSOURCEを必要とする。未検証proxyだけでFull-Hを比較対象から落とさない。

## 技術確認と結論

既存M5/signed/scalar/allocationの人工12件と、共通pipelineの選択した人工8件がPASS。初回はsparse checkoutで元M5 referenceが未展開のため1件ERRORになり、HEADの同一tracked bytesを展開して無変更testを再実行した。記録はTESTS.txtに残す。新8次PF、実H4、signal/shot/RMSE/価格/Gは実行していない。今回の20回帰は共通部品の確認で、Bの全4方式APIをこのC checkoutで新たに検証したとの主張ではない。固定B/Dの公開テスト証拠はそのまま参照する。

新候補の正式ID/mean/N/C/G/winnerは生成・採択しない。既存SOURCE/raw/cohort/grant/ledger/marker/KEEP/STOPを保全。必要な設計は得られたため調査を拡張せず、commit/push・remote SHA/tree/主要成果物照合後**mandatory STOP**。

次：**ユーザー**が共通claim・有限科学scopeを承認し、**A/PR担当・統計担当**が許可されたsignal/risk/N、**D/価格担当**が同一loweringの価格・domain、**B**が全scheduleの接続・会計、**C**が必要なbaseline固有metadata/設計差分を担当する。具体的な初手と上限はNEXT_SCIENCE_PLANへ記す。
