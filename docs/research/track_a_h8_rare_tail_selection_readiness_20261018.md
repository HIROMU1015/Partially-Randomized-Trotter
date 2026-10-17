# Track A H8：希少Taylor層・選択後効果の技術調査と未承認実行準備

2026-10-18 JST。実施は固定公開証拠の読取・metadata照合、数式設計、実装準備、独立した人工fixtureの検証だけ。**実H8 sampling・回路build/compile・bootstrap・確率再評価・感度解析・shot/費用再会計・ランキングは全て0件。** 有効grant、科学dispatch、旧予算の返却・resetは0。科学的主要claimの採否は判断していない。完了時mandatory STOP。

準備SOURCE：`432b76e0b915c6b145a7d275b4cdd42953b762bd`（初版`b902da67…`に、極小確率のsupport underflowを隠さない診断を追加）。これは準備コードのSOURCEであり、科学実行済みSOURCEではない。

## 1. 技術的結論

| 課題 | 既存証拠 | 不足証拠 | 技術的な次の作業 | 新計算の要否 |
|---|---|---|---|---|
| A：s4 | q4/q8の元lawと保存π4、対象確認n32のN4=0。別条件q1/r1にはorder4の保存観測あり | 対象2候補のs4 full-wrapper費用、層平均・分散・結論感度 | 正しい条件付き多項分布kernelとmock接続を準備。まず未観測層の許容費用を調べる案を別承認へ | 保存値の監査は今回完了。感度の数値評価は別承認。native層費用が必要なら新sampling/compileが必要 |
| B：複数2次 | s2条件付き各n8は全てN2=1。保存count分布にはN2≥2も存在 | N2≥2、N4=0に限定した対象費用。別q/r/Kの費用の移送根拠 | s2内部の部分層として直接条件付け。ちょうど2個に固定しない | 追加費用の取得には新sampling/compile。限定benchmarkを維持するなら必須とは決めない |
| C：選択後効果 | union6固定前後のSOURCE順序・保存hash、候補別n32、両軸pair、保存SE、探索winner/確認点最良 | 選択を考慮した区間、同時比較、全tail費用、厳密な母平均上界 | 保存192 trajectoryを再利用する限定解析案、依存関係・estimandを固定 | 保存値の数値統計解析にも別承認。新compileは最初から必要としない |

今回「未観測」を0費用としない。Cの選択効果、有限標本のばらつき、A/Bの希少support欠測は別問題であり、一つのSEでまとめて解決したとしない。

## 2. 固定資料の取得と再利用

GitHub originに対し7個の指定commitをexact SHAでfetchし、独立clone内のGit objectを照合した。cross-system監査は作り直さず、既存19系列・222保存値・12比較・13不足から関連事項だけ再利用した。

- [H8元map・681ec23c](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/681ec23c93fdd1090d325908f4b155a79115afdf/docs/research/track_a_h8_resource_map_result_v1.md)。execution providerは`c31ef424…`、元dataset/seed SOURCEは`cb8a56c6…`。両者を同一欄に潰さない。
- [6軸反映後の正本・96e69a01](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/96e69a01d1fae7a402aac2c4efa570a59f8cc7a0/docs/research/track_a_h8_six_axis_reaccount_run01_results_20261017.md)。通常/非相殺、5精度×倍率1/10の保存40比較を転記した。最良候補の再探索はしていない。
- [s2 run02・b21e90b8](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b21e90b8368645a5066c06a2a0a2ceb2850d162e/docs/research/track_a_h8_conditional_s2_run02_results_20261018.md)。SOURCE`03479a75…`、16 trajectory/32 native軸、正常終了。
- [横断監査・621d7d7b](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/621d7d7bed183862db8d9f01b6fdb4406f942309/docs/research/track_a_h4_h8_cross_system_evidence_inventory_20261017.md)。この監査時点ではpilot未収録だったが、今回は上記完了結果を独立に追加照合した。旧監査は書き換えない。
- [H6希少次数の先例・eee1343f](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eee1343f6e99aa7661892ddfbe1067e974e2b74a/docs/research/track_a_h6_rare_order_cost_reliability_v1.md)。考え方だけ参考とし、H6確率・RZ・tail上限をH8へ移送しない。

[入力inventory](../../references/h8_rare_tail_selection_inputs_v1.json)は36固定fileのcommit/path/Git blob/bytes/SHA256/GitHub URLを持つ。[metadata監査](../../artifacts/resource_applicability/track_a_h8_rare_tail_selection_readiness/2026-10-18/evidence_audit_v1.json)は、さらにpilot SOURCE closure8＋補助test1のGit bytesを確認した。元NPZ/QPY/private trajectory・使用済みgrant原本にはアクセスしていない。公開receiptに書かれたruntime・preparationのSHAと、現在の実体のfresh照合は区別する。

## 3. 科学条件と現在の費用比較を保持

Linear H8、隣接1.00 Å、STO-3G、保存tol-only DF rank28、16 spin modes/sector4900、T=.8、prefix7、S2、r4、K4。主要2候補は`B2_p7_q4_2nd_r4_K4`（δ=.2、R16）と`B2_p7_q8_2nd_r4_K4`（δ=.1、R32）。元finite-RTE、native symmetric-directional、cosine/sine full Hadamard wrapper、P=0、主指標shot-weighted RZ。

compiler SHA：`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。profile：`622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12`。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、topologyなし。単なるversion一致で代用しない。

以下は正本の丸め済み表を転記したもので、新たな比の計算ではない。正はPR低費用、負はPR高費用。通常と経験的非相殺を分離する。

| ε | 通常・倍率1 | 通常・倍率10 | 非相殺・倍率1 | 非相殺・倍率10 |
|---|---:|---:|---:|---:|
| .05 | −8.0230% | −8.0230% | 16.6243% | 16.6243% |
| .01 | 33.2373% | 33.2373% | 34.6849% | 34.6849% |
| .005 | 31.4549% | 31.4548% | 35.1253% | 35.1253% |
| .001 | 22.9613% | 22.9610% | 39.4150% | 39.4150% |
| .0005 | −50.8125% | −50.8139% | −29.0910% | −29.0921% |

PRは保存n32点費用・paired SE。強化184の決定論比較にはmodel単価が残る。実測winner行も全184実測最良の保証にはならない。厳密bias、母平均winner、end-to-end energy/QPE費用を認定しない。

## 4. s2 pilotから取得済みの証拠

| 候補 | 条件付き本数 | 保存実測平均RZ/軸 | 保存標本SE/軸 | 観測N2 | N4 |
|---|---:|---:|---:|---|---|
| q4/r4/K4 | 8 | 127062.125 | 952.4579937775284 | 各1 | 全0 |
| q8/r4/K4 | 8 | 255555.875 | 1317.070171651295 | 各1 | 全0 |

両軸は同じevent列・seedでpaired、単価はwhole-wrapper実測。これは元s2条件付き分布からのn8であり、「最初からN2=1へ固定した実験」ではない。n8が全てsingleだったことを層全体の費用分布の測定完了と扱わない。元n32へ足してn40にしない。

run01は`KeyError:'partition_hash'`でSTOP、受理0、compile16/trajectory8/occurrence32/control4の予約は返却されていない。修正後run02は新承認・新SOURCE・新grantで正常終了。今回旧STOP・終了gate・消費済みgrant digest/ledgerの保存内容を転記しただけで、旧契約を解除していない。

run02の保存carry込みmaximum_afterはcompile1642/1722、trajectory803/843、occurrence3195/4219、control376/376。未知の過去actualはnull。**control予算に余地があると見なして後続compileを起動できない。** 後続pilotの新control案は別承認の対象。今回新grant・新ledger予約は作っていない。

## 5. A：s4の監査と条件付き分布

保存law・保存πは以下。πの式は元SOURCEで確認し、今回これを実H8の数値で再評価していない。元mapのfinite distribution、pilot入力law、16本のreceiptはliteral一致。

| q/R | p0 | p2 | p4 | π0 | π2 | π4 |
|---|---|---|---|---|---|---|
| 4/16 | .9977962039178625 | .002202983975275984 | 8.121068615199011e−7 | .9653161176498755 | .03467088871948207 | 1.299363064277453e−5 |
| 8/32 | .9994474799589791 | .0005524691195340446 | 5.0921486901421124e−8 | .9824699434972964 | .017528427016409925 | 1.6294862951853818e−6 |

s0=(N2=0,N4=0)、s2=(N2≥1,N4=0)、s4=(N4≥1,N2任意)は排反で全体を覆う。

`π0=p0^R; π2=(p0+p2)^R−p0^R; π4=1−(p0+p2)^R`。

元確認n32では対象q4のorder0が512・order2/4が0、q8は1024・0・0。探索q4/n8も全0、q8/n8にはorder2が1個あるがorder4は0。元map全探索のorder4観測1個は別候補q1/r1/K4、R1のlawであり、whole-wrapper費用を対象q4/q8に直接適用できない。

s4 count lawは

`P(n2,n4|s4)=R!/(n0!n2!n4!)*p0^n0*p2^n2*p4^n4/π4`、`n4≥1,n0=R−n2−n4`。

[実装・sampling設計](track_a_h8_rare_tail_sampling_design_20261018.md)では、O(R²) count supportのlog重みを正規化し、countを直接抽選、位置を一様配置、元のcomponent lawへ渡す。N4=1を固定しない。元normalization/量子重みには条件付きπの補正を掛けない。これは元PRアルゴリズムの変更ではなく費用診断の標本設計。

未測定のs4平均・分散・上界はnull。保存n32のSEや観測最大、部分回路RZの足し算、H6実測、未検証費用modelでは補完しない。

## 6. B：s2内部の複数2次部分層

`s2_single=(N2=1,N4=0)`、`s2_multi=(N2≥2,N4=0)`。両者はs2内部の分割であり、s0/s2/s4と並ぶ独立の第4次数ではない。

保存s2 count分布の先頭は`P(N2=1|s2)=.9835445967192212`（q4）、`.991458036283357`（q8）。全count vectorを監査JSONへそのまま保存した。元の多項分布と一致する式は

`P(N2=1,N4=0)=R*p2*p0^(R−1)`、
`P(N2≥2,N4=0)=(p0+p2)^R−p0^R−R*p2*p0^(R−1)`。

これらの無条件single/multi確率は参照公開bundleに独立scalarとして保存されていないため、**NOT_SAVED_AS_SEPARATE_SCALAR_NOT_RECOMPUTED**。今回補数・和を新たに評価しない。部分層の確率は式と保存count vectorで後続計算に接続できる。

対象q4/q8の探索n8・確認n32には複数2次の観測証拠がなく、s2 pilotも各N2=1。他候補の集計order数が2以上でも、同じtrajectory内に2個以上ある証拠とは限らない。元private rawの個別event回収は今回せず、他条件の個別multi証拠はNOT_VERIFIEDに残す。

count lawは`P(N2=m|N2≥2,N4=0) ∝ C(R,m)p2^m p0^(R−m), m=2,…,R`。ちょうど2個に固定しない。whole-wrapperではbasis境界、rotation fusion、最適化、軸suffixを含むため、single event費用の単純加算は実測の代用にならない。

## 7. C：探索・確認・選択後効果

元SOURCEでは4精度それぞれの探索G_RZ上位2件をlexical tie-breakでunion化し、6候補へ固定した。`selection_before_confirmation.json`を書いてから確認supervisorへ渡す。公開完了監査はその原fileのSHAを保存。今回private原file自体は開いておらず、固定SOURCE順序・公開hash・最終coverageの照合まで。原receipt bytesの独立検証済みとはしない。

| ε | 探索n8での事前固定winner | 確認n32の点最良 |
|---|---|---|
| .05 | q2/r4/K2 | q2/r4/K2 |
| .01 | q4/r4/K2 | q4/r4/K4 |
| .005 | q4/r4/K2 | q4/r4/K4 |
| .001 | q8/r4/K2 | q8/r4/K4 |
| .0005 | 元4精度探索に存在しない | 保存D0以降ではq8/r4/K4 |

prefix7/S2を共通とする。確認後K4への選択を「探索で固定したK4の独立確認評価」と呼ばない。監査JSONには、探索で固定したK2の確認費用行と、確認後K4の評価を**別々に**転記した。固定候補の費用・各候補のRZ/SEは公開CSVから取得済み。

784登録groupのseedが`namespace|dataset SOURCE|input SHA|candidate|phase|replica`のhash導出と一致し、master seed重複0。確認は6×32=192、各seed/event/SOURCE/compilerについてcosine/sine pairを照合した。探索n8と確認n32のseed重複0。これは元の候補・phase別PCG64 stream設計を支持するが、seed相違だけで数学的な独立性を証明したとはしない。

候補間のreplica index一致はpaired designの証拠にならず、共通random-number設計は認定しない。**候補間共分散はnull/NOT_ESTIMATED。** 両軸内のpairingだけは保存event identityで支持される。複数ε、倍率1/10、通常/非相殺は同じ候補のn32回路費用を再利用し、shot重みが異なる。192を条件行数で増やした独立標本数へ置き換えない。

[統計評価設計](track_a_h8_selection_effect_design_20261018.md)には、事前固定候補、同時比較、選択後の候補平均、選択手続の性能というestimandの違いと、holdout/同時区間/選択を考慮した推論/探索的bootstrapの成立条件を記載。今回どれも実H8へ適用していない。

## 8. 技術的優先順位案と残す選択肢

1. **追加検証なしで限定benchmarkとして整理する案**を残す。現在の保存点比較は保持し、母平均優位・tail保証をclaimに入れない。
2. 母平均の強い主張が必要なら、まず**Cの保存値解析**と**A/Bのsymbolic閾値を数値評価するゼロcompile案**から選ぶ。新しい統計・感度実行には別承認。既存n32とlawを利用し、効果が意思決定上重要かを検査する段階。
3. native費用が判断に必要と残った層だけpilotを選ぶ。複数2次は既存s2 pilotの未被覆部分、s4は母平均の未観測tail。π4が小さいという事実だけで不要と断定せず、未測定というだけで両方compile必須とも決めない。
4. 新しい独立holdoutは、確認後に選んだ候補を別標本で評価する必要があり、保存値だけの方法では必要な仮定を正当化できない場合の案。有限holdoutでも希少層・厳密母平均を自動保証しない。

これは数値採点・科学的採択ではなく、入力の充足度と新native取得の必要性に基づく順序案。ユーザー・GPTが、主張の範囲、採用するstage、標本数、仮定・区間の意味を決める。

## 9. 未承認の最小案・拡張案

[stage別draft](../../artifacts/resource_applicability/track_a_h8_rare_tail_selection_readiness/2026-10-18/stage_contracts_UNAPPROVED.json)と[INVALID grant](../../artifacts/resource_applicability/track_a_h8_rare_tail_selection_readiness/2026-10-18/grants_INVALID.json)を作成した。

| stage | 最小案 | 拡張案 | 取得・評価対象 | 現在の認可 |
|---|---|---|---|---|
| S4_PILOT | 2/q、計4 trajectory・8軸 | 8/q、計16・32軸 | 正しいs4条件付きnative full-wrapper費用 | UNAPPROVED |
| S2_MULTI_PILOT | 2/q、計4・8軸 | 8/q、計16・32軸 | N2≥2,N4=0の条件付き費用 | UNAPPROVED |
| SELECTION_SAVED | union6の保存192 trajectory、追加compile0 | 事前固定した全endpointの探索的cluster bootstrap等 | 事前固定候補の評価/選択後効果の条件付き診断 | UNAPPROVED |
| RARE_SENSITIVITY | 保存law・shot・対照費用によるsymbolic閾値の評価、compile0 | 仮定範囲を明示した層費用の感度表 | 仮定の下で結果が変わる費用水準 | UNAPPROVED |

n2/qは両軸pair・失敗/ばらつきの記述が可能な最小運用案、n8/qは既存pilotと同じ運用規模の比較案に過ぎない。必要標本数・検出力・受理基準の保証はない。実測費用の大きさを成功閾値にしない。最小→拡張の自動移行や異常時の補充はしない。

native案は各stageの先頭trajectory/qで元のcontrol数値検査（保存policyの1e−8）を想定し、logical control計8を別予算案とした。新native adapterの最終SOURCE・PCG64/元_make_event接続、private入力bytes/runtimeのfresh照合、未使用seed/lock/ledger登録・control予算承認を完了してから科学実行を認可する。**この準備SOURCEにproduction dispatchは接続していない。** 有効そうなgrantを与えても起動不可。人工kernel/mockの準備完了と、実native接続済みの実行承認を分離する。

CPU/RAM/worker/wallの恣意的な低上限は新設していない。後続nativeは独立trajectoryを実空き資源で並列化し、同trajectoryのevolutionを両軸間で再利用する案。実効資源・既存制御・先頭controlの順序をfresh gateで確認。共有cgroup/OS/他job・既存venvを変更しない。無関係なworktree HEAD/status更新や一時的PSIだけではSTOPしない。

## 10. 人工検証・保全・公開

73件の人工/mock test PASS。小R全列挙20ケース、countとword確率の一致、一様位置、N4複数/複数N2、k+1 component・生成順、R16/R32区別、極小確率とunderflow診断、pair全wrapper mock、SOURCE/input/compiler/profile拒否、固定候補と確認後winner、SEの軸内相関、endpoint共有bootstrap、未観測費用null、探索/条件付き混合拒否、未承認実行拒否を含む。

初回はpackage `__init__`経由で不要なMatplotlib依存が読み込まれ、テスト収集前に失敗した。共有環境を変更せず、stdlib単独moduleを明示importする形に修正。最終73件はPython3.12.3でPASS。これは主担当自身の技術検査で、独立レビューや実native検証ではない。

[技術検証receipt](../../artifacts/resource_applicability/track_a_h8_rare_tail_selection_readiness/2026-10-18/technical_validation_v1.json)、[SOURCE closure](../../artifacts/resource_applicability/track_a_h8_rare_tail_selection_readiness/2026-10-18/SOURCE_closure_v1.json)、[保全](../../artifacts/resource_applicability/track_a_h8_rare_tail_selection_readiness/2026-10-18/preservation_v1.json)。既存の科学SOURCE/result/grant/ledger/marker/STOPを読み書きするrunnerは起動せず、変更は独立cloneと専用補助領域に限定。旧36証拠のGit bytes不変、他チャットのjob/file/refへの変更0。保全は他チャットの正常な更新を禁止する意味ではない。

branch：`track-a-h8-rare-tail-selection-readiness-20261018`。資料result commit・remote照合は専用publication receiptに記録する。認証失敗時は認証設定を変えず、local commitを保持して手動pushコマンドを返す。未公開を公開済みと扱わない。private raw/NPZ/QPY/runtime/cache/grant原本/認証情報の新規公開0。

**次の担当**：ユーザー・GPTが「限定benchmarkのまま追加検証しない」またはstage/仮定/標本数/予算を選ぶ。Codexは選ばれた範囲の新SOURCE・fresh bindingを用意し、明示承認後だけ実行できる。現時点ではmandatory STOP、次stage・追加科学レビュー・追加科学計算は開始しない。
