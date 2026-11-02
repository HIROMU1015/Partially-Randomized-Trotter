# H4強baseline：共通CA・RPE shots・条件付き総資源の技術準備

2026-10-26 JST。**保存B2 CDF8とB3 XDF8を既存13 round合成誤差伝播・RPE／Hoeffding・CA・serial費用へ接続する最小adapterを実装した。人工／回帰76 tests PASS、byte preflight PASS。実H4の新規科学callは0。** 固定した未承認契約は最大**86 logical units**。実科学評価の結果・総資源順位は未取得である。

- 起点／B3結果公開：`866d3ba327abd5ca38af5d7a9347f6c91294958f`。
- 新実装SOURCE：`969c17d623d56eb645417b281deadb1be4757699`。
- Branch：`codex/research-c-c1-h4-strong-ca-preparation-20261026`、独立worktree：`/tmp/prt-c1-h4-strong-ca-preparation-20261026`。
- [adapter](../../src/trottertracks/c1_h4_strong_ca_resources.py)／[passive-default runner](../../scripts/run_research_c_c1_h4_strong_ca_resources.py)／[人工tests](../../tests/management/test_research_c_c1_h4_strong_ca_resources.py)。旧SOURCEは変更していない。
- [未承認科学契約](research_c_c1_h4_strong_baseline_ca_resources_contract_draft_20261026.json)、SHA256 `813857ab62dba2cc785814045e4bad8fc6cb617fd2bbbb883beb262ed64c6b7d`。
- [SOURCE manifest](../../artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/technical/SOURCE.json)、SHA256 `881694bde50cfa1ad50a67fef86c0bffc42c165559a6f8be56e148573e7cad8c`。
- [技術検証・passive証拠・76 PASS log](../../artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/technical/)。共通運用指示2026-10-25版§8.1／8.4／10／10.3に従う。承認証明・承認原文保存・新grantは作成していない。

## 固定資料と保存証拠の適合性

指定GitHub資料を実際に取得し、ローカルの同一file blobと照合した。

| 資料 | 公開commit | 科学SOURCE | 今回再利用する内容 |
|---|---|---|---|
| [B3結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/866d3ba327abd5ca38af5d7a9347f6c91294958f/docs/research/research_c_c1_h4_b3_execution_20261026.md) | `866d3ba3…` | `b40fce5c…`、実装`ae5d2fe9…` | `B3_XDF_spin_relative`のsegment・norm guards・metrics・correctness・SOURCE／STOP |
| [強baseline結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e0c89f611eac92c939edde67890f5cc4391d098e/docs/research/research_c_c1_h4_strong_baseline_execution_20261026.md) | `e0c89f61…` | `88000639…` | `B2_CDF_relative`の同じ種類の保存証拠 |
| [旧CA／shots／費用](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3a8356a7843d3cb46bb8069955640381b490ac4c/docs/research/research_c_c1_h4_ca_resources_execution_20261026.md) | `3a8356a7…` | `18045333…` | 実装・固定target／政策のみ。旧shots／総費用を新routeへ流用しない |
| 元公平native | `f4de9f62c2ff1cbaaa75b6cdaa07217d00dfd957` | `862a97d4…` | full512 head／X／Y shell、DF/PF/own-ground来歴、共通precision/P0／serial政策 |

保存JSONのschema・field・既存SOURCEを調査した。B2 CDF／B3 XDFの両方に `compiled_reference_frobenius`、`operator_bound_pilot`、`norm_upper_pilot`、raw／compiled／reference別 `norm_guards`、full512入力／出力coverageが存在する。head／X／Y shellにも同じfieldがある。旧H4のsaved identity、各候補のDF norm＋guard、own-ground／alias、元PF13 round、shell metricsも利用できる。**追加Operator、回路build／compileや誤差再測定を要する欠損は見つからなかった。**

この適合性確認は保存fieldと来歴の技術確認であり、実B2／B3の新たな13 round伝播やCA判定ではない。新しい数値joinは将来の許可された `run` の内部だけで行う。保存source／contract／STOPと元入力identity、compiler／Gaussian／phase／条件、norm／coverage、PF候補・round対応をその実行でも再検証する。

## 固定科学条件と比較範囲

直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ=2／sector36、controlled full512。保存通常number CDF rank8のB2 relative対、保存OpenFermion1.6.1 eigh元順prefix8 XDFのB3 exact-spin＋relativeだけを比較する。元g補正κ、signed λ/G、核scalar一回、元frame順、元HamiltonianとDF近似を保持する。

全frame ordered symmetric S2、**s=1のみ**、δ=.1、m0..12、n=2^m。CA=.00159 Ha、DF reserve=.00019875 Ha、β_total=π/6、β_stat=π/12、決定論phase room=π/12。failureは各候補.01、各round/axis .01/26。own-DF ground oracle、ideal IID Born、状態準備費用除外。PRは非適用。

両armの保存DF誤差は異なる。各DFのnorm＋guardを一度だけ使い、同じCA条件への適格性を別々に判定する。旧約11.85%の条件付き総RZとB3単一kernelの約1.60%差は、それぞれの元scopeの保存結果として保全し、新しい総資源順位へ転用しない。

Gaussian `nature_jw_exact_zero_lapack_atan2_v1`、P0 `qiskit_P0_mpfr160_declared_phase_v1`、Qiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation_degree1／qubits_initially_zero=falseと既存layoutを維持する。head／kernel／readout分離・all-wire fence付きserial政策を使い、未観測の融合やlong-round再compile削減を仮定しない。

## 最小接続と既存関数の再利用

旧H4 runnerはCDF/XDF×s1/s4の4候補と旧合成profileに固定されているため、無変更で再起動すると今回対象外まで再解析する。旧frozen code／contractは変更せず、新しい単一adapter・CLI・testsの3ファイルを追加した。fitter、Gaussian、合成誤差伝播、shots solver、資源aggregatorは新しく実装していない。

| 処理 | 再利用SOURCE／関数と接続 |
|---|---|
| saved evidence | `c1_h4_ca_resources.preflight/load_bundle/attach_evidence`：旧native／DF予算／PF13／own-ground／alias／shell metricsの照合。旧合成rows／kernel metricsは採用せず、新B2／B3へ置換 |
| full512資格 | 保存segmentのlabel／coverage512×512、component allowance1e-10、signal2e-10、phase fitなし、level B、direct Frobeniusと3種類norm guardを照合 |
| 合成誤差 | `c1_h4_full_synthesis.propagate`を各候補×m0..12で直接呼ぶ。`s=1,m=m,saved_pf_error=元PF`。head／kernel／tail・両行列norm-growthを既存式のまま保持 |
| RPE／shots | `c1_h6_ca_resources.assess_saved_round` → `AccuracyTarget`／`RoundEvidence`／`assess_round`／既存Hoeffding solver／`assess_fixed_round`。PFと合成を一度ずつ入力 |
| CA | `assess_energy`：全13 roundとfailureを検証後、DF norm＋guard［Ha］と最終RPE位相のenergy intervalを接続。PF／合成を直接Haへ足さない |
| serial費用 | `serial_round_cost` → `compose_serial_cost`：axisごとに `2^m × kernel + head + readout_axis`。保存整数metrics・measurement・depth定義を保持 |
| 総費用 | `aggregate_candidate` → `aggregate_totals`／`compare_totals`：共通CA数値条件PASS候補だけ集計・登録一組だけ比較 |
| 実行／保全 | 既存`Calls`／`resource_plan`／progress、fresh output、partial evidence／STOP、候補独立scalar thread処理 |

`propagate`のcombined誤差をPFとして再投入しない。RPEへ渡すのは元PFと合成signalの別fieldで、既存関数がcombinedとの一致を確認する。決定論のattenuation=1、corrected RTE=0、reference-defined law discrepancy=0を維持する。未知の合成誤差に対する0認定は行わない。

既存の伝播がoverflowを例外にする箇所はadapterで `UNRESOLVED_SYNTHESIS`へ変換する。bound／norm guard／PFが欠損した場合もnullを保持してUNRESOLVEDとする。非有限・負値、coverage／SOURCE／候補・round／compiler・reference等の不一致、guardを失う矛盾はcorrectness FAILとする。既存fixed-N未達、coordinate／shotsの数値未資格、N≥2^53、総費用の非有限・整数範囲外も既存の未資格処理を維持し、clip／shot数増加／solver変更／fallbackはしない。

## 人工検証

**新規23＋既存回帰53＝76 tests PASS（2.18秒）。** [final log](../../artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/technical/tests_final_verified.txt)。旧H4 CA22、H6 CA29、full512 synthesisの重要2 testを再利用した。新testsは旧H4の人工fixtureも再利用し、既存関数の内容を写して検証する重複testを避けた。

正常系では人工2候補×13 round、26合成profile・26 X/Y shots pair・2 CA・26 X/Y費用pair・2集計・1比較を接続し、86/86 units、n=2^m、元PFと合成の一回計上、norm guard、fixed-N margin、Haとphaseの接続、serial head/kernel/tail、shots加重和を確認した。精度条件未達／欠損bound・guard／overflowでは当該候補の費用を生成せず、他候補の評価を保った。両候補のfixed-N未達では60 unitsで未資格、costなしとなった。

SOURCE、候補、full512 coverage、norm guard loss、nonfinite、shell binding、科学条件、PF対応の不一致はFAIL・依存する後続call停止・partial evidenceを確認した。fresh run01再起動を拒否しSTOPを上書きしない。旧runner／B3／強baseline runnerはmockで遮断、passive importはNumPy／SciPy／Qiskit／ffsimをimportしない。

初期fixtureで旧exclusive `save`に既存の人工JSONを再保存して20 testが停止した。変更は人工fixture内の更新方法に限定し、旧production保存関数は無変更。修正後21 PASSを経て不足するPF欠損／fixed-N接続2件を追加し、最終76 PASSとした。これらは人工検証の履歴で、科学retryではない。人工PASSを実H4のCA・shots・総資源PASSへ昇格させない。

## 限定科学契約と実行規模

| logical call分類 | 最大 | 単位 |
|---|---:|---|
| saved_bundle_load | 1 | pinned scalar JSON bundle・既存identity／target checks |
| evidence_attachment | 2 | B2 CDF／B3 XDFの候補別接続 |
| round_synthesis | 26 | 2候補×13round、既存propagate一回 |
| round_accuracy | 26 | 同roundのX/Y対称shots solve＋fixed-N margin検証一組 |
| energy_assessment | 2 | 候補別共通CA数値判定 |
| round_serial_cost | 26 | 適格候補×13round、X/Y pairの費用一組 |
| cost_aggregation | 2 | 候補別総費用。未資格ならnull |
| cost_comparison | 1 | B2 CDF対B3 XDFの登録一組 |
| **合計** | **86** | 過去119／65の残枠流用なし |

1候補がCA未達／未資格で費用13 pairをskipする場合は最大73 units、両方なら60 units。correctness FAILではさらに手前でSTOPする。未達は登録十分条件の未達であり、物理的な不可能性の断定ではない。主指標は条件付き総RZ、副指標はCX・各depth・size・総shots／executions。serial加重depthは全round一体compileの最小depthではない。

対象は小規模JSONとscalar計算だけで、dense512行列・NPZ／QPYの数値decodeは不要。保存原packet・reference・shell／groundのbytes／metadataを固定するが、これらの配列は読まない。新fit／DF-PF／ground／Operator／build-compile／量子outcome抽選は0。古典的処理時間・RSSと実行時資源を保存する。

既存Python3.11.1／依存lock／venvをread-onlyで利用する。起動はvenvのPython実体を直接指定する。独立した2候補のscalar chainは実際の空き資源・quota・pressureに応じた既存resource_planで並列化する。heavy BLAS処理はなく共有thread設定を変更しない。CPU／RAM／worker／wallへ新たな恣意的上限を設けず、共有環境／他ユーザーのjob／cache／venvは変更しない。

## 残る未資格事項とmandatory STOP

保存full512 Frobenius／norm guardはlevel B数値pilotで、厳密なreference・expm・Operator・normの丸めenclosureではない。新しいCA判定も登録十分条件に基づく数値資格であり、実測energy誤差・厳密CA certificate・状態準備込み／実機の総資源保証ではない。candidate別failure.01は、2候補同時・事後選択後の保証ではない。未知値／厳密資格はnull／falseを保持する。

実装の技術的blockerはなく、追加のOperator／回路実験も現時点では不要。将来の固定範囲で同CA条件を満たす候補だけのshots加重順位、各誤差・shots・単価との対応を取得できる。結果前に総資源順位や原因を断定しない。CDF/XDF一般の優劣、全分子・rank最適性、新規性、C1の継続／縮小／終了は判断しない。重要な科学レビューも実施していない。

旧SOURCE／raw／grant／ledger／marker／STOP、約12%の旧総RZ、B1/B2/B3、XDF9、H2/H6、C4／singletonの終了状態は不変。今回の技術準備では科学run01は未作成・未消費、新科学call0である。

**次の担当：ユーザー／研究C**が公開SOURCEと86-call契約の限定科学実行を別途明示許可するか判断する。**Codex**は許可後のみfresh outputで一度実行し、証拠を公開して停止する。許可を証明する別手続きは不要。

**今回の技術準備はcommit・push・remote SHA／SOURCE／成果物照合後mandatory STOP。実H4の13round合成・shots・CA・総資源を自動実行しない。**
