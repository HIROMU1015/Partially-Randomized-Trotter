# H4 B3：spin対応XDF基底＋相対Gaussian融合の技術準備

2026-10-26 JST。**B3専用の最小接続・人工検証・未承認65-call契約を準備した。新21＋既存101＝122 unique tests PASS。実H4の新規数値decode／build／compile／Operator／Statevector／保存値再解析／費用比較は0。** B3のcorrectness・改善量・研究上のGOは未判定。公開照合後mandatory STOP。

- 起点：前段公開 `e0c89f611eac92c939edde67890f5cc4391d098e`。
- B3実装SOURCE：`ae5d2fe9330108e1cfeef5080bbfe7ec74eef246`。
- Branch：`codex/research-c-c1-h4-b3-preparation-20261026`、独立worktree `/tmp/prt-c1-h4-b3-preparation-20261026`。
- [新adapter／単一route runner](../../src/trottertracks/c1_h4_b3.py)／[CLI](../../scripts/run_research_c_c1_h4_b3.py)／[B3人工tests](../../tests/management/test_research_c_c1_h4_b3.py)。旧SOURCEは無変更。
- [未承認65-call科学契約案](research_c_c1_h4_b3_science_contract_draft_20261026.json)／[SOURCE・検証・テスト証拠](../../artifacts/research_c_c1_h4_b3/2026-10-26/technical/)。
- SOURCE manifest SHA256：`8c2d85c25800bc76c23545f68f5ff31a6cd4fae049560fb51df7ea3273aab28a`。
- 契約SHA256：`0748fc10663fd96a7147abff77da19ad70e1f50f7c65811df8df3333f43972f4`。

## 固定資料・再利用する範囲

[前段227-call結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e0c89f611eac92c939edde67890f5cc4391d098e/docs/research/research_c_c1_h4_strong_baseline_execution_20261026.md)、[前段技術報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/880006396326a4d46f4a4030564279e92fadb9cf/docs/research/research_c_c1_h4_strong_baseline_preparation_20261026.md)、[固定227-call契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/880006396326a4d46f4a4030564279e92fadb9cf/docs/research/research_c_c1_h4_strong_baseline_science_contract_draft_20261026.json)をGitHubから取得し、ローカル固定版のSOURCE／schema／入力経路を確認した。共通運用指示2026-10-25版§8.1／8.4／10／10.3を適用する。重要な科学レビューは実施していない。

対象は直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ2／sector36、system full256／ancilla-last controlled full512。元OpenFermion1.6.1 eigh順prefix8 XDF、保存CDF8、元g補正κ・signed λ/G・核scalar一回・元frame順、全frame ordered symmetric S2を保持する。新規対象は **B3 XDF8、s1、h=δ=.1のcontrolled kernel一個とm0 X/Yだけ**。PR splitなし、own-XDF ground oracle／状態準備除外。CA=.00159 Ha／DF reserveCA/8、13round、RPE測定政策は背景条件として維持し、今回の将来契約で評価しない。

| 保存route（s1 kernel） | RZ | CX | total depth | 今回の扱い |
|---|---:|---:|---:|---|
| B0 CDF8 | 12,272 | 7,388 | 12,806 | 保存scalarのみ再利用 |
| B0 XDF8 | 13,944 | 7,756 | 13,113 | 保存scalarのみ再利用 |
| B1 XDF8 spin | 11,821 | 7,176 | 12,556 | spin recipe・G証拠・metricsを再利用 |
| B2 CDF8 relative | 11,062 | 7,008 | 12,127 | 主要な比較先、再buildなし |
| B2 XDF8 relative | 11,307 | 7,008 | 12,122 | 融合単独の比較先、再buildなし |
| B3 XDF8 spin＋relative | 未測定 | 未測定 | 未測定 | 次回許可後に一度だけ新規評価 |

上表は前段の保存値の引用で、今回の再集計・再ランキングではない。CDF/XDFのDF近似誤差は異なり、既報の約12%条件付き総RZとは費用scopeが異なる。B3へB1/B2のfull512・P0・adjoint・X/Y PASSを転用しない。

## 最小実装と旧SOURCEの保全

B1/B2実装SOURCE `ea5fff03be761056fdf826b4b6eebd442bfc2d89` の `prepare_xdf_basis(..., basis_policy=SPIN)` と `StrongBuilder(relative_fusion=True, basis_policy=SPIN)` を直接組み合わせる。新しいGaussian／Givens合成器、DF分解器、Hamiltonian変換、compiler、誤差評価法は作らない。

旧 `strong.execute` は3route、CDF preparation、局所CAR再構成へ固定されているため、B3一件に使うと不要な旧route処理を伴う。旧 `ROUTES` や契約validatorを変更せず、新3ファイルの専用入口へ限定した。追加コードは入力binding・単一routeのorchestration・保存／STOPであり、科学的な数値処理は既存関数を呼ぶ。

| 再利用SOURCE／関数 | B3での用途 |
|---|---|
| `c1_h4_strong_baseline.prepare_xdf_basis`／`model` primitives | B1と同じspatial eigh→exact interleaved spin lift。κ/base＋8frameのG再構成／unitarity証拠、λ符号・順序を保持 |
| `StrongBuilder`／`runs`／`combined_diagonal` | 同じS2／同一basis共有／fsumとR=Uⱼ†Uᵢ、解析的真空位相補正。対角を跨ぐ並べ替えなし |
| `c1_h6_native_precision.gaussian` | 固定exact-zero/LAPACK/atan2 Gaussian、再実装なし |
| `c1_h4_native_fair.matrix_load`／`decode`／`controlled` | 保存独立CAR/PF targetとshellの読込、全512 input/output、diag(I,U)の絶対比較 |
| `fair.compile_checked`／MPFR160 P0 | 既存Qiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation_degree1／qubits_initially_zero=false・既存layout等 |
| `fair.synthesis.segment`／`component_check` | raw/ref、P0/raw/ref、unitarity・Frobeniusの数値pilot。位相fit／full SVDなし |
| `native.actual_adjoint`／`ancilla_z`／`signal_gate` | 実inverse回路の復元と独立reference比較、raw/P0 X/Y complex signal |
| `circuit_metrics`／`compare_metrics` | 同じnative指標と差・RZ比、独自scoreなし |
| `integrated.Calls`／`resource_plan`／`CompileSerialCalls` | 既存call journal、fresh output、部分証拠、共有translator排他、空き資源に応じた並列化 |

B3のbuilder/cacheは新規instanceで、旧B1/B2のcacheを共有しない。既存keyの版・precision・basis policy・元SOURCE/input・方向付きrotationを保持し、QPY metadataへ `B3_XDF_spin_relative` を明記する。global phaseやscalarを落とさず、核scalar・中央I項の二重計上も行わない。

保存B1証拠はη・再構成残差・recipeを含むが、Uそのものの独立配列は含まない。従って保存Uとのbitwise一致は主張しない。同一SOURCE／入力／固定eigh環境／recipe、frame ID・signed λ・η順序との対応を確認し、B3の**新しいfull512独立reference gate**でoperatorを検証する。人工入力ではB1 recipeによるmodel／rotation／diagonalそのものの一致を確認した。

## 保存証拠の再利用条件と科学gate案

元packet SHA256は`8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。前段科学SOURCEは`880006396326a4d46f4a4030564279e92fadb9cf`、B0科学SOURCEは`862a97d4213121749448ca716cde820a44b99772`。新preflightは旧byte preflightを読み取り専用で再利用し、旧runnerは実行しない。旧契約のB3=0やSTOPは当時のscopeとしてそのまま保持する。

旧227-callのresult／SOURCE／STOP／契約／identity／spin、およびB0 result／SOURCE／STOP／fairnessの10 JSONを一つのevidence bundleとして確認する。保存DF reserve・元ground/alias・P0／precision・単一s1 kernel scopeを照合する。元CDFの数値配列やfit exportをdecodeしない。

referenceは前段が検証した**元Gの独立CAR由来XDF full256 S2 target**一件、groundは既存XDF sector36 oracle一件、head/X/Yは既存raw/P0 full512 shell六件を使う。referenceの再生成・候補basisからの自己referenceは行わず、ground eigensolve、DF/PF再計算、旧回路build/compileも0とする。

将来の実行順は次の通り。

1. SOURCE35・直接保存input19・原packet・旧binding／依存9版／空き資源を照合し、新規fresh outputを作る。
2. 保存evidenceと元packetを読み、B1と同じspin recipeで9blockを構築。G再構成・unitarity、spin順、frame／η／λを確認する。不適格構造や非有限値ではFAIL、fallbackなし。
3. B3 raw kernel一件と最大16境界の融合記録を生成し、full512のraw/reference最大成分1e-10を**P0前に**検査する。
4. raw actual adjoint→P0一件→compiled actual adjointを、translator変更が重ならない順で作る。compiledと二つのadjointのOperator復元は独立taskとして並列化する。
5. raw/P0/reference・unitarity、actual adjointのdagger／reference／復帰を最大成分1e-10／rtol0で比較する。saved shell／own-XDF groundを用いたraw/P0のX/Yはcomplex絶対差2e-10。
6. **全correctness PASS後だけ**新metrics一件と、B3/B1 XDF・B3/B2 XDF・B3/B2 CDFの固定3比較を保存する。B0 metricsは対応用の引用だけで新比較・再集計を増やさない。

全512入力column・全512出力成分の数値pilotで、sector-onlyや有限probe、global-phase fittingでは代用しない。Frobeniusは測定した残差であり、最大成分gateの閾値とは区別する。厳密丸めenclosure・13round合成上界・CA・shots・総費用は未資格／契約外で、unknownはnullとする。

## 65-call上限の内訳・資源・STOP

| logical call群 | 上限 |
|---|---:|
| 元input・保存evidence bundle・identity | 3 |
| B1と同じmodel prepare＋model gate | 2 |
| absolute Gaussian要求9＋relative cache miss最大16 | 25 |
| 保存reference1＋shell6＋ground decode/normalization1 | 8 |
| B3 raw build＋fusion gate＋raw full512 gate | 3 |
| P0 compile1＋raw/P0 Operator2＋segment assessment1 | 4 |
| actual adjoint build/Operator/gate各2 | 6 |
| signal reference1＋prepare2＋axis4＋complex gate3 | 10 |
| 新metrics1＋固定比較3 | 4 |
| **合計** | **65** |

model prepare一単位に既存の9spatial eigensolve／G・unitarity検査を含める。これはDFの新分解ではない。absolute Gaussianはcache hitも要求として数え、relativeはmissだけ。9blockのS2は最大17run／16境界なので上限16で、実績が少なければそのまま記録する。上限を埋める追加処理はしない。前段227単位と低水準操作数を単純比較しない。

新規QPYはraw/P0／各actual adjointの**4件**、new full512 Operator packetも**4件**。spin/model/fusion/segment/adjoint/XY証拠、3比較、call journal、resources、SOURCE、STOPを保存する。Gaussian生成・compileの古典wallはjournal、全体CPU/RSSはresourcesから取得できる。

旧timingではOperator復元が支配的だったため、raw gateという逐次依存を維持した上で、P0後のcompiled／raw-adjoint／compiled-adjoint復元3taskを既存resource_planで並列化する。現passive snapshotはaffinity32 CPU／空きRAM約52.5 GiBで、worker3を選択。実行直前にquota/pressure/空き資源を再確認する。CPU/RAM/worker/wall capなし、BLASはjob内で協調、cache/tempはtask内、共有環境・他ユーザーjobは変更しない。速度向上倍率は未測定。

fresh outputは`artifacts/research_c_c1_h4_b3/<execution-date>/run01`、retry/resume0。最初のSOURCE/input/spin/位相/順序/reference/compile correctness FAILで依存callを停止し、開始済みdecodeを安全回収してpartial evidenceを保存する。正常・FAILいずれも公開照合後mandatory STOP。結果を見たrank／route／閾値／compiler変更、旧run再開は行わない。

## 人工・回帰テストと保全

新B3の21 testsは、両optionの同時適用、B1 model一致、正負時間、control0/full32、scalar/identity位相、actual adjoint、X/Y wrapper、cache/route/QPY metadata、非有限・不適格spin拒否、SOURCE/入力binding、raw gate前後のSTOP、fresh output・部分証拠を確認した。人工end-to-endではB3のみの4 QPY／4 Operator、3比較と全call分類を通過。旧runner/CDF preparation/CDF export/13round propagationを呼ぶとFAILするmockを入れている。

最終B3単体は**21 PASS／6.78秒**。既存のreal-Qiskit74件は混在batchで全PASS、旧Stage2 mock27件は独立processで**27 PASS／.72秒**。合計**122 unique PASS**で、旧101 testsのSOURCEは不変。途中の混在batchは91 PASS／4 FAILだったが、FAILは新preflight fixtureの保存SOURCE／契約名と新manifest／契約名の衝突に限定された。`old_` prefixと独立manifest名に直し、最終B3全21件がPASSした。診断ログも保全し、混在batch全95 PASSと表現していない。

SOURCE35・直接保存input19、旧preflight経由を含む保全対象88件と原packetのbyte／schema／環境preflightはPASS。実H4の配列・QPYを数値decodeせず、新規科学call0。旧B0/B1/B2・H2/H4/H6・XDF9のraw、SOURCE、grant／ledger／marker／STOP、C4／singleton終了状態、元workspaceを変更していない。

## 次の担当とmandatory STOP

技術的なblockerはないが、実H4 B3のfull512 correctness・native費用は未検証である。将来の結果では、B3/B2 CDFのRZ・CX・各depth・sizeの差とtrade-offを報告し、単一指標だけの全体勝者・一般分子の優位・同CA総資源・独立新規性・正式なC1継続／終了を認定しない。差が小さい場合も無断で候補探索を増やさない。

**ユーザー／研究C**が固定SOURCE・65-call案を確認し、B3だけの科学実行を別途明示許可するか判断する。**Codex**はその許可後のみfresh run01を一度実行し、証拠を公開して停止する。承認証明書・承認原文保存・新grantは不要。

**今回の技術準備はcommit・push・remote照合後mandatory STOP。** 実B3 build/compile・Operator/Statevector・native比較、s4／13round、CA/shots/総資源、追加fit・別研究・重要科学レビューへ自動進行しない。
