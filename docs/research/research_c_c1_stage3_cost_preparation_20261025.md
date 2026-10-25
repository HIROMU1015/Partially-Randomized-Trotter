# C1 Stage 3：8候補限定native費用評価の技術準備

2026-10-25 JST。起点 `f85ac4c131b939bf4b8146a98997e61565345d45`。
Branch `codex/research-c-c1-stage3-cost-preparation-20261025`。

**cost-only SOURCE・人工13件＋既存回帰81件＝94 tests・byte preflightがPASS。未承認の科学契約は最大936 logical units。実H4数値decode・回路build／compile・保存shotsによる総費用再解析はすべて0。公開照合後mandatory STOP。**

[未承認cost-only契約](research_c_c1_stage3_cost_contract_unapproved_20261025.json)／[SOURCE・人工結果・preflight・STOP](../../artifacts/research_c_c1_stage3_cost_technical/2026-10-25/)。既存科学結果・SOURCE・grant・ledger・marker・STOPは無変更。新承認証明書・承認記録・grantは作成しない。正式共通指示2026-10-25版は確認済みのSHA256 `e1e3e2625e2b728977402bded6583658e7fb762160f5e787fd45572af0095720`と一致する。

## 対象と接続SOURCE

保存P-H4（直鎖1.00 Å・STO-3G、Nα=Nβ=2、sector36／full256）、CDF保存rank8 number・元g補正κ／XDF保存rank10。CA0.00159 Ha・DF reserve CA/8、δ0.1、m0..12、own-DF ground-state oracle・状態準備除外・ideal finite-RTE IID lawを保持する。

各armで、決定論ordered symmetric S2のs1／s4と、PR L_D2のs4/r16/K0／s4/r16/K2、計8候補だけを評価する。元frame順・one-body・scalarを変更しない。元20候補の保存値を保全し、対象外12候補は「今回のcost pilot未選択」として出力する。科学的NO-GOとは扱わない。

新SOURCEは`src/trottertracks/c1_stage3_cost_only.py`・専用CLI・人工testの3ファイル。旧SOURCEを変更せず、次を再利用した。

| 既存実装 | 用途 |
|---|---|
| Stage1保存export loader／一般K／Stage2 native value model | 保存κ/J/U/constantとXDFからnativeの値表現を準備。新fitなし |
| `c1_stage3_molecular.MolecularBackend` | 入力whitelist、Gaussian cache、full-frame provider、readout shells、native calibrator、QPY保存 |
| `c1_stage3_connection`・修正済み`native_bridge` | LD2 tail／identity分離、有限RTE設定・既存sampler、元native emitter、serial費用積算、fresh occurrence materializer |
| P0 `compiled_wrapper`／既存`circuit_metrics`／`component_check` | phaseを保つcompiler、RZ/CX/depth等の費用、局所raw／compiled照合 |
| `c1_stage3_qualification` | 全有限wordが既存32枠に収まるときだけsupport網羅。未資格と点推定を区別 |
| `c1_stage3_integrated` | 元候補順、call journal、partial progress、利用可能資源に基づく並列計画 |
| `c1_stage3_accuracy.compare_totals` | 費用範囲の比較のみ再利用。CA・shots再算定・旧総費用gateには接続しない |

旧molecular `prepare()`はCAR・ground／alias・exact evolution・PF/RTE referenceを含むため、そのまま再起動しない。新subclassはnative値に必要な部分だけを準備し、保存shotsの専用loaderを使う。Stage2保存D(s1)のshared kernel費用は各armで再利用し、QPYのhash来歴は保持するが再decodeしない。D(s4)とPRの新kernel・3 readout segmentだけを将来build／compileする。

## 保存shotsと条件付き費用

未来の実行で読むのは、保存shots科学commit `f85ac4c131b939bf4b8146a98997e61565345d45`の結果／STOPと対象8候補の13round×X/Y記録である。科学SOURCEは`f8fd2d373fc81353330e732c46b3b5d471e7c683`。保存の十進整数をそのまま用い、shots式・PF/RTE誤差・attenuation・ground／aliasを再計算しない。

主指標は、保存された楽観的shotsを用いる**条件付き期待総RZの校正点推定**。副指標はCX、RZ depth、CX depth、total depth、circuit size。各round／axisの費用を旧`compose_serial_cost`で作り、

```
n_m = s * 2**m
estimated_C(m,axis) = n_m * estimated_E[C_one_occurrence] + C_head + C_axis_tail
optimistic_total_metric = sum_m,axis N_saved_opt(m,axis) * estimated_C(m,axis)
```

を集計する。8×13×2＝208費用行を保存する。主指標の比較はCDF/XDF同設定4組と、各arm内のPR2設定×決定論2設定の8組、計12組に固定。候補の追加探索・最終プロトコル選択・総資源順位付けは行わない。

旧`aggregate_totals()`はsame-accuracy資格を要求するため、`same_accuracy=True`を偽って呼ばない。今回はそのgateを保持し、保存楽観Nでのcost-only加重和だけを最小追加した。十進整数Nと宣言されたsample meanの有理数積・和を保持し、点推定floatに加えてrational文字列も出力する。これは校正値の表現精度を保つ処理であり、samplingや未知誤差の保証ではない。

**楽観的shotsは実際の保証付きshotsではない。** `synthesis_signal_error`／`actual_law_signal_error`／`CA_achieved`はnull、`guaranteed_total_resource_superiority`はfalseを保持する。CDFとXDFのDF物理近似精度の差も残る。同CA達成や保証付き総資源優位を報告しない。

## trajectory・basis共有・depth・law support

PRは既存master seed `2026102503`と、arm／candidate／sample indexを含む独立streamで、各候補32 occurrenceを校正する。同じwordが偶然再出現した場合、compileは候補内でcacheするが、sample meanでは出現回数を保持する。有限bit samplerによる点推定を理想law平均の厳密値へ転送しない。K2の稀なwordを32標本が観測しない可能性もあり、未観測wordの費用を0と認定しない。

長roundの量子protocolは**各shot・各coherent occurrenceでfresh独立trajectory**を使う。費用の`n*E[C]`は線形期待値の積算であり、1個のsample回路をn回再使用するprotocolではない。小さな人工native sequenceで、occurrenceごとのfactory呼出し・別seed・all-wire fenceを確認した。未来のcost-only runnerは長q回路をmaterializeしない。

校正sampleを各roundへ再利用しても、校正誤差を独立とみなしてsqrt(n)で縮めない。旧serial helperどおり費用範囲もn倍する。depthは、native kernelとreadout segmentをfenceで区切って実行する**scheduled serial sum**であり、全長をglobal compileしたdepthではない。

P0はQiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation_degree1.0、qubits_initially_zero=falseで固定。両armにfull-frame Gaussian、隣接する完全同一basisの共有・角度結合を同じ規則で適用する。literal/sharedの未観測利得や、occurrence境界の追加融合割引は入れない。headとX/Y readoutは既存どおり別々にcompileし、ancilla measurementを含め、system状態準備費用を除外する。

費用範囲は、既存32-word枠に**全ての支持word**が収まり、compiler費用を網羅できたときだけmin/max envelopeを付ける。その範囲は支持lawの期待値を覆うもので、sample平均のconfidence intervalではない。sample最大値・最小値や標準誤差だけでsupportを資格付けしない。理想lawで正の重みがbinary64でunderflowしている場合もsupportを勝手に削らず未資格とする。

網羅できなければ`UNRESOLVED`、費用上下限null、比較は`POINT_ESTIMATES_ONLY`。範囲が得られても「固定楽観Nでの費用範囲の関係」に限り、合成／law誤差やCA・実shotsの資格には昇格させない。

## 正しさと人工検証

新しい各segmentはraw／P0 compiledのfull operator最大成分誤差1e-10、compiled unitarity1e-10で照合し、global phase fitはしない。これは局所compilerの診断である。旧Stage2のnative emitter検証と人工検証を再利用し、今回のcost-onlyでは独立分子CAR／PF/RTE／X/Yを測り直さない。sampleの数値残差を全law合成誤差上界へ転用しない。

| 人工検証 | 結果 |
|---|---|
| 新規8候補cost-only end-to-end／unit／mock | 13 PASS |
| 既存Stage3精度・統合・feasibility・shots回帰 | 81 PASS |
| 最大936 call、8候補・208行・12比較、対象外12保持 | PASS |
| 変動するoccurrence費用の平均を積算、fresh seed／factory／fence | PASS |
| 点推定／全support範囲、未資格の優位拒否、CA未知の保持 | PASS |
| 保存Nの任意精度整数、有理数加重和、scope／欠測／非有限値拒否 | PASS |
| 人工分子配列で実native値準備・旧calibrator・P0・metricsへ接続 | PASS。128校正seed重複なし。CAR／PF/RTE／shots式はmockで禁止 |
| 人工raw／compiled相対位相の破壊検知、partial証拠・fresh／retry拒否 | PASS |
| 実入力／保存shotsのbyte-only preflight | SOURCE19／保存入力20 hash／既存5依存版PASS。科学JSON decodeは禁止して照合 |

今回の実H4 decode・実H4回路build／compile・保存shotsを使った総費用再解析・新fit・PF/RTE再計算はすべて0。実際の8候補native費用やPRのshots増加を補えるかという科学結果はまだ得ていない。人工回路だけをbuild／compileしたPASSから分子の優位を推定しない。

## 未承認契約・科学call・STOP

[新契約](research_c_c1_stage3_cost_contract_unapproved_20261025.json)は19カテゴリ、**最大936 logical units**。旧5,637／639／282の実行枠を再利用しない。

| 科学call | 上限 |
|---|---:|
| input／fit export／native value models | 1＋1＋2 |
| 保存shots bundle／保存Stage2 metrics／LD2 split | 1＋1＋2 |
| full Gaussian basis requests | 20 |
| PR finite-RTE設定／support probe | 4＋4 |
| PR occurrence校正sample | 128 |
| 新segment build／compile／phase・unitarity照合／metrics | 各133 |
| native calibration／PR費用範囲判定 | 8＋4 |
| round／axis serial費用展開 | 208 |
| 楽観的Nによる合計／固定pair比較 | 8＋12 |
| 合計 | **936** |

133新segmentの上限はPR4×32 unique word、D(s4)2、readout shell3。D(s1)2は保存Stage2 metricsを再利用する。cache hitや既存supportの小ささにより実callは減り得る。各logical unit内の単純scalar演算・matrix積を二重計上せず、複合処理の範囲は契約に記す。byte preflightは0。

実行順はpreflight→fresh output→保存shots／native値→対象8だけの校正→局所compiler照合→208費用行→8合計→12比較→証拠保存→mandatory STOP。最初の科学的FAIL・compiler不一致・保存shots不整合で後続依存callを停止し、既に動作中のcallを回収、raw／compiled QPY・部分metrics・progress・call・failure／STOPを保全する。support未資格は点推定として続行するが保証付き優位へ進めない。retry／resume0、新fit／optimizer0。fresh outputは`artifacts/research_c_c1_stage3_cost_science/<date>/<fresh-run-id>/`。

既存resource planがaffinity／cgroup quota／空きRAMから最大8独立candidate workerを選び、process内BLASを調整する。Gaussian cache・候補内word dedupで重複を避ける。支配的なPR4taskは並列、既存calibrator内の32sampleは逐次であり、これは残る性能上の制約として記録する。CPU／RAM／worker／GPU／wallに人工上限を設けず、未検証GPUのspeedupは主張しない。task専用worktree・cache・outputだけを使い、共有環境・既存venv・他ユーザーに変更を加えない。

CLIは既定でbyte preflightだけ。

```bash
/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python \
  scripts/run_research_c_c1_stage3_cost_only.py --mode cost-only \
  --contract docs/research/research_c_c1_stage3_cost_contract_unapproved_20261025.json \
  --input /path/to/saved/P-H4.npz
```

別明示科学許可後だけ同じentryへ`--execute --output <fresh>`を追加する。今回、その実行はしていない。承認証明書・承認記録・新grantは不要。

次はユーザー／研究Cが8候補・32校正sample・936-call契約と点推定／supportの制約を確認し、限定cost-only科学実行を許可するか判断する。Codexは別明示許可後だけ保存SOURCE・input・shots・fresh outputで実行する。今回はcommit・push・remote照合後**mandatory STOP**し、実H4の費用評価へ自動進行しない。
