# 研究C：F2 κ Hermiticity・6-call限定診断結果

2026-10-27 JST。**DIAGNOSTIC_COMPLETE、6/6 calls COMPLETE、correctness FAIL0、retry/resume0。** 保存hと保存gの厳密縮約に既に非対称成分があり、NumPy縮約・減算にも非対称な丸め寄与を認めた。κ補正の演算だけで非対称性が生じたケースではない。元積分生成や内部reduction treeへの一意な原因帰属は未解決である。

κの反Hermitian成分は登録Stage 1数値allowance内だが、F2のexact Hermiticity条件は不成立のまま。これは許容差の異なる二つの判定結果であり、旧F2のSTOPを解除するものではない。入力・κ・閾値・targetは変更していない。

## 固定SOURCE・対象・範囲

- 科学実行SOURCE／起点：`802b2b0a16d3bd26ae2554e96a41c1bf8155a2e2`。独立branch：`codex/research-c-F2-kappa-diagnostic-execution-20261027`。
- Scope：`C_PR_AWARE_DF_KAPPA_HERMITICITY_DIAGNOSTIC_V1`。
- [固定契約](research_c_pr_aware_df_F2_kappa_diagnostic_contract_draft_20261027.json) SHA256：`ffeda0e5ffaf5e0b08d3a94818e824218a6570429e590117176fcbb63242e7ae`。
- [SOURCE manifest](../../artifacts/research_c_pr_aware_df_kappa_diagnostic/2026-10-27/technical/SOURCE.json) SHA256：`408fe5114bd32deef82279aae7d9c14461e5e4de097d6473846a29d0b84b178c`、138 files。
- [診断原本](../../artifacts/research_c_pr_aware_df_kappa_diagnostic/2026-10-27/diagnostic-run01/)／[実行前確認](../../artifacts/research_c_pr_aware_df_kappa_diagnostic/2026-10-27/diagnostic_execution_preflight.json)／[要約](../../artifacts/research_c_pr_aware_df_kappa_diagnostic/2026-10-27/diagnostic_execution_summary.json)／[postflight](../../artifacts/research_c_pr_aware_df_kappa_diagnostic/2026-10-27/diagnostic_execution_postflight.json)。

保存直鎖H4・隣接1.00 Å・STO-3G、Baseline、通常number CDF rank8の元入力文脈。Nα=Nβ=2/sector36、full8-mode/256＋last ancilla/512は既存F2の文脈で、今回operatorは構築していない。既存元順LD2・partial S2 δ=.1/s4/h=1/40/r16/K0/m0..12、own-DF oracle・状態準備除外も文脈のみで、PF/RTE/trajectory/回路を評価しない。κ補正式の診断には時間・PF splitを適用しない。

Stage A保存input SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` と、元P-H4 snapshot SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` を固定した。両方のh/g/scalarだけをwhitelist loadし、dtype/shape/finiteと要素同一性がPASSした。state、energy、J/U、trajectory、QPYは新decodeしていない。

準備済みmodule／runnerを無変更で使用した。Python3.11.1／NumPy1.26.4／FE_TONEAREST、既存 `c1_stage1.corrected_kappa`、`operation_lengths/allowance`、`Calls`を再利用。歴史的な`UNAPPROVED`という契約metadataは変更せず、今回のチャット実行許可に基づいて一度実行した。承認証明・承認原文保存・grantは作成していない。

## 実行gateとcall実績

| 順序 | logical call | 状態 |
|---:|---|---|
| 1 | input_load | COMPLETE：二つの保存入力のh/g/scalar限定 |
| 2 | input_identity | COMPLETE／PASS：元packetとの値同一性、schema、finite |
| 3 | kappa_generation | COMPLETE／PASS：変更のない元g補正式とTの整合 |
| 4 | binary64_diagnostics | COMPLETE：h/T/κのnorm、exact predicate、Stage 1比較 |
| 5 | exact_dyadic_diagnostics | COMPLETE／PASS：厳密分解恒等式とH/A分解 |
| 6 | classification | COMPLETE：観測の切分け、未資格/null保持 |

旧F2の残3 callsは再利用していない。新fit/optimizer/CAR reference/exp/build/compile/sampling/PF/RTE/CA/shots/総資源・C1再解析・F3は0。科学的計算はこの登録6 unitsで終了した。以後は保存JSONの転記・provenance照合・結果公開だけを行う。

## Binary64のHermiticity残差

以下は[今回のbinary64.json](../../artifacts/research_c_pr_aware_df_kappa_diagnostic/2026-10-27/diagnostic-run01/binary64.json)の保存値で、ΔM=M−M†。3行ともfloat64、4×4、finite。Normは数値診断であり、validated operator certificateではない。

| M | max-entry | Frobenius norm | spectral2（数値） | exact Hermitian |
|---|---:|---:|---:|---|
| h | 1.1102230246251565e-16 | 2.2511913350766406e-16 | 1.5859120326984456e-16 | NO |
| T=NumPy sum_r g_prrq | 2.2898349882893854e-16 | 4.012944507652734e-16 | 2.768541423506225e-16 | NO |
| κ=h−T/2 | 5.198544862934386e-17 | 8.660473972585522e-17 | 6.070658533574577e-17 | NO |

κの絶対scaleはmax-entry `2.2648930431337604`、Frobenius `3.7256513070963813`。`||κA||F/||κ||F = 1.1622765066727537e-17`。非対称なindexは原本へ全件保存し、κは対称化していない。

κに限って登録済み`binary64_operation_scale_guard8_v1`を再現した結果：

- error：`8.660473972585522e-17`
- allowance：`3.518379690624931e-11`
- **Stage 1数値条件：PASS**
- **F2 exact predicate：不成立**

allowanceは絶対条件であり、h/Tへは適用しない。今回のκ residualとallowanceの表示値は旧Stage 1記録と同じだが、それだけで旧export κ全要素のbit同一性を認定しない。小さい数値残差を理由にexact条件やtargetを事後変更しない。

## Exact dyadic分解と原因の切分け

[exact_dyadic.json](../../artifacts/research_c_pr_aware_df_kappa_diagnostic/2026-10-27/diagnostic-run01/exact_dyadic.json)は保存binary64の実虚部を有理数として解釈した結果。元の連続値の積分やSCF精度を推定したものではない。

T*を保存gの厳密縮約、T̂をNumPy返却値、K*=h−T*/2、K̂をcorrected_kappa返却値とすると、E_T=T̂−T*、E_K=K̂−(h−T̂/2)について

`ΔK̂=Δh−ΔT*/2−ΔE_T/2+ΔE_K`

が有理数で完全一致した。E_Kは1/2倍と減算の丸めをまとめて表す。Frobenius²・entry-L1上界・非零index・exact entriesを原本に保存した。

| 項目 | 保存判定 | 解釈 |
|---|---|---|
| A：保存hのΔh | 非零 | 入力hの段階で既に非対称性がある |
| B：保存gの厳密縮約ΔT* | 非零 | NumPy縮約前にも、保存gが定めるT*は非Hermitian |
| ΔK* | 非零 | 厳密に補正式を計算しても、今回の保存入力からexact Hermitian κにはならない |
| ΔE_T | 非零 | 縮約の丸めにも反Hermitian寄与がある |
| ΔE_K | 非零 | 1/2倍・減算の丸めにも反Hermitian寄与がある |
| C：演算だけが対称性を破った | false | 入力からの継承成分もあり、純粋な演算由来ケースではない |
| D：SOURCE/input/convention不整合 | 観測なし | 固定hash／SOURCE来歴と元h/g/scalar同一性の確認範囲 |
| E：元積分生成／内部reduction treeへの原因帰属 | UNRESOLVED | SCF、MO変換、積分処理のどの段階で生じたかは評価していない |

「保存gの厳密縮約が非Hermitian」は、全gのpermutation symmetryを新たに網羅検査したとの主張ではない。A/Bの寄与と丸め寄与は符号を持ち、単純なnorm加算を因果寄与率として扱わない。今回の結果だけで元h/gの生成実装に科学的な誤りがあるとは断定しない。

数学的κH=(K̂+K̂†)/2、κA=(K̂−K̂†)/2はexact診断recordだけに保持した。κAのentryの実虚絶対値和による保存spectral上界は

`4218292557764739 / 40564819207303340847894502572032`。

これは4×4 κAだけに対する有理数上界であり、CAR head・完全trajectory・raw回路との差の上界ではない。κHをbinary64へ再丸めして入力やF2へ渡していない。κ bridgeやU→polar Q bridgeの数値評価は行っていない。

## 資源・テスト・保全

実行前にaffinity32 CPUs、MemAvailable `55240732 kB`、CPU/memory pressureの直近値0を確認した。cgroup quotaは現在のmountでは取得できずnullで、無制限と認定した値ではない。n=4の逐次依存する診断であり、NumPyベクトル化と小さい厳密有理数縮約を使用した。CPU/RAM/worker/wallの人工cap、共有venv/cache/OS/他job変更はない。

[resources.json](../../artifacts/research_c_pr_aware_df_kappa_diagnostic/2026-10-27/diagnostic-run01/resources.json)：runner wall `0.06288424972444773 s`、parent peak RSS `41532 KiB`。heavy workerやbackground jobは起動していない。準備時125 tests＋29 subtests PASSをSOURCE同一性とともに再利用し、今回はSOURCE変更がないため同じ人工テストを重複実行していない。

実行後にSOURCE138、入力2、契約、旧引用証拠、既存preservation inventoryを再hash照合した。原本9ファイルの各SHA256をpostflightへ固定。旧F2/B1/F1/C1、旧SOURCE/raw/grant/ledger/marker/STOPは不変。旧F2 raw/CARは未取得/nullのままである。今回の診断のPASSは、その未実行gateのPASSではない。

## 次の最小判断とmandatory STOP

今回の限定診断は完了した。**ユーザー／研究C**が、literal非unitary κを維持するK0か、数学的exact κHを新理想target候補とし独立model bridgeを付けるK1かを判断する。両者は異なるtargetであり、今回どちらも新規採択していない。

**Codex**の次の作業は、方針採択と具体的な依頼後に、新targetの表現、norm増大・誤差伝播、U bridgeとの二重計上防止、必要な人工検証と新しいF2契約を準備すること。新F2科学実行にはさらにその固定範囲の明示許可が必要。追加の元積分生成診断や重要科学レビューも今回から自動開始しない。

結果・SOURCE・call journal・STOPを公開し、remote SHA/Git tree/SOURCE/input/成果物照合後に**mandatory STOP**。旧F2再起動、残3-call流用、retry/resume、F3、sampling、C1再解析、shots/総資源へ進まない。
