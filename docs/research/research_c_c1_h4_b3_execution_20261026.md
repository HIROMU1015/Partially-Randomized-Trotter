# H4 B3：spin対応XDF基底＋相対Gaussian融合の65-call限定実行

2026-10-26 JST。**固定SOURCEを変更せず一度実行し、65/65 COMPLETE、correctness FAIL／inflight／retry／resumeは0。B3のfull512 raw・P0・actual adjoint・complex X/Yはすべて数値PASS。** 新B3 XDF8の単一kernelはRZ10,885、CX6,892、total depth11,973で、保存B2 CDF8より登録した6指標すべて小さかった。同CAでの総資源順位は今回評価していない。

- 科学SOURCE：`b40fce5c62c9fff97f68073cbc02182e4bbf2b9b`、tree `adb519dcefb5d261d0e0309644d07fa646b795dd`。
- 実装SOURCE：`ae5d2fe9330108e1cfeef5080bbfe7ec74eef246`。実装・runner・testsは今回無変更。
- Branch：`codex/research-c-c1-h4-b3-execution-20261026`。独立worktree：`/tmp/prt-c1-h4-b3-execution-20261026`。
- [固定65-call契約](research_c_c1_h4_b3_science_contract_draft_20261026.json)、SHA256 `0748fc10663fd96a7147abff77da19ad70e1f50f7c65811df8df3333f43972f4`。歴史的なdraft名／metadataは変更していない。
- [SOURCE manifest](../../artifacts/research_c_c1_h4_b3/2026-10-26/technical/SOURCE.json)、SHA256 `8c2d85c25800bc76c23545f68f5ff31a6cd4fae049560fb51df7ea3273aab28a`。
- [機械可読要約](../../artifacts/research_c_c1_h4_b3/2026-10-26/execution_summary.json)／[原本run01・QPY4・operator4・journal・STOP](../../artifacts/research_c_c1_h4_b3/2026-10-26/run01/)／[metrics CSV](../../artifacts/research_c_c1_h4_b3/2026-10-26/native_metrics.csv)／[3比較CSV](../../artifacts/research_c_c1_h4_b3/2026-10-26/native_comparisons.csv)／[correctness CSV](../../artifacts/research_c_c1_h4_b3/2026-10-26/correctness.csv)。

## 対象と実行範囲

直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ=2／sector36、system full256／ancilla-last controlled full512。保存OpenFermion1.6.1 eigh元順prefix8 XDF、number表現、signed λ/G、元g補正κ、核scalar一回、frame0〜7の順を維持した。比較先CDFも保存rank8である。全frame ordered symmetric S2、s=1、h=δ=.1のcontrolled kernel一個を新規評価し、own-XDF ground oracleを使うm0 X/Yだけを検証した。PR splitは非適用、状態準備費用は除外。

CA=.00159 Ha、DF reserveCA/8=.00019875 Ha、13round m0..12、既存β／failure配分は背景条件として維持したが、この実行でCA／shots／13round総資源を評価していない。保存DF norm＋guardはCDF `7.666215916301914e-5`／XDF `9.533901455091894e-5` Haと異なる。両方の保存DF予算PASSを再利用し、新しいDF誤差計算はしていない。

元P-H4入力は公開commit `86b8b002704aced8ca239414080f53793de81658` の `artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz`。SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` の同一bytesをtask-local `.external/P-H4.npz`へ配置した。旧SOURCE／結果／STOPを上書きせずfresh run01を一度だけ使用した。チャットの明示許可に基づき実行し、承認証明・承認原文保存・新grantは作成していない。

## 再利用・実行前後の照合

[技術準備](research_c_c1_h4_b3_preparation_20261026.md)で確認した `prepare_xdf_basis(SPIN)` と `StrongBuilder(relative_fusion=True)` をそのまま使用した。spatial eighからexact interleaved spin liftを作り、隣接run境界の `R=U_following† U_previous` と解析的真空位相補正を既存実装で処理した。Gaussianは `nature_jw_exact_zero_lapack_atan2_v1`、compiler phaseは `qiskit_P0_mpfr160_declared_phase_v1`。

P0はQiskit1.3.0／opt1／seed17／basis rz,sx,x,cx／approximation_degree1／qubits_initially_zero=false、既存layout・境界規則を維持。NumPy1.26.4／SciPy1.14.1／OpenFermion1.6.1／PySCF2.7.0／Qiskit Nature0.7.2等を既存lockと照合した。共有venv／ライブラリ／共有cacheは変更せず、既存venvのPython3.11.1実体を直接起動した。

独立referenceは保存 `h4_native_fair/run01/operators/XDF_s1_reference.npz` の元raw CAR由来system targetを使用し、B3係数から再生成していない。head/X/Yのraw・compiled shell6件と保存own-XDF8 groundも再利用した。旧B0／B1／B2の回路、CDF数値配列、DF/PF/ground/referenceは再計算していない。

[preflight](../../artifacts/research_c_c1_h4_b3/2026-10-26/execution_preflight.json)／[postflight](../../artifacts/research_c_c1_h4_b3/2026-10-26/execution_postflight.json)で、SOURCE35＋直接input19と旧bindingを含むprotected88件、元入力、契約、manifest、前段raw60が不変であることを確認した。新rawは24 files／7,674,764 bytes、QPY4、full512 operator NPZ4、すべてのbytes hashを保存した。

## 全gateと数値精度

行列component allowanceは絶対1e-10、complex signalは2e-10、rtol0。global phase fitを行わず、control=0 identityとcontrol=1 evolutionの相対位相を含む全512入力×全512出力を比較した。有限probeやsector内だけの確認ではない。

| gate／観測 | 結果 | 最大差・範囲 |
|---|---|---:|
| SOURCE・入力・保存scope・環境 | PASS | protected88不変 |
| spin構造・9 G再構成／unitarity | PASS | 2.220446049250313e-15／1.1102230246251565e-15 |
| B1 recipe・frame／signed eigenvalue順 | PASS | 保存eigenvalue差0 |
| 16相対Gaussian境界・位相・S2順序 | PASS | 登録順序と解析的真空位相を保持 |
| raw／独立reference | PASS | 6.1781099539291495e-15 |
| compiled／raw | PASS | 2.67982690330494e-12 |
| compiled／reference | PASS | 2.6837375605171582e-12 |
| raw／compiled unitarity | PASS | 1.0103029524088925e-14／3.715916463420399e-13 |
| raw actual adjoint：dagger／reference／recovery | PASS | 5.45547559346042e-15／4.501806227948487e-15／6.8352004951677554e-15 |
| compiled actual adjoint：dagger／reference／recovery | PASS | 9.664053404248954e-15／2.6856303898180274e-12／3.6604073747047953e-13 |
| complex X/Y：raw／reference | PASS | 3.5566146649215975e-15 |
| complex X/Y：compiled／raw | PASS | 3.470082518625064e-13 |
| complex X/Y：compiled／reference | PASS | 3.5047630802313664e-13 |
| native metrics・登録3比較 | PASS | 下表 |

compiled／referenceのfull512 Frobenius差は `4.738879866495084e-11`。これは数値pilotの観測値であり、reference／expm／Operator／normを含む丸め誤差の厳密enclosureではない。保存B1はU配列そのものを含まないため、Uの歴史的bitwise一致は主張しない。同じSOURCE／入力／recipe／signed λ順を照合し、新しい独立full512 gateでB3の作用を確認した。

## 単一kernelのnative費用

下表のB0／B1／B2は保存scalarの引用で、新規測定はB3一件だけ。すべてH4、rank8、s1、h=.1、同precision／P0のcontrolled kernelである。state preparationとhead/readout shellはkernel費用に含めない。

| route | RZ count | RZ depth | CX count | CX depth | total depth | size |
|---|---:|---:|---:|---:|---:|---:|
| B0 CDF（保存） | 12,272 | 5,873 | 7,388 | 6,440 | 12,806 | 23,543 |
| B0 XDF（保存） | 13,944 | 6,005 | 7,756 | 6,468 | 13,113 | 26,740 |
| B1 XDF spin（保存） | 11,821 | 5,793 | 7,176 | 6,328 | 12,556 | 22,376 |
| B2 XDF relative（保存） | 11,307 | 5,564 | 7,008 | 6,220 | 12,122 | 21,706 |
| B2 CDF relative（保存） | 11,062 | 5,577 | 7,008 | 6,280 | 12,127 | 21,011 |
| **B3 XDF spin＋relative（新規）** | **10,885** | **5,524** | **6,892** | **6,192** | **11,973** | **20,606** |

登録した差はB3−比較先である。B0に対する追加の再集計・新比較callは行っていない。

| 比較先 | ΔRZ | ΔRZ depth | ΔCX | ΔCX depth | Δtotal depth | Δsize | RZ比B3/比較先 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 XDF spin | −936 | −269 | −284 | −136 | −583 | −1,770 | .9208188816512985 |
| B2 XDF relative | −422 | −40 | −116 | −28 | −149 | −1,100 | .9626779870876448 |
| **B2 CDF relative** | **−177** | **−53** | **−116** | **−88** | **−154** | **−405** | **.9839992768034713** |

この固定kernelでは、spin対応基底とrelative融合を併用したXDFの登録6指標がB2 CDFより小さくなった。RZ差は約1.60%であり、一般分子・他compiler・13roundやshots加重費用への外挿はしない。CDFとXDFのDF近似は同一でなく、共通CAに対する新しい順位を認定しない。既報約12%の13round条件付き総RZは、当時の実装と資格に限る結果として保全する。

## 科学call・実行資源

| call kind | 実績／上限 |
|---|---:|
| saved_input_load／saved_evidence_load／saved_identity | 各1／1 |
| native_model／model_gate | 各1／1 |
| gaussian_build | 9／9 |
| relative_gaussian | 16／16 |
| reference_load／ground_load | 各1／1 |
| shell_operator_load | 6／6 |
| kernel_build／fusion_gate／raw_gate／compile | 各1／1 |
| operator_decode／adjoint_build／adjoint_operator／adjoint_gate | 各2／2 |
| segment_assessment／signal_reference | 各1／1 |
| signal_prepare | 2／2 |
| signal_axis | 4／4 |
| signal_gate | 3／3 |
| metrics | 1／1 |
| cost_comparison | 3／3 |
| **合計** | **65／65** |

全65 attemptがCOMPLETE、FAIL0、inflight0、retry0、resume0。[call timing](../../artifacts/research_c_c1_h4_b3/2026-10-26/call_timing.csv)の累積時間は並列callの重複を含み、wallとの単純比較をしない。

preflight時はaffinity32 CPU、MemAvailable55,197,068 KiB、CPU／memory／I/O pressure avg10は0。cgroupの識別情報を記録したがcpu.max／memory.maxは参照不能であり、quota無制限とは認定していない。raw gateをP0より先に逐次実行し、P0／adjoint構築後の独立したcompiled／raw adjoint／compiled adjointの3 Operator decodeはworker3で並列化した。job内BLAS1でoversubscriptionを避け、共有設定は変更していない。新しい恣意的CPU／RAM／worker／wall上限は追加していない。

runner wall **33.39811956137419秒**、process CPU **68.4018669秒**、peak RSS **505,604 KiB**。speedup倍率は測定していない。人工・回帰検証は同一SOURCE／環境の**122 unique PASS**を再利用し、不要な再実行を避けた。内訳と途中fixture失敗を含む履歴は技術報告とtechnical原本に保持した。

## 限界・次の担当・mandatory STOP

今回取得したのは、登録した一つのH4 XDF8 B3 kernelの全行列数値correctnessと3件のnative比較である。厳密丸めenclosure、s4、13round合成誤差伝播、最終CA、shots、条件付き総費用、実機／状態準備込み費用、新規性、正式なC1継続／縮小／終了は未資格／未評価。`CA_achieved`／`shots`／`total_cost`／`rigorous_enclosure`のnullを維持した。

新fit／optimizer／rank探索、DF/PF／ground/reference再計算、旧route build／compile、CDF数値decode、別分子条件への拡張は0。旧H2/H4/H6・B0/B1/B2・XDF9のSOURCE／raw／grant／ledger／marker／STOP、C4／singletonの終了状態を保全した。実行中のSOURCE変更、fallback、閾値緩和は行っていない。重要な科学レビューも今回実施していない。

**次の担当はユーザー／研究C**。この限定比較を確認し、C1を区切るか、B3の同CA比較に必要な13round合成・RPE shots・費用の再資格付けを追加するかを判断する。**Codex**は具体的依頼と必要な別の明示許可を受けた範囲だけを担当する。今回の結果から次の契約・計算・レビューを自動開始しない。

**証拠をcommit・pushしremote SHA／SOURCE／入力／成果物を照合した後、mandatory STOP。run01は消費済みで再起動しない。**
