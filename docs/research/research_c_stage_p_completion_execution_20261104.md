# 研究C：Stage P未完了項目の15-call後続実行結果

2026-11-04 JST。**独立した後続契約を一度実行し、15/15 COMPLETE、FAIL・retry・resume 0、worker終了code 0。** Compiled actual inverseの全512列、Head/X/Y shell、単一occurrenceのraw complex signal包囲を取得した。旧41-call runの30 attempts／29 COMPLETE／未完了1という記録は変更せず、旧残り11 unitsも転用していない。

## SOURCE・対象・実行前確認

- 科学SOURCE：[2c454ed089fd6995f22c87d0a34886e1f680f3a1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/2c454ed089fd6995f22c87d0a34886e1f680f3a1)、tree `4975adb1e8f3726c4d9db56842ef554e1f6cad37`。
- Branch：`codex/research-c-stage-p-completion-20261104`。
- [固定15-call契約](research_c_stage_p_completion_contract_20261104.json) SHA256 `b5b2b1bafad6c2833a8448f84f16efd725897355e35f1addeab0434bf93ed89d`。
- [SOURCE manifest](../../artifacts/research_c_stage_p_completion/2026-11-04/technical/SOURCE.json) SHA256 `5e99ddebb232bfd2d0d71208f1424b2a25ca37f5fa76ecc803b7eb40d91a28e2`。SOURCE165 file、保存入力69件。
- [準備・人工28／関連96 PASS](research_c_stage_p_completion_preparation_20261104.md)。旧消費済みH/Pへfreshを要求する歴史的coldテスト2件は除外し、新cold gateを使用した。これはローカル検証であり、CIや外部独立再現ではない。
- Fresh出力：[run01](../../artifacts/research_c_stage_p_completion/2026-11-04/run01)、[launch01](../../artifacts/research_c_stage_p_completion/2026-11-04/launch01)。

対象は保存直鎖H4・1.00 Å/STO-3G、number CDF8／元順LD2、partial ordered S2/s1/r32/K0、h=δ=.1、Nα=Nβ=2／物理sector36。今回のkernel／inverse certificateはcontrolled全512入力列・全出力を対象とする。Wordは`CDF_proposed:F6:0`と`CDF_proposed:F7:6,7`の交互16回だけ。Endpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、κ・scalar・frame順・既存P0、own-DF ground oracle／状態準備除外、共通CA/RPE政策は不変。

[実行preflight](../../artifacts/research_c_stage_p_completion/2026-11-04/execution_preflight.json)はPASS。公開SHA/tree・本文・clean HEAD、SOURCE/input/package、FE_TONEAREST、fresh不在と空き資源を照合した。Python3.11.1／NumPy1.26.4／SciPy1.14.1／Qiskit1.3.0／python-flint0.8.0・FLINT3.3.1。今回の明示的な実行指示の範囲内で、SOURCE固定・人工検証から科学実行まで進め、承認証明・承認原文保存・新grantを作成していない。

## 再利用と15-call実績

[旧結果5e1450b1](research_c_stage_p_literal_execution_20261104.md)のC→R／R→D／C→D／raw actual inverseは再計算しなかった。旧科学SOURCEはf658f72d5、今回の追加SOURCEは2c454ed08と区別する。保存records／各digest／word／angle／target／phase／512列の完了証拠を照合した。Head bridge26-call、Stage R、fit／optimizer、PF/RTE、sampling、build／compile、shotsは新規0。

[Journal](../../artifacts/research_c_stage_p_completion/2026-11-04/run01/science_calls.jsonl)は15 attempts／15 COMPLETE／FAIL0。[STOP](../../artifacts/research_c_stage_p_completion/2026-11-04/run01/STOP.json)はretry/resume0、mandatory STOP、Stage C未実行を記録する。

| 区分 | COMPLETE |
|---|---:|
| 保存証拠／records／identity gate | 3 |
| compiled actual inverse・全512列 | 1 |
| 保存shell QPY読込 | 6 |
| shell certificate group | 3 |
| 単一occurrence signal | 1 |
| summary | 1 |
| **合計** | **15** |

## Compiled actual inverseと保存forward包囲

正本は各JSONのexact rational endpoint。以下の表は表示近似である。全列column2上界からFrobenius上界を求め、operator2 normを包囲した。Finite probeからの外挿、phase fit、Gaussian途中gateのsector射影は使用していない。

| 対象 | operator2誤差上界・表示近似 |
|---|---:|
| 保存C→R（再利用） | 1.585733058571e-11 |
| 保存R→literal D（再利用） | 3.957579638642e-14 |
| 保存C→literal D（二項の和・再利用） | 1.589690638209e-11 |
| **actual compiled inverse→exact C†** | **5.542118759907e-15** |
| actual compiled inverse→D†：新inverse項＋保存C→D | 1.590244850085e-11 |

[compiled_actual_inverse.json](../../artifacts/research_c_stage_p_completion/2026-11-04/run01/compiled_actual_inverse.json)は同じ実現compiled回路のactual inverse recordを、exact forward-record adjointと比較したもの。新しいwordのinverseを引き直していない。Arb192、全512列、相対位相・global phaseを保持し、評価roundingも包囲する。Nonzero上界を0へ置換せず、非unitary literal Dの逆行列との同値性は主張しない。

[columns.jsonl](../../artifacts/research_c_stage_p_completion/2026-11-04/run01/compiled_adjoint_columns/columns.jsonl)は各完了列を保存し、[progress](../../artifacts/research_c_stage_p_completion/2026-11-04/run01/compiled_adjoint_columns/progress.json)は512/512を記録する。最終集計は元のcolumn番号順で、既存Arb primitiveと算術順序は不変。保存列をrestartに使う機能はない。

## Head/X/Y shellと単一occurrence信号

ここでのHead shellはancillaのHadamard preparationであり、full-CAR／sectorの分子head bridgeとは別である。保存raw／compiled QPY6件を読み、gateが厳密にancilla-onlyであることを検査して2×2へ縮約した。登録末尾measurement以外の除去やphase fitはない。埋込み9-qubit operatorにも同じ誤差上界を使用できる。

| shell | raw→ideal | compiled→raw | compiled→ideal |
|---|---:|---:|---:|
| head | 1.274483261863e-57 | 7.499399432609e-17 | 7.499399432609e-17 |
| X | 1.274483261863e-57 | 7.499399432609e-17 | 7.499399432609e-17 |
| Y | 1.274483261863e-57 | 4.329780281177e-17 | 4.329780281177e-17 |

正本は[head](../../artifacts/research_c_stage_p_completion/2026-11-04/run01/head_shell.json)／[X](../../artifacts/research_c_stage_p_completion/2026-11-04/run01/X_shell.json)／[Y](../../artifacts/research_c_stage_p_completion/2026-11-04/run01/Y_shell.json)。信号会計には各compiled→ideal項を一回だけ使い、他二項は診断である。

[finite_signal.json](../../artifacts/research_c_stage_p_completion/2026-11-04/run01/finite_signal.json)で、保存per-word誤差eとStage Rのexact norm aを使い、作用差≤e_Q+e+a e_H、各観測量差≤(1+a)倍で包囲した。Normを1へ、population range M=1+aを2へ丸めず、literal targetへ根拠のないcap2を適用しない。

| 固定word・単一occurrenceのraw信号 | 誤差上界・表示近似 |
|---|---:|
| X | 3.179411274016e-11 |
| Y | 3.179404934778e-11 |
| **complex signal** | **4.496362061560e-11** |

これらは同じliteral Dと数学的compiled gateの有限word信号差の包囲である。期待bias、IID sampleの推定値、13roundの校正やactual CAではない。Mの適用に必要なall-q compiler-domain前提を新しく証明したものでもない。分子Head bridgeや旧B2二項をこのper-word／shell会計へ重複加算していない。

## 実行状態・資源・保全

[Supervisor exit](../../artifacts/research_c_stage_p_completion/2026-11-04/launch01/exit_status.json)はreturncode0／terminating signal null。[worker log](../../artifacts/research_c_stage_p_completion/2026-11-04/launch01/worker.log)とheartbeatを保存した。旧プロセス消失の原因は未確定のままであり、今回の完走を原因特定や全障害防止の証明とは扱わない。

実資源から32 workerを選択し、計算wallは約1667.16秒（約27.79分）。親peak RSSは135392 KiB、reaped child CPU合計は約51607.27秒。Worker合計のpeak RSSは未測定で、親RSSを総RAMと扱わない。開始時空きRAM約52 GiB、実行中のmemory pressureを確認した。人工的なCPU/RAM/worker/wall cap、共有環境・他ユーザーjobの操作はない。速度向上率は主張していない。

[技術検査](../../artifacts/research_c_stage_p_completion/2026-11-04/execution_verification.json)はPASS。15-call journal、全512列の重複なしcoverage、progressとcertificateの一致、phase／Arb192、保存forward上界とのexact和、shell2×2、exact a/M binding、SOURCE／契約コピーと保護対象236ファイルのhash一致を確認した。新run／launchの16 file、3,189,822 bytesを保存した。旧raw・4 QPY・旧SOURCE／契約／grant／ledger／marker／STOPは変更していない。新QPY build／compileは0。

## 結論と次担当

旧／新の固定証拠を合わせ、対象1-wordのper-word operator包囲、両actual inverse、Head/X/Y shell、単一occurrence raw complex signal包囲が揃った。**独立15-callの未完了項目評価は完了した。旧41-call runを正常完了へ書き換えていない。**

全law一様synthesis、全合法wordのcompiler成功、期待bias校正、Stage C、RPE shots、actual CA、総資源優位は未資格・未評価のまま。次担当はユーザー／研究Cで、この有限word証拠を踏まえて次の校正・保証範囲を判断する。Codexは追加sampling／全law／CA等へ自動進行しない。

結果・SOURCE・契約コピー・journal・列別raw・終了状態をcommit／公開し、remote SHA/tree・本文・主要blobを照合する。**公開照合後mandatory STOP。**
