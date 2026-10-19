# H6 full m5 CT01：sine一軸の実検証完了

2026-10-19。**元compilerの未実施sine一軸が新raw基準を通過し、正常終了した。保存cosineの別受入と合わせ、対象H6 full m5 p19/q2の両軸receiptが揃った。** 新compile一回・compiled状態作用一回。63項目の終了監査後、結果をlocal commitして**mandatory STOP**。公開pushなし。

[summary](../../artifacts/resource_applicability/track_a_h6_m5_ct01_sine_run01/2026-10-19/summary.json)、[raw検査](../../artifacts/resource_applicability/track_a_h6_m5_ct01_sine_run01/2026-10-19/raw_gates.json)、[軸別単価](../../artifacts/resource_applicability/track_a_h6_m5_ct01_sine_run01/2026-10-19/axis_cost_receipts.csv)、[終了監査](../../artifacts/resource_applicability/track_a_h6_m5_ct01_sine_run01/2026-10-19/end_audit.json)。

## 固定対象・実行承認

H6直鎖・隣接1 Å・STO-3G、保存tol-only1e-8/observed DF19、full PF p19/q2、T=.8、P=0、`4th(m5_best)` / `PF_M5_BEST_20D64C2_BINARY64_V1`。11段signed S2合成、symmetric-directional control、scalar一回、full Hadamard cosine/sine、13qubit/ancilla12、保存入力状態一つを対象とした。

SOURCEは`885bbd7199c9ada26704b7db75b99767734e92e7`、branchは`track-a-h6-m5-ct01-sine-run01-20261019`。利用者の「次の作業に進んで大丈夫」を、直前に提示した未実施sine一軸の一回実行への承認として結合した。[準備](track_a_h6_m5_ct01_sine_readiness_20261019.md)、[起動・新承認](track_a_h6_m5_ct01_sine_run01_start_20261019.md)。旧DRAFT・無効grant・prepare-only receiptは上書きしていない。

承認plan SHA256は`32849ff8539758dff5ed1502168cff1e537f62eae10de929cfae1d8ebf1d0928`、承認config SHA256は`a8542db4372bf626d1d893aa2fc0ea1f07d7a2a0bc4f74f4afb602e36f6afbf9`。CT01採択文書は`94d18a979cdc1a390fa7e2eb3ae2f18f700b47d5`。

Qiskit1.3.0・rz/sx/x/cx・元preset opt1・seed17・num_processes=1、NumPy1.26.4/Python3.12.3、provider SOURCEc31ef424…を維持した。compiler設定hashは`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。代数的1qubit縮減・再合成無効のcompilerへは変更していない。

## 数値検査と軸別単価

| 検査 | 保存・実行区分 | raw差 | 基準 | 判定 |
| --- | --- | ---: | --- | --- |
| cosine compiled vector | 元recoveryの保存値を再利用 | 1.504043707445665e-9 | finiteかつ1e-7未満 | CT01で別受入PASS |
| sine terminal変換 vs 保存imag | 新規・単一ancilla変換 | 7.249756350802272e-14 | finiteかつ1e-9未満 | PASS |
| sine compiled vector | 新規compiled一作用 | 1.493814218810128e-9 | finiteかつ1e-7未満 | PASS |

状態ベクトルの2-norm raw差をそのまま使い、phase alignment・事後位相補正・再normalizationは行っていない。compile vector以外のnative/occupation/Hadamard等の基準は1e-9未満のままである。

| 軸 | 取得・受入 | compiled RZ／軸 |
| --- | --- | ---: |
| cosine | 元recovery SOURCEf743a882…の保存compileをCT01で別受入 | 831,662 |
| sine | 今回SOURCE885bbd71…の新compile・検証 | 831,662 |

単価は実測receiptであり、model推定ではない。shotを掛けた総費用の再会計やPRとの再ランキングは今回実施していない。

元cosineの1e-9未満基準に対するFAIL・STOP・quarantine receiptは保持した。今回の受入で旧FAILをPASSへ書き換えていない。sineのraw差も1e-9を超えており、両軸の今回の受入根拠は採択済みCT01の1e-7未満である。これを元1e-9基準での合格として扱わない。

## 再利用と実行証拠

元native evolution・preparation・scalar、occupation照合、cosine Hadamard等の保存5チェックPASS、保存cosine wrapper/初期状態/未compileベクトルをSHA結合して再利用した。cosineの再compile・再作用、native/oracle/referenceの再生成は0。

sine wrapperは保存cosineの末尾H直前へSdgを一つ挿入して作成した。期待ベクトルは保存cosine未compileベクトルにancillaのH Sdg Hだけを作用させて得た。元provider builderとの一致は準備時の人工回路で確認済みで、今回も保存signalのimagとの照合を実行した。実H6の全未compile sine回路を再生する処理は省き、その期待ベクトルと新compiled回路の一作用を比較した。

[実行stage](../../artifacts/resource_applicability/track_a_h6_m5_ct01_sine_run01/2026-10-19/process_metadata.json)は、保存入力復元→compile→compiled状態作用の固定順序。各stageを先に一回予約し、grant/marker/ledgerによって再入場・retryを拒否した。

外部wallは1879.902621秒（約31分20秒）、終了receiptのmtimeは2026-10-19 **23:18:38 JST**。launcher native exit0・worker exit0、launcher/workerとも回収済み、自己signal0。外部wait4 peak RSSは22,557,356 KiBだった。終了receipt時刻はfile metadataの観測であり、追加の科学計算で得た値ではない。[終了receipt](../../artifacts/resource_applicability/track_a_h6_m5_ct01_sine_run01/2026-10-19/external_exit_witness.json)。

今回の一つのsine回路にはcompile→作用の依存があるためproductive worker一つと独立監視を用いた。末尾変換はベクトル化し、保存cosineと事前検査を再利用して重複計算を省いた。元compiler条件を維持し、並列数を増やすための追加候補は作っていない。host/上位cgroupのPSIや他worktree更新だけで停止する仕組みは使用していない。

## 終了監査・保全と解釈の限界

終了監査63項目は全PASS。SOURCE/input/runtime全2025file、13入力role、承認configとplan、CT01文書、保存cosine受入、3段階台帳、結果4 artifactのサイズ/SHA、数値gate・単価receiptの一致、全exit/reapを照合した。回路や配列を復元しないmetadata監査で、新compile・状態作用・数値再解析は追加していない。[SOURCE結合](../../artifacts/resource_applicability/track_a_h6_m5_ct01_sine_run01/2026-10-19/source_binding.json)、[native報告](../../artifacts/resource_applicability/track_a_h6_m5_ct01_sine_run01/2026-10-19/native_report.json)。

旧SOURCE・結果・grant・ledger・marker・STOPと準備成果等2816fileは作業前後のSHA不変。元pilotの58受理軸と29/30 task PASS・1 STOPの履歴も不変。元の全runを遡及的に正常終了へ変更せず、今回のcosine別受入とsine新受入を独立した二軸receiptとして結合する。[保全](../../artifacts/resource_applicability/track_a_h6_m5_ct01_sine_run01/2026-10-19/preservation.json)。

準備時の人工13＋回帰15件PASSは固定SOURCEの証拠として再利用した。作者自身の検査であり、独立レビューではない。private QPY/NPZ・有効grant/configは公開資料へ含めていない。

今回確認できたのは、この保存H6状態・固定full m5一構成のsineと保存cosineについて、元compilerの回路がCT01の技術基準を満たすことである。全状態・全候補・厳密bias・母平均費用差の保証ではない。追加sampling・SCF/DF/state/reference・signal生成・追加候補・費用再会計・再ランキング・refit・自動retryは0。m5_bestの主比較採用AD03は保留し、他系列の科学契約や進行中作業へ新しい結果を自動適用しない。

## 次の担当

**ユーザー**：今回揃ったtechnical pilotの証拠を確認し、m5を用いた後続比較・会計の範囲を判断する。今回の結果だけで追加計算を起動しない。

**Codex（別途承認後）**：承認された最小範囲について、既存58軸と今回の独立二軸receiptの来歴を保持して計画する。全候補への拡張・費用会計・主比較採用へ自動進行しない。

報告書・転記・リンク・旧履歴・非公開情報を検査した[構造検査60項目](../../artifacts/resource_applicability/track_a_h6_m5_ct01_sine_run01/2026-10-19/structure_validation.json)も全PASS。

local commit後**MANDATORY STOP**。公開pushなし。
