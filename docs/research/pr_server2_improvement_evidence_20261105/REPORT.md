# SERVER 2：PRアルゴリズム改善・既存証拠の追加調査

2026-11-05 JST。対象は `HIROMU1015/Partially-Randomized-Trotter`。**文書・静的SOURCE・保存metadataの調査を完了。新しい科学的検証、数値再解析、科学レビュー、研究方針採択は行っていない。公開照合後mandatory STOP。**

成果物は [追加証拠CSV](SERVER2_IMPROVEMENT_EVIDENCE.csv)、[資料・同一性の未解決](SERVER2_EVIDENCE_GAPS.md)、[SOURCE再利用map](SERVER2_REUSE_MAP.md) と本書の4ファイル。

## 1. 固定点と前回からの追加範囲

| 用途 | 固定commit | 今回の扱い |
|---|---|---|
| SERVER 2 Phase 1 | `8834de0cf938b13490c08e05006e3c566fed7c7a` | 4研究root・360保存先参照・8,761観測行の元記録を維持。再生成しない |
| Phase 2 integration | `604f0235840feb18d368f99ea0828f370ce70fc8` | 元record/run/cohortとCOPY_OF/REUSES等の読み分けを再利用。統合台帳を変更しない |
| 70研究カード・SOURCE索引 | `a9bdcf1d44061daceffe896637e2c92fcb3353b4` | 選定33系列の元study ID・条件・記述statusを継承。127参照の固定blobが旧索引と一致 |
| 今回の文書閲覧・作業起点 | `5b0971a2d9fc6ef474008d26d14af85e324b4b24` | 既存SERVER 2差分報告を含む保存snapshot。元研究の実行SOURCEとは区別 |

Phase 1付録の改善研究カード対応と未展開の記載を利用し、Track B RB-01〜RB-18、関連PR-2/3、回路共有、P-A〜D・R3、FR、表現対照、C4を追加整理した。さらにPhase 2C/Dの設計→実装→B binding→F1旧停止/修正版→F2資格停止/資源旧停止/修正版、PS48/一時終了、非prefix未検証、G6独立人工受入/実API512、BF1旧停止/保存復旧を別行にした。研究Bと研究Cは関連する別系列の実績・後続記録への接続として含めた。

**CSVの52行は33系列の家族単位記録と19段階・履歴記録であり、52の独立実験ではない。** 元8,761行を再作成・置換・合算していない。PR-01のPR2_M1/M2とOPT-01のM06-Fは前回登録済みbaselineへの再接続で、新実験の追加ではない。個々の旧全gridやrawの網羅転記ではなく、前回対象外だった改善系列を証拠階層別に追加する調査である。研究B/B2-DとTrack B、legacy DF12とtol-only DF10/19/28、Pauli prefixとDF fragment prefixを統合しない。

## 2. 実際の確認範囲

親projects・ユーザー共通・新cloneのAGENTS、既知4研究rootの適用指示、正式共通運用指示2026-10-25版（§8.1/§8.4/§10）を読取確認した。PROJECT_MAP、研究概要、11月1日の研究全体地図・claim/gap表、RESEARCH_RESULTS_INDEX、指定landscape/Phase2A/2C/2D、algorithm_codesignを入口にした。旧snapshotの「次段階」は実行指示として扱わない。

Phase 1と既存TSVが明記する4研究rootについて、今回も本人UID・read/search権限・origin対象repositoryを確認した。新規出力は本人所有の独立cloneと専用branch内だけ。4rootを内部pathで公表せず、Phase 1のS2-ROOT識別を保つ。既知private保全索引もmetadataとして参照したが、360参照を再走査せず、未知保存先・他ユーザー・他hostへ探索を広げていない。旧SERVER_A/B/ENV-LOCAL-02を現在のSERVER 1/2へ新対応付けしない。

固定Git blobの読解と、Python標準ライブラリによるCSV/JSON文字列読取・AST静的解析・参照照合だけを用いた。科学moduleをimportせず、test/runnerを起動しなかった。数値tokenは `parse_int=str, parse_float=str` を使い、平均・SE・比・順位・費用を新しく算出していない。既存literal転記方式とpath/selector確認を再利用した。gzip/NPZ/QPYをdecodeせず、RNG replay・simulation・state/Hamiltonian生成・回路build/compileは0件。sub-agentも使用していない。

利用可能affinity・空きmemoryを読取確認し、CPU/RAM/worker上限を追加しなかった。小さいmetadata処理であり科学workerを投入せず、OS/scheduler/共有venv/module/cache・他jobに変更を行っていない。通常sandboxの起動失敗には通常ユーザー権限の実行経路を用い、OS設定を変更していない。

## 3. 検証階層ごとの確認結果

| 改善系列 | 保存証拠が示す到達点 | その証拠で未確立の範囲 |
|---|---|---|
| 非prefix分割 | Phase2A RQ-02とPR-2契約に具体差の条件、既存prefixの比較がある | 非prefixの新しい具体的生成/分割則と同精度優位は未検証。prefix勝者を非prefix成功と呼ばない |
| G6 Return Aggregation / General Full / Closed P5 | 条件付き数理構成、実装、synthetic生成/finite-bitと既存監査がある | G6定理から全分子native資源利益・独立新規性へ飛躍しない |
| G7〜G10 / R1 / Parent allocation | 固定synthetic provider・finite target・confidence下のnative/十分N/資源地図が存在。G9 matched CTS比較も既済 | 全wrapper合成・分子/DF移送・一般IS最適性・全古典費用の優位は別。T利点とCTSのCX利点を保持 |
| M1-S | H4 DF12の保存signal、限定理論・整数witness、部分独立計算経路がある | 旧full-H/HD未保存、overall UNDETERMINED。後続gamma/pi3/Stage Sは保存信号を再利用し、多round RPE/native全費用は未確立 |
| C1 G6–DF | H4のexact入力、選択6eventのoperator/adjoint/phase/X/Y、Ordinary/Full/P5R各1024pathのlogical pilotがある | 全324label/全pathのstrict native、D半stepと準備/読出しを含むfull-wrapper費用、十分N、公平な全方式CPU比較は未取得。一時凍結を維持 |
| Phase2D F1修正版 | 固定4mode合成fixtureの4arm・registered correctness・64wrapper/compile検査を完了 | interval未資格のSS fallbackは最適選択でない。分子・認証N・新規性の結果でない |
| Phase2D F2修正版 | 同合成fixture、別K0/q1条件でfull-wrapper全件operator/signal/meanと1-shot実資源を取得 | gamma/統計資格・N・N加重費用・physical同精度総資源は対象外。資格停止契約は解除されていない |
| Gaussian singleton PS48 | 同operator/law/保存targetで48wrapperの同値性を確認 | 一律適用のRZ/CX削減は支持されず、局所改善と一時終了を保持。全F2 law重みや独立選択規則の成功でない |
| Gaussian積 / interstep相関 / C4 | 静的な既知法対応と具体差未成立を記録 | 性能で敗れた実験、または新しい生成法の成功としない |
| 研究B / 研究C | 別toyのD内部/outer/finite-RTE/nativeと、H4 PR-aware DF・条件付きN/Gの保存実績がある | 研究Bは資格/rare費用未決で一時保留。研究CはPR内部改善と強決定論対照への不支持を分離。小step後続は自己目的更新でPF/native費用未評価 |

CSVの7階層列は独立に読む。`UNKNOWN`は不存在やFAILを意味しない。`NOT_EXECUTED`/`NOT_ESTABLISHED`は記載対象の範囲内であり、他系列へ一般化しない。`original_status`が既存catalogの説明文の場合は `status_source` に明記した。元結果の厳密statusは固定参照へ戻れる。SOURCE実装の存在と科学的correctnessは別である。

## 4. 失敗と後続更新の保持

| 旧記録 | 後続 | 今回の記録方法 |
|---|---|---|
| BF1 `09c9e895…` serialization停止 | `6d2645a0…` saved-only R0、限定closure | CSV051/052。元INCONCLUSIVE・未完了bridgeを保持。新signal/独立cohortを加算しない |
| G9 v1 API停止 / G10 v1-v2 resource・I/O停止 | G9 v2 / G10 v3完了 | RB-09/10の原文・version historyへ接続。旧prefixを無条件に採用しない |
| M1-S Gate0 v4停止 | v5・部分独立照合・finite v2・gamma/pi3/Stage S | RB-15の役割別SOURCEを保持。後続を新しい物理入力の独立反復と数えない |
| C1 partition→state hash→reference zero-sign→P5 worker停止 | 新R1→R2→別P5R完走 | RB-17の新旧input hashと元報告を保持。旧P5の処理済trial/終了原因UNKNOWNを埋めない |
| F1 `594bab9e…` metadata TypeError | `b0dad8cc…` / SOURCE `8192530b…` | CSV037→038。旧実transpile0と後続registered correctness完了を別記 |
| F2 `06ce7e5b…` compiled operator不一致 | `dc7378ab…` / SOURCE `9468c708…` | CSV039→040。位相保持compile接続修正後の別run。旧3FAIL/STOPを保持 |
| F2 statistical qualification停止 | 別per-shot-only契約の資源計測 | CSV041と039/040。資源PASSをN資格の解除にしない |

修正SOURCEの実行は過去の保存事実であり、今回修正・再実行したものではない。上表は履歴対応で、新しいPASS/FAIL・科学GOを出していない。

## 5. LOCALへ渡す識別情報

CSVは `original_study_id`（RB等）、元条件、明示SOURCE、SOURCE役割、固定結果commit/path/blob、run/cohort、科学入力hash、履歴、独立性、未検証事項を保持する。JSONを埋め込んだ列は通常のCSV parserで読む。行順・仮IDから科学的同一性を判断しない。

- 同一保存証拠候補は、**result commit/path/blob/selector**から照合する。異なるpathの同じblob、同数値、同じbranch名だけで結合しない。文書・JSON root単位の行を元実験selectorへ展開する場合はLOCAL側の既存規則に従う。
- `static_code_versions`は固定文書で確認したmodule版で、科学runのSOURCEへ自動置換しない。`recorded_source_components`は段階別役割を保持する。SOURCEが単一に確定しない行はUNKNOWNのまま、原表へ戻る。
- F1/F2は同じB physical/arm入力を共有し、F1とF2のT/q/K/lawは違う。cohort・input hash・compilerを同時に見る。PS48も同じ保存targetを再利用する。
- C1の旧partition expected、新R1 partition、state-vectorとstate-metadata hashを分離する。公開hashがあることをrigorous physical-model bindingとしない。
- G6独立人工受入、512 live correctness、16,896 live限定pilotを区別する。別sessionの技術独立性を別host/別機関の物理再現へ格上げしない。

**SERVER 2だけでLOCAL/SERVER 1との同一性・独立性は確定していない。** 今回は統合台帳や科学的claimを編集していない。[既存SERVER 2差分報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5b0971a2d9fc6ef474008d26d14af85e324b4b24/docs/research/pr_server2_evidence_delta_20261105/REPORT.md)のH8長時間確認やP3などは別の追加調査として再利用し、今回の改善行と別実験として二重加算しない。

## 6. 資料の取得と保全・公開

選定33系列の127参照は本人所有の既知Git storeから読取でき、既存索引のblobと一致した。そのうち49参照のPython SOURCEはASTで定義/import構造も確認した。さらに修正版mixed/PS48/DF/C1とmetadata parserの26 SOURCE・runner・test参照を静的に確認した。これは参照件数であり、独立codebase/実験数ではない。

GitHub connectorから次の4既存資料を固定commitで取得し、応答本文とSERVER 2 Git bytesを文字列比較した。すべて一致した。その他の全参照を今回GitHubから全件再取得したとは主張しない。

| 固定commit | 取得した参照 |
|---|---|
| `b0dad8cc2f7a4cb821f53185c4a65f3a42e8f381` | `artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24-r1/F1_RESULT_SUMMARY.json` |
| `dc7378ab34015287215bcf4ce52147e46d94ecd9` | `artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24-r1/RESULT_SUMMARY.json` |
| `28cfabb1d47e0e1824bce1e154a9c289738fa2b9` | `src/trottertracks/algorithm_codesign/return_aggregation.py` |
| `185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e` | `docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md` |

取得不能・未公開・未確認は [EVIDENCE_GAPS](SERVER2_EVIDENCE_GAPS.md)に区別した。private rawの全量取得、完全性・独立backup・復元性、旧UWC原本との同一性は確認していない。原資料にある非公開保全の記載を、今回の現物再認証にしない。

本作業の書込みは新cloneの4公開成果物と私用一時metadata、新clone内のGit管理に限定した。元SOURCE/raw/result/grant/ledger/marker/STOP/archive/個別科学契約、元branch/main/既存台帳に変更操作を行っていない。全原本の再hashを実施したという保全証明ではない。独立cloneは既存object storeへ依存し、独立backupではない。

公開branchは `codex/pr-server2-improvement-evidence-20261105`。4ファイルだけを新commitへ含め、force push・branch削除・archive整理・元worktreeのcheckout/reset/clean/gc/pruneは行わない。通常git pushはSSHのhost key検証とHTTPS認証情報の不在で失敗した。共有設定や資格情報を変更せず、接続済みGitHub APIで同一の4ファイルtreeを新commit・新branchとして公開する経路を用いる。最終公開commitとremote SHA/内容照合の実測は最終チャットで渡す。CSV52行/34列、固定参照172組、引用token、4ファイル間リンク、匿名化の技術照合を完了した。作者自身の検査であり、独立した科学レビューではない。

**mandatory STOP：SERVER 2追加調査の公開照合で終了。** 次の担当は**LOCAL Codex**：この52行と固定参照をSERVER 1成果物/既存integrationへmetadataとして突き合わせる（今後の引継ぎ先であり、今回未実施）。SERVER 2 Codexの追加科学計算、実装、科学レビュー、研究方針採択は行わない。
