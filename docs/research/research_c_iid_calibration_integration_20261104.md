# 研究C：Stage P／Compiler／Fresh IID-grid校正基盤の技術統合

識別子：2026-11-04。作業完了日：2026-11-05 JST。
**TECHNICAL_INTEGRATION_COMPLETE_SCIENCE_NOT_READY — MANDATORY STOP**。

Branch：`codex/research-c-iid-calibration-integration-20261104`。
独立worktree：`/tmp/prt-c-iid-calibration-integration-20261104`。
これは接続実装・人工検証の報告であり、新しい実H4の結果、重要科学レビュー、実行許可を与えない。

## 起点とA/Bの取り込み

共通親はStage P完了 `c01bd2742ca88071d17db1fc3e3aec6cb9f3cb9e`。
正式共通指示2026-10-25の§8.1・§8.4・§10を適用した。

| 役割 | 固定commit / tree | 扱い |
|---|---|---|
| B：fresh標本管理 | `9098de49f1bd816dbaf810eb63815c841169236d` / `47e8a84562759dfc8def9b05f9e41136073afe36` | 新branchの起点。既存全ファイルを保持 |
| A：batch certificate・失敗率 | `ce3185513578a8329efe364c748b5ce2aacdda45` / `ba132cca7ea72019583baa5ebbe1be656fc08a4f` | 共通親との差分から未収録の追加ファイルのみbyte一致で取り込み |
| P0 adapter | `60dcaa94fd44675bc3184e409aa4ebdef7a07dbf` に由来 | A/Bで同一。B既存ファイルを再利用し重複取り込みなし |

無条件のbranch mergeやcherry-pickは行わない。Aの原SOURCE・報告・人工benchmark・未承認契約・STOPもそのまま取り込んだ。
4つの索引はBの内容を維持して今回の接続への入口だけを追加した。
既存SOURCEを更新すると固定A/B preflightを破るため、既存コードは修正せず、新しい薄い接続adapterを追加した。
これが新規実装を選んだ理由であり、sampler、P0、Arb算術、統計engineは再実装していない。

原報告： [B](research_c_fresh_iid_calibration_infrastructure_20261104.md)、
[A](research_c_stage_p_totality_calibration_feasibility_20261104.md)。

## 接続仕様とデータフロー

実装： [c_iid_calibration_integration.py](../../src/trottertracks/c_iid_calibration_integration.py)。
非実行仕様： [connection contract](research_c_iid_calibration_connection_unapproved_20261104.json)。

| 段階 | 再利用と追加検査 |
|---|---|
| Fresh入力 | Bの53-bit整数replay・K0 order→atom消費・`validate_sample`。SOURCE/candidate/law/model/word/angle/occurrence/round/axisを照合 |
| Raw/P0 | 呼出側のcallbackを一度だけ呼ぶ。人工P0テストは既存`compile_once`を使用。Raw/compiled recordの全gate/global phase digest、phase policy、compiler identity、compiledの親raw IDをbind |
| Target | 既存literal referenceのtarget全体/target ID、candidate、word ID/順序/angle、Arb192を照合。期待targetは呼出側が独立に固定 |
| Per-word | Aのbatch C→Rと既存literal R→Dを同一rawで接続。既存のC→D二脚triangleと同じ数学的targetを維持 |
| 記録 | Bの排他的保存・元標本保持・outcome schema。raw、compiled、reference、失敗したcertificate候補、完了済みの脚をpartial evidenceとして保持 |
| 統計入力 | `check_certificate`で保存されたidentity/coverage/precision/phase/triangleを再照合し、Bの`calibration_payload`→既存`calibration_inputs`へ渡す |
| 統計計算 | Bの`evaluate_calibration`から既存`threshold_tail`・`variance_sensitive`・`paired_round_upper`・`failure_accounting`へ条件付き接続 |

`bind_operator`は呼出側のrecordとwordの関連を固定する機能であり、builderがwordを正しく実装したことの証明ではない。
将来の実runnerはBの`rte_events`と既存builder/P0を使い、科学SOURCE・入力・targetの生成元まで固定する必要がある。
今回のadapterに実H4 runnerや科学実行switchを追加していない。

`sample_id`は一意。同じwordの別標本・別occurrenceは保持する。同じwordでも別SOURCE/law/angle/phaseのcertificateを無断再利用しない。
標本はcallback呼出前に保存され、戻り値欠落・割込みはINCOMPLETE、数値欠落はnull/未資格として残る。Missingを0にしない。

## 高速certificateの適用範囲

| 比較 | 評価器 | 共通unitary除去 / 上界2 |
|---|---|---|
| C→R | `c_arb_column_batch.compare` | 厳密に共通するnative unitary因子だけに適用。global/controlled相対位相を保持 |
| R→D | `c_per_word_population_range.compare_literal` | 適用しない。非unitary literal CAR targetをそのまま評価 |
| C→D | 2脚のtriangle | 適用しない。R→Dのreference不確かさを重複加算せず、非unitary差を2へclipしない |

Arb192、全入力列・全出力、外向き丸め、Frobeniusからoperator2 normへの上界を保持する。
因子除去後の列はnorm同値な残差operatorの全基底列であり、元operatorの同名列ではない。
小register人工入力のcertificateは人工scopeのまま。統計へのfull512資格は別に要求する。
今回のfull512人工入力はscalar literal blockと人工native recordから構成し、保存Stage P recordを再評価していない。
Aの約6.38倍は既存の人工benchmarkの観測に限る。今回の実H4の速度や未知word性能を主張しない。

## Compiler失敗とSTOP

B既存のSUCCESS、RAW_BUILD_FAILED、P0_COMPILE_FAILED、NATIVE_INELIGIBLE、PER_WORD_CERTIFICATE_FAILED、NUMERICAL_UNEVALUABLE、INCOMPLETEを使用する。
GENERATEDはBの中間状態として維持し、別のoutcome schemaは追加していない。
`process_sample`は最初の失敗でBのSTOPを保存する。再抽出・retry/resume・fallback・次標本への自動継続はない。
割込み時も標本・partial/outcome・STOPを保持し、割込みを再送出する。

`compiler_failure_summary`はcoherent occurrenceごとのraw/compile状態をAの既存trial語彙へ変換する。
Pipeline失敗とraw成功条件付きcompiler失敗を分離し、certificate/数値失敗だけでcompiler失敗と数えない。
固定planned ID列からの除外・置換・順序変更・重複sample IDを拒否する。未訪問occurrenceと非停止未解決はUNKNOWNのまま、固定N confidenceを返さない。
明示的に仮定されたIID・confidence・failure policyがなければ上界はnull。
人工的な固定N完了入力だけでAのconfidence上界とabort関連機能への接続を検証し、既存即STOP engineを全件継続へ改造していない。

失敗wordを捨てた成功条件付き標本を元lawの期待biasへ渡さない。
全word compiler成功はfalse。有限の成功標本やcompiler失敗率上界は全word成功証明ではない。
実PCG64のbit replayは数学的IIDモデルの保証と区別し、`PCG64_joint_stream_qualified=false`を維持する。
架空のtotality/population/support/round enclosureを明示した人工テストの条件付きPASSは、科学READYを意味しない。

## 検証と保全

[統合テスト](../../tests/management/test_research_c_iid_calibration_integration.py)は17指定ケースを、既存関連テストと合わせて検証する。
SOURCE・law・word/angle/phase、raw/P0例外、数値不能、未完了/欠落、非unitary拒否、標本除外と重複、round/occurrence/X/Y、IID/replay、未採択confidence/policy、旧証拠流用、Stage C拒否、STOP/retry、独立SOURCE/契約を含む。

テスト件数、継承失敗の起点再現、環境・ログの正本は [tests.json](../../artifacts/research_c_iid_calibration_integration/2026-11-04/technical/tests.json)。
新規55＋A既存40＋B関連276＋Stage P関連95＝466 PASS。継承失敗5件は固定B起点で再現し、新規回帰失敗は0。
起点既知の消費済みfresh出力/SOURCE不一致を除外した理由を記録し、旧契約やSOURCEをテストに合わせて変更していない。
新規の実H4 sampling/build/compile/Arb評価、保存科学結果の再解析、RPE shots、Stage C、actual CA、重要科学レビューは0。

既存の固定venvを読み取り利用し環境を変更していない。CPU affinity32、開始時available RAM約51GiB、GPU driverは利用不能。
独立テスト群を並行実行し、重い既存列評価は元のresource planを使用した。人工的CPU/RAM/worker/wall capは追加していない。
小さい逐次callbackテストは失敗時STOPと標本順序の検証を目的として逐次実行した。新しい速度比較は行っていない。

初回の起点回帰は既存B worktreeで実施したため、pytest/bytecode cacheを生成・更新した可能性がある。
以後の起点確認は独立したBのgit archive snapshotで行い、以後の全テストを`-B`/`PYTHONDONTWRITEBYTECODE=1`/`-p no:cacheprovider`で実施した。
他jobのprocess/lockを操作せず、cacheの削除や修復も行っていない。
科学SOURCE/raw/契約/grant/ledger/marker/STOP、A/Bの既存tracked bytesと履歴の保全は [preservation.json](../../artifacts/research_c_iid_calibration_integration/2026-11-04/technical/preservation.json)で確認する。

技術SOURCE：[SOURCE.json](../../artifacts/research_c_iid_calibration_integration/2026-11-04/technical/SOURCE.json)。
これは今回の接続実装の識別子で、既存科学SOURCEを付け替えない。
Passive専用入口は`PYTHONPATH=src <固定venv>/bin/python -B scripts/prepare_research_c_iid_calibration_integration.py`。
[passive preflight](../../artifacts/research_c_iid_calibration_integration/2026-11-04/technical/passive_preflight.json)はA/Bとpilotの元の条件を読み取り専用で確認し、scientific callsは0。
新しい承認証明・grant発行・承認原文保存は要求しない。

## 残る実装と未採択の科学条件

実装上は、科学SOURCE/targetに固定された実runner、direct法のrigorous state/shell/round評価器、採択scopeに適合するfailure-aware estimator/継続runnerが未実装・未資格である。
Callback接続と架空round入力の検証を、これらの完成と扱わない。

未採択・未資格のまま残すものは、全word P0成功/停止、全word synthesis、PCG64 joint IID、compiler failure/abort・非停止/判定不能の科学的扱い、tail/variance/direct選択、N・δ_cal・δ_compile、RPE全運用abort会計、Actual CA achieved。
`all_q_words_compile_success_qualified=false`、`Stage_C_ready=false`、`actual_CA_achieved=null`を維持する。
科学結果・研究方針・科学資格は変わらないため、研究概要・科学manifest・科学ノートは更新していない。

[固定7-call pilot契約](research_c_fixed_pair_batch_pilot_contract_unapproved_20261104.json)は原本byteを保持した。
元契約の`executable=true`は実装された実行経路の属性であり、今回の実行許可を意味しない。
**今回pilot実行0。`--execute-fixed-pair-pilot`を実行していない。**
Stage P `c01bd274`、P0 `60dcaa94`、Stage R `1c0e1d1b`、Head `369e2dfa`、期待bias準備`bd560a45`の科学証拠とSTOPを保持する。

## 公開と次の担当

独立branchにcommit/pushし、remote SHA/tree、公開報告本文と主要blobを照合する。
検証記録の最終版は公開後の別技術記録として同branchに保存する。A/B branchをmerge/rebase/force-pushしない。

次は**ユーザー／研究C**が未採択の方法・failure/abort scope・N/confidenceを判断し、7-call pilotを別途実行するかを決める。
その判断と具体的依頼後、**Codex**が必要なrunner/数値資格・新科学SOURCE・call範囲・fresh出力・資源計画を整える。
新科学実行と重要科学レビューにはそれぞれ別途明示許可が必要であり、この報告から自動進行しない。

**Mandatory STOP。今回許可された技術統合・人工検証・公開照合で終了する。**
