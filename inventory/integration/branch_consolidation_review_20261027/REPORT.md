# Static research branch consolidation review

LOCAL / 2026-10-27T16:11:07.128871+09:00。現存remote branch **484本**を分類した。着手時は482本、並行Research C/DF監査の新規2本を取り込み、削除0本・既存tip変更0本（公開前remote観測間）。整理専用inputは `2fb002f96b704955a20f72c29ca5277f4abbac2f`。

## 結果

研究系列10、機能別subgroup28、具体的統合関係14グループ。共通処理または類似実装の静的対応を示したものは11グループであり、数値同等性や同一科学成果を意味しない。TYPE_1=2、TYPE_2=4、TYPE_3=5、TYPE_4=3。実コード共通化が有用な候補4、証拠索引だけが適切な候補5、既存継承のため新merge不要2、独立保持/延期3。

|研究系列|現存branch数|
|---|---:|
|DF_NATIVE_TRACK_A|47|
|DF_PF_HIGHER|64|
|ENVIRONMENT_AUDIT|1|
|PAPER_EVIDENCE|18|
|PAULI_REPRESENTATION|8|
|RESEARCH_B_B2D_OUTER|23|
|RESEARCH_C|83|
|SHARED_ALGORITHM_HISTORY|79|
|TRACK_B_CODESIGN|159|
|UNRESOLVED_RESEARCH_FAMILY|2|

研究系列・機能への対応はBRANCH_FAMILIES.csv、具体的重複はIMPLEMENTATION_OVERLAP.csv、TYPEと採用/不採用/検証/保持はINTEGRATION_CANDIDATES.csvを参照。1 branchは複数の機能関係へ参加し、14グループの件数はbranch件数と加算しない。全branchについて同じ深さのAPIレビューを行ったわけではない。

## 優先する8グループと代表候補

|順|ID|対象|代表候補|
|---:|---|---|---|
|1|INT-06|DF/RTE共通ライブラリの既継承と入口整理|`all-r-coherent-opt2-reoptimization`|
|2|INT-07|B2-D evaluatorの既統合と独立拡張|`codex/research-b-final-pause-handoff-20261026`|
|3|INT-09|原稿/CEM/図表・boundary証拠の入口統合|`codex/pr-boundary-evidence-integration-20261027`|
|4|INT-02|S2/Yoshida4係数の重複とformula allowlist|`all-r-coherent-opt2-reoptimization`|
|5|INT-01|整数count検証の共通coreと呼出し側policy|`all-r-coherent-opt2-reoptimization`|
|6|INT-03|canonical JSONとdigestのpolicyを保つ共通化|`all-r-coherent-opt2-reoptimization`|
|7|INT-05|Track B lossless storageの3完全clone|`track-b-g6-construction-paper-evidence-20261018`|
|8|INT-08|DF/Pauli/RPEの評価scope・N/G条件の証拠統合|`codex/pauli-jw-research-summary-20261026`|

実ファイル/完全commit/API/signature/差分/テスト名/採用対象/独立保持/未解決設計はINTEGRATION_TARGET_PLAN.mdの8節に記載した。代表候補は機能の再利用入口であり、全研究の採択を代行しない。

## 確認できた重複と相違

- PF係数のtrotter_2nd_list / yoshida_4th_listは、trotterlibとB2-D baselineで関数body ASTの完全一致2組。dispatcherの対応次数は異なるためS2/Y4 allowlistを維持する。

- Track B ET V5/V6/V7 storage.pyは同じGit blobの3 path。旧comparison storageとはschema/locking/historyが異なるため、旧archiveを変換せず新規callerだけの委譲を検討する。

- DF/RTEの選定11 common moduleは4 branchで同blob。all-r→H4 seal、B2-D static-cost-outer→final-pauseの祖先関係を確認した。既存継承済みのmoduleを再mergeする必要はない。

- count validatorのNumPy整数受理、JSONのensure_asciiとstr/bytes、Hadamard wrapperのprep/control/measurement、Pauli SDK/native IRの符号/bit order/global phaseはAPI目的が近くても契約が異なる。性能・数値安定性・科学的優劣の新しい判定はしていない。

## 使用資料・静的確認の範囲

正式運用指示 research_project_instructions_unified_2026-10-25.md を実ファイルで確認し§8.1/8.4/10を適用。既存のbranch保持監査、Git graph/索引、SOURCE/result対応を再利用した。PROJECT_MAP・現状/目的/方法の関連指示、VALIDATION_STATUSの関連範囲、validation_manifestを確認した。primary checkoutの研究概要は2026-09-28版であり、全系列の最新結論として扱っていない。選定branchの固定README/APIと版別CEMを併用した。

473 non-B3 branchのtip subject/変更相対pathを確認し、9保護refは公開metadataだけ。選定15 branchの117 code-path/commit対応、64固有code blob、953,893 bytes、873関数/メソッドをASTで静的比較した。さらに19 test sourceを静的確認、補助資料30件中28件取得。選定code path不在11対応、補助index不在2件は未読として記録。新規追加2 branchはpublic commit metadataのみ。全branch全ファイル・raw全件を再走査していない。

公開CEM旧43行と新50行の93版別metadataを再利用し、577行のRESEARCH_EVIDENCE_RELATIONS.csvに484 branchの固定保持tipと併記した。保存された数値結果を新規計算・再集計・評価していない。branch行のSOURCE/input/cohortが既存索引だけで確定しない場合は不明を残した。相対path hintはtip変更範囲であり全資産・物理供給契約の完全証明ではない。

## 科学的保全・限界

同じmodule/HEADは同じ科学結果を意味しない。Hamiltonian/geometry/basis/DF rankとpolicy/L_D/delta/time/law/precision/cost scope/SOURCE/cohortを版別に保持する。独立confirmation、negative result、失敗・中断・STOP、固定grant/ledger/marker、one-shot/retry/resumeを上書きしない。C_useはempirical envelopeであり厳密上界と呼ばず、H4/H6からH12を外挿しない。最終total-cost評価の完了を主張しない。

DF-native B3の5保護worktree・既知2 ref・保留7 refに内容アクセス/操作0。旧Track B/G6 B3と区別した。Research C現行研究とN1/N3旧SOURCE未解決はTYPE_4として保留。PROTECTED_AND_DEFERRED.md参照。

## 公開・非変更範囲

公開物はこのdirectoryの7資料だけ。非公開branch対応・内部path/host/IP・private配置・認証情報・非公開復旧情報は出力しない。機微を含むbranchは既存GITHUB-ANON IDを再利用し、正確な対応と詳細静的比較を本人所有の非公開領域に保持する。既存の公開/非公開原本・archive・bundleを変更しない。

研究branch削除0、merge/rebase/cherry-pick0、統合branch作成0、研究code/test/docs編集0、worktree削除/解除0、科学計算/数値再解析0、テスト実行0、A/B接続/操作0。GitHub公開は既存整理専用branchへの資料追加commitと通常fast-forward pushだけ。公開前後にremote SHAと内容を確認する。

## 次の判断とmandatory STOP

ユーザー: 上位8から次に進めるgroupを選び、代表入口、API互換policy、採用/不採用差分を判断する。コード作業としてはINT-02（S2/Y4係数）の小範囲から始める提案であり、まだ実装許可を受けた扱いにはしない。資料入口だけならINT-06/07/09を先行候補とする。

Codex: 次回の明示指示後に選択groupだけ実装・必要なAPI/synthetic regressionを行う。科学数値の再検証・fit/compileを必要とする場合は独立した科学実行許可の範囲を確定する。B3と現行研究の契約は別判断のまま保持する。今回の調査・分類・計画・匿名化公開後はmandatory STOP。
