# PF直接compile：loader修正・残り216軸の承認済み再実行

## 現在の結果：2026-10-20 07:03 JST終了・partial mandatory STOP

残りH8の22軸はすべて成功し、本attemptは終了した。新216軸は成功110、control不一致STOP106、未投入0、native終了未確認0。全108 workerとcoordinatorの実wait4終了・回収を確認し、現在の稼働0。STOP106軸ではwrapper build/compileは未実施であり、compileに失敗した106軸とは扱わない。以下の05:18/05:20・起動前の記述は時点履歴である。

| 系列 | 今回成功 | 今回control STOP | 未投入 |
| --- | ---: | ---: | ---: |
| H2 DF3 | 26 | 46 | 0 |
| H4 DF10 | 62 | 60 | 0 |
| H8 DF28 | 22 | 0 | 0 |
| 合計 | 110 | 106 | 0 |

成功110と旧572 task受入を区別して保存し、合計682軸の直接ゲート数を照合した。旧実行のglobal final gate欠測は保持し、全788軸成功・旧run全体PASSとは認定しない。新規モデルfit・費用再ランキング・shot再計算・samplingは0。将来の同一候補に対するqモデル比較には、new_gate_counts.csv/combined_observation_index.jsonのtask ID・軸・SOURCE・compiler hash付き整数観測を利用できる。今回モデル誤差を計算していない。

新SOURCE/依存の実行末gateと自動監査gateは38,686件PASS。110行のCSV/JSON整数値、完全pair、旧572行の保存値、108 native worker identity、coordinator exit0/reap、call ledgerを照合した。自動metadata auditorのexit2とobserver STOPは、216未達のpartial結果を保存した後の設計どおりの終了である。証拠不整合による監査失敗と混同しない。observer自身のexitは自己報告という限界を継承する。

実時間6,439.953秒（約107分20秒）、worker CPU計52,330.987秒（約14.54 CPU時間）、最大単一worker RSS6,433,300,480 bytes（約5.99 GiB）、標本観測したowned worker RSS合計最大41,486,770,176 bytes（約38.64 GiB）。最大観測worker64。合計RSSは標本最大であり連続観測の厳密な最大ではない。終了gate時刻は2026-10-20 07:03:26 JST、自動監査gateは07:03:34 JST。

残106軸はH2/H4 B0I計96軸とH2 B0 prefix0/S4の10軸。B0Iの静的preparation選択不一致と、prefix0/S4の原因未確定を区別する。失敗時のcontrol個別errors/vectorは保存されていない。修正・追加科学実行は本attemptに自動追加せず、旧STOPを保全してmandatory STOPする。詳細科学レビュー・最終evidence freezeは未開始。

最終証拠は[results01](../../artifacts/resource_applicability/track_a_pf_direct_compile_retry216/2026-10-20/results01/completion_audit.json)、final_structure_validation.json、terminal_exit_evidence.jsonに保存した。completion_audit.jsonのworker_reserved_callsという既存fieldには実worker call総数が入るため、実compile110と親予約216をfinal_resource_summary.jsonで明確に分けた。原自動監査は変更しない。

次の担当はユーザーで、残106軸への追加対応と新規成果物の公開を判断する。今回は新規公開承認がなくpushしていない。

## 以下は準備・実起動時点の履歴

2026-10-20 JST。利用者の「その方針で進めて、再実行まで行って」を、残り216軸の修正・再実行承認として記録した。基点はa80f89b435340898f8cc46940e9080c9bcd3b403、旧科学SOURCEは220b9644df6e114116737fd041973973fcf86cf5。当初は封印・起動前の準備段階であった。以下の実行記録を現状として優先する。

[packet](../../artifacts/resource_applicability/track_a_pf_direct_compile_retry216/2026-10-20/README.md)。対象はlinear H2/H4/H8、隣接1 Å、STO-3G、tol-only DF3/10/28、T=.8、P=0、symmetric_directional。H2 72軸、H4 DF10 122軸、H8 DF28 22軸、計216軸＝108 cosine/sine pair。保存受入572軸（H6/H8/legacy DF12）は旧task来歴を保持して再利用し、再コンパイルしない。旧profile等の48軸は対象外。

## 修正と検査

H2/H4の配列receiptは元実装どおりascontiguousarray後のshape・dtype・bytesを照合する。H8は保存policy REUSED_SAVED_P21_FULL_FOCK_SCALARの入れ子だけを展開し、元scalarへ保存Cを一度だけ加える。元receiptを変更しない。

分離sessionのobserverがcoordinatorを起動し、実wait4でcoordinatorを回収する。coordinatorは各workerを実wait4で回収し、PID/start ticks・exit・CPU・RSS・結果SHAを保存する。observer自身の終了は自己報告であり、外部からのnative終了証明と呼ばない。起動元終了後の継続と成功・失敗の終了監査を人工processで確認した。新13件＋既存境界13件PASS。実分子の追加試験compileは行わず、本契約内のcontrolで検証する。

SOURCE・入力・provider・compiler/runtime・profile・旧受入結果をbyte照合して新sealへ固定する。旧実行のglobal final gate欠測は補わず、572軸のtask単位受入という限界を継承する。旧SOURCE/raw/grant/ledger/marker/STOPは変更しない。

## 実行と資源

新216軸だけのgrant・ledger・marker・STOPを独立作成する。予約はnative108、control432、wrapper build/compile各216。新sampling/fit/bootstrap/shot再計算/費用再ランキングは0。技術異常は該当task・契約でSTOPし、本attempt後の自動retryを許可しない。

並列数は実際のaffinity/cgroup・空きメモリとtask予約量から決める。H8は同一n16/q8/S4/profileの確認済み旧worker peak RSS 5,336,395,776 bytesを基準に構造比で見積もる。見積もりはadmission用で、プロセスのRAM上限ではない。CPU/RAM/worker/qの独自上限、PSIの一時上昇による強制終了、無関係worktreeの一括凍結を設けない。共有設定・他利用者を変更しない。

## 次の担当と停止条件

Codexがseal、1回の分離起動、保存結果・終了監査を行う。新SOURCEと結果の公開範囲は完成後にユーザーが判断する。詳細科学レビュー・最終evidence freeze・モデルfit・費用再会計は行わない。完了または実行不能時に理由・成功/STOP/未実施・時間/資源を保存しmandatory STOPする。

## 実起動記録（2026-10-20 05:18 JSTの暫定値）

科学SOURCEは`0c5fcc83ef55045c2c64746aadfaf3128667b058`。依存38,686件のseal gate・起動直前gate PASS。observer PID199948はPPID1/session199948、coordinator PID200066は別session200066で、起動元の終了後も実workerが継続している。実CPU affinity 64に応じて初回64 workerを投入した。固定worker上限ではなく実資源admissionによる値である。

保存されたnative終了確認時点（2026-10-19T20:18:27.474399+00:00）は成功86軸、control STOP106軸、未終了24軸。STOPはすべてCONTROL_MISMATCHで、native/controlの後、wrapper build/compile前。H2/H4 B0IとH2 B0 prefix0/S4に分かれる。STOPを成功へ繰り上げず再試行しない。残りH8等の正常workerは継続する。この時点で全体完了・全worker回収済みとは認定しない。

保存SOURCEの静的確認では、B0Iのloaderが元providerと異なり_prepare_discardを選ぶ不一致を確認した。事前の静的照合で見落としていた点である。数値寄与やH2 prefix0/S4の原因は未検証。失敗したcontrolの個別errors/vectorは保存されておらず、保存記録だけでは原因を確定できない。sealed SOURCEは変更しない。

[RUNNING_HANDOFF.json](../../artifacts/resource_applicability/track_a_pf_direct_compile_retry216/2026-10-20/RUNNING_HANDOFF.json)に実PID/start ticks・進捗・終了receipt経路を保存した。observerが全worker終了後にmetadata auditorを一度起動し、成功/STOP・ゲート数・旧572との来歴をresults01へ保存する。216未達ならpartial結果を保存してexit2となり、observerもSTOPを記録する。これは再実行承認ではない。最終報告・結果commitは終了確認後にCodexが行う。

本source/起動記録の新しい公開範囲は未承認でpushしていない。旧40ファイルの公開承認を今回へ拡張しない。最終evidence freeze・詳細科学レビューは未開始。

05:20 JST追記：running_structure_validation01.json時点でH2/H4は終了し、成功88軸・control STOP106軸・H8実行中22軸。source16file不変、observer/coordinatorのPID/start ticks一致。最終結果は未生成。
