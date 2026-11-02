# Phase 2D-F2：限定actual資源計測

ユーザーが共通finite targetに対するγ保証を前提としない限定資源計測への変更と、一回の科学実行を明示承認した。
SOURCE固定後に一回実行し、**compiled operatorの3件FAILで停止した**。
[実行結果・残差・件数・費用・STOP](EXECUTION_REPORT.md)を現在の結果として参照する。
全raw wrapper／referenceは完了、compile後60 logical caseはPASSだが、4armの期待資源比較とmean集計は未完了。
認証N・N加重費用・同精度総資源優位は対象外で、未資格を解消済みと扱わない。

起点は資格確認 `2b57cd33d75fb756ce0d5fb712d0379137118ac9` とF2-PRE `909b13a6f5f618fc3c1ba54ce3e56764e784d12f`。
指定されたQUALIFICATION_REPORT.mdは当該commitに存在しないため、同じcommitの
[README](../pr_algorithm_improvement_phase2d_f2_qualification_20261024/README.md)とqualification.jsonを参照した。
旧統計資格付き[契約](../pr_algorithm_improvement_phase2d_f2_preparation_20261024/contract.draft.json)・資格gateは維持し、
[独立した限定契約](contract.json)を追加した。publicのsource_commitは自己参照を避けnullとし、実行packetで公開SOURCE SHAをbindingする。
正式共通指示2026-10-23の§8.4・§10・§10.3に従う。

対象は4 mode・2粒子・occupied(0,1)・L_D=1・残差density-square 2fragmentの合成fixture。
geometry／AO basis／分子DF rankは非該当。T=1/5、q1、delta=1/5の一点、r2、h1/10、K0。
Hamiltonian・state・sector・D/R・scalar・faithful identity tailを変更しない。
SS/SP/PS/PPのLは14/58/26/51。全ordered trajectory6,837、X/Y13,674＋共通prep/readout診断2で
full-wrapper build／compile要求は各13,676。reference90/128digitsは27,356評価。
異なる作業単位を科学call総数へ合算しない。診断の費用はfull-wrapper主費用から差し引かない。
既存compilerはQiskit1.3.0、opt1、seed17、rz/sx/x/cx、topology-free、approximation_degree1、
qubits_initially_zero=False。P0のrtol0・phase fitting禁止・全数値閾値と独立referenceを維持する。

## 実装と再利用

- 既存f2_executionの科学処理をmeasure_f2_circuitsへ抽出し、build／reference／compile／mean／exact weighted集計を共有した。
- 元のexecute_f2は統計資格付き経路を維持し、限定契約を拒否する。f2_preflightとf2_statisticsは変更しない。
- f2_resources.pyは独立stage `F2_RESOURCE` のpreflightとNを生成しない集約のみを追加。旧runnerはstageを拒否する。
- 新入口はscripts/run_pr_dictionary_f2_resources_saved.py。既存claim_packetと出力／STOP保存処理を再利用する。
- 数値primitive、compiler、metrics、exact IR compile cache規則は既存を使用する。worker内の同じ独立event operatorだけをcacheする。
- 管理testは既存6件＋限定経路4件の10件PASS。mockで旧grant拒否・旧統計経路の維持・N=nullを確認し、研究回路を事前生成しない。

旧F2 runnerを直接使うと統計資格とshot／selector計算を要求するため、この限定経路が必要である。
資格フラグの偽装・既存gateの無条件解除・統計式の代理実行は行わない。
保存B入力7fileをhash照合して新packetへコピーし、辞書生成と旧F1 law／価格／compileを再利用しない。

## 実行・出力・停止

新しい承認原文・契約・SOURCE closure・入力・環境をbindingし、read-only preflight後だけ新one-shotを消費する。
CPU affinity32と約53 GiBのMemAvailable、F1 worker RSSを確認した。GPU driverは利用不可で、この小系はCPUの独立trajectory／compile並列を使う。
BLAS等はworker内各1threadとしてprocess間の過剰競合を避け、worker数は実空きRAMとaffinityから選ぶ。
CPU・RAM・wallの恣意的なcapは置かず、共有環境・他者jobを変更しない。
固定SOURCEを実行中に変更しない。失敗も保全し、retry／resume・追加実験は行わない。

planned raw outputは `artifacts/pr_algorithm_improvement_phase2d_f2_resources/<execution-date>/<fresh-run-id>/`。
各wrapperのraw／compiled metrics、QPY、独立reference比較、actual block mean、正規化law、attempt CPU/wall/RSS、compile aliasを保存する。
共通physical bias1件は既存登録済みの数値診断であり主比較の保証には使わない。
主結果は各arm／axisの確率重み付き1-shot期待actual費用で、trajectory MCへの変更はしない。
N・N加重費用はnull、η・α・γ・interval資格は未成立のまま表示する。
辞書生成費は保存B再利用で未再計測であり、0費用とみなさない。

旧SOURCE・raw・grant・marker・STOPとユーザーworkspaceを保全する。publicには安全な要約・各case費用・hash台帳を公開し、
承認原文・grant・内部path・非公開rawは公開しない。
正常完了・失敗のいずれも **mandatory STOP**。次はユーザー／GPTが結果とtradeoffを確認し、γ保証や認証N比較へ進む価値を判断する。
