# ローカル0 WSL任意workerの引渡し

サーバー2が唯一のcontroller、Windowsが中継、WSLはhost_id=`local0-wsl`の任意worker。
既存RomuTail `local`とはprofile/spool/cacheを分離した。required_hostsへの追加は不要。

## 実機環境

通常のWindowsユーザー`ABE\romuk`で、既存Ubuntu WSL2が起動した。
Windowsは16論理CPU・約32GB RAM。WSLは16 CPU、MemTotal 16,099,660 KiB（約15.35 GiB）、
観測時MemAvailable 15,421,400 KiB。OS機能追加、再起動、.wslconfig変更は不要だった。

WSL既定ユーザーはroot。標準Pythonは3.12.3でnumpy/scipyがないため、既存の研究venv
`/root/myproject/Evaluation_numGate_highorder_DF/venv/bin/python`（3.12.3）を再利用した。
既存venvにはnumpy/scipy/mpmathがあり、更新・installは行っていない。
専用領域は`/root/projects/Partially-Randomized-Trotter-local0-wsl/.local0-wsl/`。
PROFILE.current.json、spool、dp-cacheを用意し、同時Q3枠1、DP並列16から設定変更可能。

## 4候補のWSL実機dry-check

計画33928bf80c、元計算SOURCE28d976b99、割当基点ab75bdf60。
共通修正c84786fd4（server1専用profileとRomuTail portable adapterの統合）を照合して再利用した。

| 方式 | WSLでの結果 | 検査内容・不足 |
|---|---|---|
| Full PF | PASS | 元339計画の実REQUEST、入力hash・count・metadataを既存3.12.3で検査 |
| B0I | PASS | 同上、TRUNCATED_PF |
| PR-Pauli | PASS（公開実行route33e） | c847公開assignmentを実際に検査。元bd42 routeのSOURCE受理不足は履歴保全 |
| PR-DF | 不足／HOLD | portable adapterとWSL内側27e SOURCEを配置したが、実Python3.12.3は要求3.11.0rc1 exactに不一致 |

Pauliの元339 REQUEST bytesは保存した。公開33e route実行用copyとの比較で、構成・X/Y shot配分・
SOURCE manifest・3入力blob hashが一致することを確認。変わるのは公開routeに伴うcanonical request hashで、
job IDは同じ805877c2ab6c3e03f7f1。元hash 2ee58c...、実行hash ed04775...をPAULI_ROUTE_UPDATE.jsonへ記録した。
元bd42のFAILをPASSへ書き換えず、別出力の33e実機PASSを使用する。

他ホストのPASSは転記していない。原READINESS・ADAPTER.log・resolved request・現在runtimeのindexと
共通関数から導出したHOST_ELIGIBILITYを添付した。WSL自身の判定は3 PASS・1 FAIL/HOLD。
実H4 Q3、signal生成、circuit compile、本番queue投入は0件。

PR-DFを割当可能にする次の作業は、WSL専用Python3.11.0rc1環境の準備と同profileでの再dry-check。
OS機能追加・再起動の必要はなく、この不足を共通Pythonへのfallbackやguard緩和で隠していない。

## 人工検収

最終共通版c84786fd4で`acceptance-common-96309940`を実施。
サーバー2の本物のDistributedExecutor/JobStore/loopback APIを、新しい専用directoryで使用した。
人工job 1件、同時枠1。required_hostsはserver1/localで、local0-wslを含めなかった。

- 受付停止中はWSL slots=0、人工jobはQUEUEDのまま。qg1/RomuTailは実SSH/mailboxでREADY。
- Join後にWSLが受領・実行し、server2がACCEPTED、WSL spoolはTRANSFERRED。exit=0。
- Pause後も同じWindows relay PIDでqg1/RomuTailのlast_successが進み、WSLだけslots=0。
- Resume後、同assignmentを再submitしてduplicate=true。ARTIFICIAL_EVALUATION_STARTは1回、stage bytes不変。
- WSLとserver2のOUTCOME byte hashが一致。test relay/controllerは終了済み。

job=d586dd258b6cd4173c4c、attempt=191ab88e06bf4008b118abcb70fe36b7。
qg1/RomuTailは通信継続を検査し、このfixtureではjobを実行していない。
先行566d版の人工1件は別directory・別attemptで保存し、混同しない。
Windows relayの人工unit 4件、1コマンドprofile/adapter検査15件もPASS。
検査は作成者自身が実施し、独立レビューではない。

## サーバー2が取り込む設定

同梱Git bundleの固定commitをfetchし、受領manifestのhashを確認してから取り込む。
`SERVER2_IMPORT.json`にhost登録、Windows worker項目、WSL profile、readiness descriptorを用意した。
既存3hostとrequired_hostsは維持し、optionalのworker項目だけを追加する。
新しい科学runの許可・queue投入は別の範囲。引渡し時はWSLの新規受付を停止している。

利用者の4操作は[tools README](../../../tools/local0-wsl/README.md)を参照。
