# 固定H4 Q3候補：ホスト別dry-check証拠の受領と割当条件

最新の完了状態・使い方は [FINAL_HANDOFF](FINAL_HANDOFF.md) を参照。以下の途中checkpointは履歴として保全する。

最新：[12セル準備の引渡し](TWELVE_CELL_HANDOFF.md)。現在10 PASS・Pauli remote2件未確認。下記の7 PASS表と8 PASS追記は過去checkpointで、最新状態はHOST_ELIGIBILITYを参照。

返送済みのサーバー1・ローカルの8セルを受領し、転送manifestの52ファイルを全件hash照合した。
遠隔は4 PASS・4不足・未確認0。サーバー2の保存済み4セルと合わせ、
**12セル中7 PASS・5不足・未確認0**である。
遠隔dry-checkは再実行していない。新Q3・signal・sampling・価格較正・compile・collect・本番queue投入は0件。

このPASSは記録された候補・SOURCE・入力・route・interpreter・host pathの乾式準備状態であり、
科学的MSE適合や本番開始の許可を意味しない。
通常controllerの割当には、この候補別証拠に加え、現在のqueue所有状態とworkerの実行環境claimの照合が必要である。

## 固定入力と証拠snapshot

対象はlinear H4・1 Å・STO-3G・固定DF10 targetの次の4 REQUESTである。
各candidateの物理schedule・X/Y配分・評価契約・価格条件は
[前回の計画](../rpe_distributed_h4_plan_20261117/REPORT.md)を引き継ぎ、今回変更していない。

| 方式 | job/request ID | configuration IDの先頭 |
|---|---|---|
| Full PF | `9b9b4b5132ff89768c5c` | `11cf8ca6677a` |
| B0I（TRUNCATED_PF） | `72864f6cf968d3b514b1` | `a6e557770a2e` |
| PR-Pauli | `805877c2ab6c3e03f7f1` | `8bf6eb88c735` |
| PR-DF | `48494ae31276a64c46ce` | `ce9c7f082340` |

配布基点は `33928bf80cca18ce3f1fa3ddde88d27d7e2b7e5b`、科学adapter・計算checkoutは
`28d976b99cdf8329adcc1b79327e329e947ef774`。
[INPUTS.json](INPUTS.json)に12セルの元locator、保存先、SHA256、具体的な不足理由を収録した。
12件すべてでjob ID・計算SOURCE・元REQUEST hash・固定入力blob hashを339版assignmentと照合し、
`RESOLVED_REQUEST.json` の実bytes hashも各READINESSの記録と一致した。

遠隔原本は次の既存directoryを読取専用で参照した。

```text
/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-h4-plan-20261117/docs/research/rpe_distributed_h4_plan_20261117/remote_results
```

[RETURN_RECEIPT](evidence/remote_return/RETURN_RECEIPT.json)が示す
[TRANSFER_MANIFEST](evidence/remote_return/TRANSFER_MANIFEST.json)は52ファイル、計225654 bytes。
manifest SHA256は `ac703a2388af404aaf1e2d4617a8161344a6b8d1e3e6ca65d7e46b12386bab73` でreceiptと一致し、
52ファイルの欠損・hash不一致は0件だった。
[今回の読取照合結果](evidence/remote_return/VERIFICATION.json)と、相対pathを保った[evidence](evidence/)を保存した。
後続の`runs/`、automation archive、ZIP、local cacheは新snapshotに取り込んでいない。

サーバー2は遠隔受領扱いにせず `LOCAL_SOURCE` とした。
準備高速化版の最終first計測で既に実施した4件を、公開commit
`1c9c248be` の `docs/research/rpe_h4_readiness_fast_20261117/benchmarks/first/host_checks/` からbyte一致で保存した。
[server2 DRY_CACHE](evidence/server2/DRY_CACHE.json)の各 `result_sha256` もsnapshotと一致した。
これらも今回再実行した結果ではない。

## 12セルの具体的な結果

所要時間は各READINESSに保存されたchecker内部の秒数であり、ホスト性能の比較測定ではない。
表のexitはadapterの終了code（PASS=0、不足=1）。readiness checker自身の不足時exitは2である。

| host | 候補 | 結果 | adapter exit | 保存秒数 | 到達点・不足理由 |
|---|---|---|---:|---:|---|
| server2 / qg2 | [Full PF](evidence/server2/dry/FULL_PF/READINESS.json) | PASS | 0 | 1.023732 | count scope・metadata PASS |
| server2 / qg2 | [B0I](evidence/server2/dry/TRUNCATED_PF/READINESS.json) | PASS | 0 | 0.917035 | count scope・metadata PASS |
| server2 / qg2 | [PR-Pauli](evidence/server2/dry/PR_PAULI/READINESS.json) | 不足 | 1 | 0.632643 | 保存assemblyのentry SOURCE不一致 |
| server2 / qg2 | [PR-DF](evidence/server2/dry/PR_DF/READINESS.json) | PASS | 0 | 2.419595 | Python 3.11.0rc1、既存AST/runtime guard・metadata PASS |
| server1 / qg1 | [Full PF](evidence/server1/dry/FULL_PF/READINESS.json) | PASS | 0 | 0.818836 | count scope・metadata PASS |
| server1 / qg1 | [B0I](evidence/server1/dry/TRUNCATED_PF/READINESS.json) | PASS | 0 | 0.714466 | count scope・metadata PASS |
| server1 / qg1 | [PR-Pauli](evidence/server1/dry/PR_PAULI/READINESS.json) | 不足 | 1 | 0.459380 | 保存assemblyのentry SOURCE不一致 |
| server1 / qg1 | [PR-DF](evidence/server1/dry/PR_DF/READINESS.json) | 不足 | 1 | 0.257825 | adapter実Python 3.12.3。固定runtime guardが要求する3.11.0rc1と不一致 |
| local / RomuTail | [Full PF](evidence/local/dry/FULL_PF/READINESS.json) | PASS | 0 | 0.407353 | count scope・metadata PASS |
| local / RomuTail | [B0I](evidence/local/dry/TRUNCATED_PF/READINESS.json) | PASS | 0 | 0.347231 | count scope・metadata PASS |
| local / RomuTail | [PR-Pauli](evidence/local/dry/PR_PAULI/READINESS.json) | 不足 | 1 | 0.231860 | 保存assemblyのentry SOURCE不一致 |
| local / RomuTail | [PR-DF](evidence/local/dry/PR_DF/READINESS.json) | 不足 | 1 | 0.080169 | 内側SOURCEの絶対path欠損によりimport停止。adapterのruntime guard・後続metadataには未到達 |

保存証拠だけを方式別に整理すると、Full PFとB0Iは3ホスト、PR-DFはserver2だけがdry-check PASS、
PR-Pauliは全ホスト不足となる。方式や幅をホストへ固定する規則ではなく、上記の特定REQUESTに対する結果である。

Pauliの3件は同じエラーであり、保存packetの受理完了とは扱わない。

```text
Unsupported saved Pauli assembly SOURCE: entry code differs: src/trottertracks/pauli_signal_entry.py
```

PauliのC保存最安410199237 RZの通常入口への価格binding不足も、今回解消していない。
assembly SOURCE guardや固定科学SOURCEを書き換えてPASSにした記録はない。

PR-DFの固定外側routeは `23160cc876807c0300be3de811f767b5810640a7`、
内側SOURCEは `27e70efe533862f38f4080aabb9cdd3ad1749fe9`。
localで欠損していたpathは次のものだった。

```text
/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/d2-pr-df-ld-split-q3-source-20261115/scripts/run_pr_df_fullh_total_q3_fixed.py
```

[server1 HOST_FACTS](evidence/server1/HOST_FACTS.json)では内側pathは検査前からあるsymlinkで、
解決先HEADは要求SHAと一致したが、指定interpreterは3.12.3だった。
[local HOST_FACTS](evidence/local/HOST_FACTS.json)では指定interpreterの3.11.0rc1を確認しているが、
内側pathが存在しなかった。今回の受領作業ではsymlink作成、runtime変更、package installを行っていない。

## checker・worker transport・科学adapterの区別

READINESSのトップレベル `python`・`python_version` はchecker processのもので、
科学adapterのinterpreterとは限らない。
server2の保存記録ではcheckerは3.12.3、adapterの `command[0]` は既存h6環境のPython 3.11.0rc1である。
後者は[DRY_CACHE内runtime fact](evidence/server2/DRY_CACHE.json)と、
[PR-DF adapterのruntime検査結果](evidence/server2/dry/PR_DF/adapter/DRY_CHECK.json)に記録されている。
INPUTSではcheckerとadapterの実行path・version・根拠を分離した。
server1とlocalはcheckerとadapterがそれぞれ同じprofile interpreterを使っていた。

新しいcontroller/worker transportの実装SOURCEと、固定科学adapterのSOURCE28dは別に扱う。
現在workerが申告するhost IDだけで保存PASSを引き継がず、科学adapterに使うinterpreter・route・入力・pathのclaimを照合する。
transport自身のPython版を科学adapter版として扱わず、checker版からも推定しない。
controller側を更新しても、保存された科学的評価契約やSOURCE guardの条件は変わらない。

## 通常入口と割当の変更

実装固定commitは `fccc2f66034262debd5add888f7bff660d503509`。
[CODE_SOURCE](CODE_SOURCE.json)にcode/testのhashを保存した。
実装済みの同等なホスト別割当機能は関連branchで見つからず、既存worker検証・READINESS・受領manifestとqueueを拡張した。
方式名・4つのrequest IDを割当実装へ直書きしていない。

`--prepare-readiness` が、候補選定後にserver2の結果と返送済みREADINESSを結合し、
[HOST_ELIGIBILITY.json](HOST_ELIGIBILITY.json)と、候補別の許可ホスト・保留理由を持つREADYを生成する。
既定descriptorは [EVIDENCE.json](EVIDENCE.json)。別の保存証拠は `--host-readiness FILE` で指定できる。
証拠欠損・異なるREQUEST/配分/SOURCE/入力/route/runtimeはUNCHECKED、同条件で確認した不足はFAILを保持する。
checker自身のPython版をadapterの版として採用しない。

新しい結果directoryを使う通常コマンドは次の通り。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-q3-host-eligibility
/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python -B \
 scripts/run_rpe_schedule_search.py \
 --problem docs/research/rpe_distributed_h4_plan_20261117/inputs/PROBLEM.json \
 --round docs/research/rpe_distributed_h4_plan_20261117/inputs/FULL_HISTORY_PLAN_ROUND.json \
 --run-dir /home/AbeHiromu/projects/rpe-h4-host-readiness-NEXT \
 --plan-only --prepare-readiness \
 --preparation-cache /home/AbeHiromu/projects/rpe-q3-host-eligibility-checks/cache \
 --host-readiness docs/research/rpe_q3_host_eligibility/EVIDENCE.json
```

`HOST_ELIGIBILITY.json`、`READY.json` と `DISTRIBUTED_OPTIONS.json` が出力される。
最後のファイルは通常roundのexecutorへ結合する設定断片で、科学実行許可や予算は含まない。
`executor.distributed.host_eligibility` に今回生成したpolicyのpathを指定する。
controllerが新規実REQUESTを扱う場合、policy省略時も無条件割当せず保留する。
`required_hosts` は従来どおり参加待ちであり、この許可ホスト集合の代用ではない。

通常の分散roundに使う追加設定の要点は次の通り（今回この科学runは起動していない）。

```json
{
  "scheduler": "concurrent",
  "dp_workers": 3,
  "prefix_checkpoint_rounds": [5, 11],
  "prefix_cache_directory": null,
  "distributed": {
    "host_eligibility": "/absolute/new-preparation/HOST_ELIGIBILITY.json",
    "computation_root": "/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-q3-source-28d976b99",
    "computation_source": "28d976b99cdf8329adcc1b79327e329e947ef774"
  }
}
```

controllerのREQUESTを固定339と一致させるため、controller側のprefix cache pathをREQUESTに足さず、
host側profileで既存のcache pathへ解決する。現在の4REQUESTのcontroller側DP設定は上例と一致する。
workerのprofileは、dry-check済みのroot/interpreter/route_views/DP/cacheを保ち、
新しい実行専用の `work_dir` と `slots` を追加できる。これら2つの運用fieldだけではdry-check判定を無効にしない。

controllerはqueueの新jobを予約する前に、REQUEST全体hash・科学SOURCE・入力hashとpolicyを照合する。
全host非PASSなら `HOST_ELIGIBILITY_HOLD` とhost別理由を返し、予約・LAUNCH・科学予算消費を発生させない。
この技術保留をMSEのUNKNOWNや不適合評価として保存しない。
claim transactionでもhost membershipと現在のhost fingerprintを照合する。
非対応の先頭を飛ばす場合は既存priority順の次の独立候補へ進み、
先行PASSを仮定した `CHAIN_LOOKAHEAD_IF_PREVIOUS_PASSES` を穴埋めのために追加しない。
適合hostがbusyならQUEUEDで待ち、通信断を理由に別hostへ再投入しない。

### 新transportと固定科学SOURCE

workerは未割当のmetadata challengeに対し、現在のSOURCE8ファイル・HEAD、固定入力blob、
route view、profileの実Python executable/versionを読んでfingerprintを返す。
これはadapter・Q3の起動ではなく、attempt IDも生成しない。
同じinterpreter pathでPython版が変わった場合も旧PASSでclaimできない。
relayはhost IDを代理申告するだけでなく、このhost上の現在値をmailbox `readiness` 経由で運ぶ。

次の科学runには新commitのcontroller・worker transport・relayを使用する。
worker transportのcheckoutと、profile.rootで指す科学adapter SOURCE28dのcheckoutを分ける。
SSH mailbox用relay設定の `workers[].root` は新transport checkout、`workers[].profile` は新しいworker profileを指す。
そのprofile内の `root` は固定28dのままにする。
計算用localはRomuTail、local0は中継のみである。qg1/RomuTailへserver2から直接接続する前提は追加していない。

新しいtransportのworker起動は、適切な接続先URLへ次の形式で行う（次の承認済みrun用）。

```bash
python3 -B /path/to/new-transport/scripts/run_rpe_q3_worker.py \
 --profile /path/to/new-worker-profile.json --controller http://127.0.0.1:FORWARDED_PORT
```

local0は同commitの `scripts/run_rpe_q3_relay.py --config NEW_RELAY.json --start` を使い、
状態は `--status`、停止は `--stop`、再開は同じconfigで `--start`。
旧人工runのconfig・controller・spool・中継は再起動しない。
旧28d worker/relayだけでは新claim用の現在fingerprintを申告できず、新しい実jobは保留される。
既存ASSIGNED/RUNNING/UNKNOWN/完了転送待ちattemptの回収はこの新規割当判定で拒否せず、
同じhost・attemptで従来の照合・回収・重複採用防止を使う。
完了結果の再利用も新規割当gateより前の既存経路を維持した。

## 検収結果

[通常準備の結果](checks/normal_preparation/READY.json)は通常schedule入口から実際に生成し、
7 PASS・5 FAIL、Full PF/B0Iは3host、DFはserver2のみ、Pauliは全host保留を再現した。
初回context cacheなしでREADYまで107.76秒、ローカルREPORT完了108.77秒。
この確認ではserver2の4件の明示dry-check（確率評価なし）も通常準備経路で実行した。
遠隔8件は保存証拠だけを使用し、再実行・利用者への再依頼は0件である。
CLI完了後に設定断片とhost表示の整形を追加し、同じ保存結果から設定断片生成を単独で再確認した。
今回の目的は割当の正しさであり、この時間から速度倍率は算出していない。

[REQUEST_EQUIVALENCE](checks/REQUEST_EQUIVALENCE.json)では、4 REQUEST・4 assignment・
旧PowerShell手順・旧ZIPの計10ファイルがbyte一致した。
既存の科学run、固定339入力、遠隔出力path、UNKNOWN・STOPへ変更は加えていない。

[TESTS](checks/TESTS.json)に変更箇所の関連検査を保存した。
保存証拠10、transport3、readiness4、queue6、store12、controller障害5、通常入口1の計41件がPASS。
実装者自身とrootによる検査であり、独立した科学的再検証とは呼ばない。
過去の実2台/3台・切断復帰の人工検収そのものは再実行していない。

[ENTRY_ACCEPTANCE](checks/ENTRY_ACCEPTANCE.json)は `scripts/run_rpe_schedule_search.py` の通常method-round分岐を、
代替driver_factoryなしで通った人工queueの記録である。
人工4方式・job全体hashから生成したpolicyをファイル経由で読み、3hostのloopback HTTP poll/state/completeを使った。
server1/localが先にPF2方式を各1件取得し、DFはownerなしのQUEUEDに残った後server2が取得した。
Pauliは予約0・LAUNCH0・科学予算消費0、具体理由付き `HOST_ELIGIBILITY_HOLD`。
完了runを通常入口から再開して3件のLAUNCHの原bytesは不変、全方式の追加起動0だった。
実adapterと人工adapterのleaf呼出しは禁止しており、人工完了OUTCOMEだけを転送した。

追加のqueue検査で、唯一の適合hostがslots=0なら待機し、非適合hostは取得できず、
適合hostが空いた時に同jobを取得することを確認した。
欠損・別配分・別入力・別SOURCE・別route・別runtimeのPASS非流用、
旧ASSIGNEDの同attempt回復、二重完了通知の採用・予算各1回も確認した。

今回の新規実H4 Q3、signal生成、価格較正、compile、本番queue投入はすべて0件。
実機の新transport handshakeと実H4計算速度は未測定で、次の科学実行を開始する作業は含めていない。
