# ローカル0：実H4候補の遠隔dry-check結果

2026-11-17 JST。qg1・RomuTailの今回の実入力8セルを検査し、4 PASS・4不足・未確認セル0。
今回の判定は固定候補の乾式準備状態であり、未評価MSEの適合、Q3性能、DP cache HITを認定しない。

## 固定配布物と確認範囲

- 配布commit：`33928bf80cca18ce3f1fa3ddde88d27d7e2b7e5b`
- 計算SOURCE：`28d976b99cdf8329adcc1b79327e329e947ef774`
- ZIP SHA256：`5589dbf00cd4e04d8a6325cf5e33a82091f6952dc201872de45c8b8faa1f4534`
- サーバー2の固定commit内ZIP、配布実ファイル、local0取得物、両ホストの配置物のhash一致を確認。
- 各ホストでSOURCE HEAD、指定8ファイルのhash、assignmentの全入力blob hash、3 routeのHEADを確認。tracked scripts/src/configs/schemasの差分0。
- qg1では指定のSOURCE viewと3 route viewを新設。RomuTailではSOURCE viewを新設し、SHAとtracked source一致の既存3 route viewを再利用。
- 固定readiness scriptから`validate_assignment`と既存adapterの明示的な`--mode dry-check`だけを実行。各候補の不足後も残る候補を検査した。

## 2ホスト×4候補

| 候補 / request ID | qg1（server1） | RomuTail（local） |
|---|---|---|
| Full PF / `9b9b4b5132ff89768c5c` | **PASS**：固定入力・count scope・metadata PASS。[READINESS](server1/dry/FULL_PF/READINESS.json) / [ADAPTER](server1/dry/FULL_PF/ADAPTER.log) | **PASS**：固定入力・count scope・metadata PASS。[READINESS](local/dry/FULL_PF/READINESS.json) / [ADAPTER](local/dry/FULL_PF/ADAPTER.log) |
| B0I (`TRUNCATED_PF`) / `72864f6cf968d3b514b1` | **PASS**：固定入力・count scope・metadata PASS。[READINESS](server1/dry/TRUNCATED_PF/READINESS.json) / [ADAPTER](server1/dry/TRUNCATED_PF/ADAPTER.log) | **PASS**：固定入力・count scope・metadata PASS。[READINESS](local/dry/TRUNCATED_PF/READINESS.json) / [ADAPTER](local/dry/TRUNCATED_PF/ADAPTER.log) |
| PR-Pauli / `805877c2ab6c3e03f7f1` | **不足**：保存packetは受理未完。既存intakeのassembly SOURCE照合で停止。[READINESS](server1/dry/PR_PAULI/READINESS.json) / [ADAPTER](server1/dry/PR_PAULI/ADAPTER.log) | **不足**：保存packetは受理未完。同じassembly SOURCE照合で停止。[READINESS](local/dry/PR_PAULI/READINESS.json) / [ADAPTER](local/dry/PR_PAULI/ADAPTER.log) |
| PR-DF / `48494ae31276a64c46ce` | **不足**：実Pythonは3.12.3。要求3.11.0rc1 exactの`runtime_check`で停止。[READINESS](server1/dry/PR_DF/READINESS.json) / [ADAPTER](server1/dry/PR_DF/ADAPTER.log) | **不足**：Pythonは3.11.0rc1だが、旧callerが参照する内側SOURCE絶対pathが欠損しimportで停止。[READINESS](local/dry/PR_DF/READINESS.json) / [ADAPTER](local/dry/PR_DF/ADAPTER.log) |

readiness呼出しのexitはPASS=0、不足=2。adapter自体のexitはPASS=0、不足=1。
不足4件は例外終了のためadapter/DRY_CHECK.jsonを出していない。READINESS.jsonとADAPTER.logは8件すべて存在する。

## PR-Pauliの受理段階

両ホストとも固定SOURCE・入力hash・route HEADの検証は通過し、保存packetのlocatorを用いて`A.accept_saved(...)`へ進んだ。
`rpe_pauli_signal_intake._entry_assembly_source`が次の例外を出したため、保存packetの受理成功とは扱わない。

```text
Unsupported saved Pauli assembly SOURCE: entry code differs: src/trottertracks/pauli_signal_entry.py
```

固定routeは`bd42dd1e95eb1d4152216d62792ad3421b803f7c`。SOURCE guardの緩和、別routeへの差替え、信号再生成は実施していない。

## PR-DFの内側SOURCEとruntime

固定外側routeは`23160cc876807c0300be3de811f767b5810640a7`、旧callerは`e0b3cad2a024070389f757160f3a20d7da02c5ff`。
要求される内側SOURCEはSHA `27e70efe533862f38f4080aabb9cdd3ad1749fe9`、絶対pathは以下。

```text
/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/d2-pr-df-ld-split-q3-source-20261115
```

qg1ではこのpathは**今回開始前から存在するsymlink**であり、実パスは以下。解決先のHEADは要求SHAと一致、tracked scripts/src差分0。

```text
/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/rpe-distributed-q3/routes/prdf-kernel-27e70efe5
```

qg1の指定interpreter `/home/AbeHiromu/venvs/trotter-common/bin/python` は実体`/usr/bin/python3.12`、版3.12.3。
実際に固定`runtime_check`の`Use successful fixed-AST Python3.11.0rc1 environment`で停止し、後続metadataは未確認。
既存symlinkがあることだけで配置契約を全面適合とはせず、PR-DFはHOLD。

RomuTailでは要求絶対pathそのものが存在しない。import時に同path内の`scripts/run_pr_df_fullh_total_q3_fixed.py`が見つからず停止。
指定interpreter `/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python` は実体`/usr/bin/python3.11`、版3.11.0rc1（candidate、serial=1）を別途実機確認。
adapterの`runtime_check`と後続metadataには未到達。

今回symlink作成・変更、SOURCE/AST guard書換え、runtime契約変更、package install、共有設定変更は0件。
実パス・版・hashは各[server1/HOST_FACTS.json](server1/HOST_FACTS.json)、[local/HOST_FACTS.json](local/HOST_FACTS.json)を参照。

## 配置先の判断と実行境界

- **Full PF：qg1またはRomuTailへ割当可能（乾式準備状態）。**
- **B0I：qg1またはRomuTailへ割当可能（乾式準備状態）。** Full PFと別ホストへ分ける配置も選択可能。性能差は測定していない。
- **PR-Pauli：両ホストとも割当保留。** 保存packetのassembly SOURCE受理不足を維持。
- **PR-DF：両ホストとも割当保留。** qg1のruntime不一致、RomuTailの内側path欠損を維持。
- サーバー2の既存dry-check結果は今回再実行していない。全方式の3ホスト実行可能性、本番開始許可は認定しない。

8件のREADINESSでは`scientific_evaluation_calls=0`、`queue_operations=0`。
全adapterのJOB_STARTは`mode=dry-check`。回収stageログに科学実行stageはなく、科学結果のOUTCOME/RESULTも生成されていない。
実Q3・信号生成・sampling・価格較正・compile・本番queue投入・既存job collectは0件。
前回人工runのcontroller・relay・spoolは起動・再開していない。

48ファイル（各ホスト24）をremoteのCOLLECTION_MANIFEST.jsonとlocal0でhash照合し、一致を確認した。
不足の原記録と空のPASS用ADAPTER.logもそのまま保持した。

次段階はサーバー2担当がこの記録を用いて割当計画を更新し、Pauliの受理経路とPR-DFの配置/runtime条件の解決案を整理すること。
契約変更や新科学実行は今回の作業に含めない。本番queueへの投入は行わない。
