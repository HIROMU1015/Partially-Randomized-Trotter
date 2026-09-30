# PR-2 M1-B1実行契約修正 v2

日付：2026-09-30
状態：`EXECUTION_SOURCE_IMPLEMENTED_AUTHORIZATION_NOT_YET_FROZEN`

## 修正理由

実行前外部レビューは`REVISE_CONTRACT_BEFORE_AUTHORIZATION`と判定した。研究方針、194 random cell、
16 baseline cell、32 trajectory、12,448 full wrapperという科学的範囲は変更しない。修正対象は次の
2点だけである。

1. 科学実行module/runner/testを先に実装し、それらを含むsource commitを固定してから、別commitの
   execution authorizationを作る。将来追加される未固定コードを事前認可しない。
2. 科学runnerはcompiled resource mapを作るだけであり、研究判断を自動選択しない。

## terminal status

科学実行が作る正式なterminal statusは次だけである。

- `M1_B1_COMPILE_MAP_COMPLETE_AWAITING_REVIEW`
- `IMPLEMENTATION_GATE_FAILED`

`CONTINUE_RESOURCE_STUDY`、`NARROW_TO_TECHNICAL_NOTE`、`STOP_DUPLICATIVE`、
`COMPILE_RESULT_INCONCLUSIVE`は、compile map完成後に別reviewで与える研究判断であり、science runnerの
statusではない。

`pr2_matched_accuracy_m1_b1_result_schema_v1.json`は旧zero-compute bundleの監査履歴として変更せず保持するが、
科学実行には使用しない。execution authorizationが指定できるのは、上記2 statusだけを許す
`pr2_matched_accuracy_m1_b1_result_schema_v2.json`である。

## 固定する実行identity

authorizationは、実際に12,448 wrapperを生成・compileする次のsource一式を含む40文字source commitと
各SHA-256を固定する。

- `src/trotterlib/pr2_matched_accuracy_m1_b1_contract.py`
- `src/trotterlib/pr2_matched_accuracy_m1_b1_execution.py`
- `scripts/run_pr2_matched_accuracy_m1_b1.py`
- `tests/test_pr2_matched_accuracy_m1_b1_execution.py`
- `artifacts/pr2_matched_accuracy_m1_b1_contract/2026-09-30/pr2_matched_accuracy_m1_b1_result_schema_v2.json`

zero-compute execution plan v2は、そのsource commitからcandidateごとのrequest seedとbenchmarkが実際に
生成する32 trajectory seedを固定する。wrapper identityはsource commit、M1-A SHA、compiler、candidate、
axis、trajectory index/seed、wrapper semanticsを含む。cosine/sineはtrajectory seed列を共有するが、axisを
含むwrapper keyは別である。

cacheは`source commit / candidate fingerprint`単位の別SQLiteとし、cross-cell reuseを禁止する。
checkpointはtask fingerprintが完全一致する場合だけ再開に使用できる。run identityにはsource commit、
authorization SHA、execution plan SHA/fingerprint、worker数を含める。

## 不変の科学的範囲

- M1-Aでaccuracy適格だったB2 145＋B3 49を追加・除外しない。
- B0 12＋B1 4は全件compileする。不適格B0 4件はcompleteness用で、matched-accuracy frontierへ入れない。
- signalを再評価しない。
- random cellは32 trajectoryで強制停止し、追加96を行わない。
- 最大6 process worker、各BLAS thread 1とする。
- held-out pathのresolve/stat/hash/load、transfer、winner精密化、S3を行わない。
- completion後は必ず外部研究reviewへ戻る。

本修正自体はM1-B1科学実行を認可しない。次の順序を必須とする。

\[
\text{execution source commit} \rightarrow
\text{source-bound zero-compute plan} \rightarrow
\text{result-prior execution authorization commit} \rightarrow
\text{別review} \rightarrow \text{M1-B1実行}
\]
