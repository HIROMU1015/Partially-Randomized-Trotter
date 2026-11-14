# checkpoint wrapperのfingerprint適用漏れ：回帰修正

**後続の実適用状況：** この修正はSOURCE0dcで実行を試みたが、同workerは残検査中に異常終了して [mandatory STOP](../d_full_h_p2_completion_20261115/REPORT.md)。実H4での完了・性能は未確認。以下の未適用という記載は、当時の技術準備の履歴である。

ユーザーの「今回のようなCodexのミスを減らしたい」に対応し、共通指示と具体的な失敗の回帰検査を整えた。**この修正は実行中SOURCE887へ未適用。科学workerは停止・再実行していない。**

## コードで確認した不具合と修正

SOURCE887の `resume_saved_path` はcompose/legacy/parity後に高速fingerprint contextを終了し、その外でX/Yの `build_native_hadamard_wrapper` を呼んでいた。同wrapperは内部でevolutionのcanonical fingerprintを再計算して不変性を検査するため、checkpoint経路ではここが旧recursive writerへ戻る。通常経路では `native.cost_trial` の高速context内でwrapperを構築する。

修正は、保存pathのX/Y wrapper構築・元recordとの照合も既存 `use_fast_fingerprint` の内側で行う最小変更。canonical bytes・hash・整合性検査・scalar/phase/parity/XY・元compile価格の照合は保持する。sampling・compile・数値gateの新規再実行は追加しない。

## 回帰検査

既存toy/fake compilerのcheckpoint testを再利用し、旧 `_Writer._circuit` に入ると `SLOW_FINGERPRINT_FALLBACK` で失敗させる。別経路のtest PASSやcontextの存在だけを判定せず、実際のwrapper内部呼出しまで検査する。

- [修正前](FINGERPRINT_REGRESSION_BEFORE.txt)：1 testが意図したFAIL。tracebackはcheckpoint→wrapper内部整合性検査→旧canonical writerを示す。
- [修正後](FINGERPRINT_REGRESSION_AFTER.txt)：既存9 test PASS。checkpointのsampler/compile呼出し0、元events/fingerprint一致、phase/parity/XY PASS。通常経路のX/Y process並列、4-qubit toy＋ancillaの実Qiskit transpile/QPY/compiled-vector、失敗時no-retryも通過。
- H4の追加試運転・sampling・compile・新科学再解析は0。人工の遅い経路再現は確認済みだが、実行中workerの現在のstackは未取得であり、その瞬間の処理位置とH4での速度改善率は未確認。

## 共通運用への反映

`/home/AbeHiromu/projects/AGENTS.md` に短い共通節を追加した。着手時の指示と実行経路の対応、早期の並列化/高速化、通常/再開/再利用経路の適用確認、長時間処理の段階別観測、実測に基づく残時間、最小の再発検査を定める。CPU稼働だけを処理の前進やPASSと扱わず、見積り超過時は対象を絞って調査する。既存のSOURCE/科学許可/保全/mandatory STOPは維持し、新たな承認帳票や管理frameworkは作らない。

現workerのSOURCE/出力へ修正を差し込んでいない。G36223 path1は保存済みPASS、path0の最終検査は継続中、全較正のC引渡しは未完了。C fit/凍結・HOLDOUT・law E[C]/N/Gへ進まない。
