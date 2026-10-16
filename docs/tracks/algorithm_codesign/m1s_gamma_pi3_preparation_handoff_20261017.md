# M1-S π/3限定照合：技術準備と採択への引継ぎ

2026-10-17。準備完了、契約候補未採択、科学実行未承認。今回のユーザー提示文は採択／実行承認とは扱わない。新たなGPT詳細レビューは開始していない。

[契約候補](m1s_gamma_pi3_interval_gate_contract_candidate_20261017.md)と[machine契約候補](../../../artifacts/track_b_m1s_gamma_pi3_preparation/2026-10-17/v1/execution_contract_candidate.json)が対象である。SOURCE固定commitは`2f800f179c7b29589b2f9a7e532408b16b8fe6d2`、実行対象8ファイルをmachine契約のSHA256 inventoryで固定した。machine契約候補SHA256は`7989546fe64e644c01cb0080bd4623cc1b23504a5f421f446078742646db4c78`。最終公開commit／remote SHAは[publication receipt](../../../artifacts/track_b_m1s_gamma_pi3_preparation/2026-10-17/v1/publication_receipt.json)と完了報告を参照する。

- 保存G3/G6の32対照について、保存γ*区間`[L,U]`と保存π区間`[p_lower,p_upper]`を将来比較できるSOURCEを用意した。`3U<p_lower`で成立、`3L>=p_upper`で不成立、それ以外は未解像。新π・新root・新signalを生成しない。
- G9の16対照は閾値照合をせず絶対差未解像として保持し、全48対照を報告する。scopeはH4 linear1.00 Å／STO-3G／DF rank12／G3/G6/G9、保存δ1/8,1/16 Ha^-1・R2,4・T1,4 Ha^-1の既存cellである。
- 成立でもRPE成功・整数shot削減・native費用／総資源改善は保証しない。研究の継続／停止、decoder採択、候補選別、科学的GOは自動判定しない。
- exact rational比較は保存された条件付きcoverageに依存する。新SOURCEの数値correctness、実入力でのresource使用・runtime、元platform仮定は未検証。
- 新契約のresource候補はCPU1 worker／GPUなし／memory2GiB／input1MiB／result/log各16MiB／最大48対照／fraction field1024／rational 65536 bits。現行ユーザーpolicyに従いCPU／wall capなしを候補とする。旧V1のCPU160秒／wall320秒や全旧契約は変更していない。

検査は[static_validation.json](../../../artifacts/track_b_m1s_gamma_pi3_preparation/2026-10-17/v1/static_validation.json)の95 PASS／FAIL0、[preservation_audit.json](../../../artifacts/track_b_m1s_gamma_pi3_preparation/2026-10-17/v1/preservation_audit.json)の12 PASS／FAIL0。AST/schema、closed guardのmetadata検査、固定SOURCE・入力・レビュー原本のhash照合だけを行った。toy／実入力の数値correctness testは実施していない。チェックの修正・metadata guard検査は科学試行ではない。

旧非index7719ファイル、親checkoutの追跡1284ファイル・index・HEAD、既存209worktreeのHEADを保全。元レビューはSHA256 `18492941acc4fc7fe7fd46b6f002c21f324e52a661ec13bc0a024c5e86ad7303`で原本／copy同一。旧V1失敗・完了raw・消費済markerも同じbytesである。監査はローカルの技術記録でありCIや外部独立再現を主張しない。

未決は科学的／運用上の採択だけである。

1. **ユーザー**：限定decision gate `gamma_star<pi/3` と上記契約候補・条件付きcoverage・resource候補を採択するか判断する。
2. **Codex**：採択された場合に正式契約・採択記録を新規commitへ固定し、実行せずSTOPする。
3. **ユーザー**：固定SOURCE／正式契約に対する限定one-shot数値照合を別途明示承認する。
4. **Codex**：専用authorizationとmarkerを用い、承認された照合だけを実施し、失敗を含め保全してmandatory STOPする。

今回の新数値評価・保存境界への閾値代入・shot forecast・native合成・RPE simulationは0件。新authorization／grant／markerも0件。**Mandatory STOP。**
