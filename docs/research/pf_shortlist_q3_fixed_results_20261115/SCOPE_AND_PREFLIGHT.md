# PF shortlist6候補：固定範囲と実行前照合

2026-11-15 JST。計画正本はB50be68b803の`docs/research/pauli_q3_three_quarter_20261115/PF_FINITE_PLAN.md`。新しい科学評価は以下の6構成×各1vectorの**最大6 Q3 call**だけ。ユーザー指定の今回範囲に従い、新しい承認証明書・grant・承認原文保存等は作らない。

## 重複照合（共通指示§10.5）

- 既存B branch `codex/rpe-total-dp-acceleration-20261114` のHEAD50be68b803、worktree cleanを確認し、同じ研究branchを再利用した。他担当worktreeは読取観測のみ。
- 関連local/remote ref、直近のQ3/MSE/shortlist Git履歴、Bの必要N/readiness・Pauli2結果群、Aの完成PACKET_INDEX/B_INTERFACE_CHECK、CのALIGNMENTを確認した。B50beのPF6候補はMSE/N/G=nullで、A1adb7e65もrisk/MSE/N/G call0。旧H_eff・partial decoder・Pauli結果は物理候補又は推定量/配分が違い、この6件のtotal Q3として再利用しない。
- 確認した資料・関連refでは同じ6物理候補＋total decoder＋本vector＋target/loss/lawの保存Q3結果は見つからなかった。公開ref照合はB50be68b803、A1adb7e65a9、C c628e9e844、独立decoder f3c84aa854。確認範囲外の結果について不存在を証明したものではない。
- 自ユーザーの関連Python scientific runnerをread-onlyで確認した範囲では同一job稼働なし。全project/tmp/agent履歴の広域走査はしていない。他processを変更していない。
- Aの保存signal全78round（6×13）は再取得から除外。Pauli control・3/4配分、旧PR-DF DPも再計算しない。今回新しく必要なのは**同一PF保存packetに対する本配分のQ3だけ**。

## 入力と物理同一性

A REPORT **1adb7e65a9e7c3adc32491d853f31d124393ead7**、統合数値SOURCE **fb9a1878b677b7ee116b52e804e5a01cdaa96004**、完成carrier **06a15ee45beb177c018f6260ea4a5ac8cf66cfe6**。A PACKET_INDEXの固定configuration/packet locatorを直接消費する。元producer SOURCEと統合producer SOURCEを区別する。

B50beの`accept_completed_pf_shortlist.build()`とAのB9c0 intake6/6 PASSを再利用。C0b4b2d941の既存候補対応へ各round physical ID・元canonical・law・signed scheduleを照合した。新plannerやsignal packetは作成していない。

`T.create`→`T.bind_signal(original_packet_source=...)`で全6構成の元/total IDとSOURCEを照合済み。Full PF S2/Y4/M5とB0I p7 S2/Y4/M5を別構成で保持。元formula・actual幅/q/T・scalar・H/state/snapshot/sector・prior・normalizationを変えない。raw=corrected、B1/logB0/attenuation1、deterministic law、R/K/κ非該当を確認した。scalarを再計算・加算しない。raw・targetのUNQUALIFIED等も元のまま。

## 固定科学条件

- 全13round、両軸共通vector：`[59,55,51,47,43,39,35,31,27,23,19,15,11]`。
- allocation ID：元label＋`_DET_PAPER_SHAPED_CONTROL_01`、shot policyはUSER_SUPPLIED。paper-shaped由来のcontrolで、十分N・最適Nではない。
- Z1＋T1 total decoder、zero HOLD/tie argmin、全事象`(Ehat−Eproxy)^2`。proxy −2.1663874486347634 Haを明示し、元packetのproxy採択/ground欄を変更しない。
- τは十進εE=`0.00159360010199040`のexact二乗。既存`q3_summary`/exact判定helperを再利用する。
- 保存deterministic PF binary64 rawに条件付くIID二項測定モデル、確率則`EXACT_AFFINE_OF_BINARY64_RAW`。旧Pauli契約から**target_lawをdeterministicへ明示**し、PR finite-K/RTE lawを後付けしない。
- ρ1、round4225、verification55043173、非終端frontier8192、transition141559036。count-only上限は3600／38582272／117985808で範囲内。閉包/質量/数値gateの実際の成立は各科学call内で検証する。
- 旧Minkowskiは独立参照。直接riskへδE/δE²を二重加算しない。

## 実装と資源

`run_pf_shortlist_total_q3_fixed.py`はPF6 locator専用の新しい薄いcaller。Pauli専用workerをラベル変更して呼ばず、既存T APIを直接使用する。旧runnerから再利用するのはSOURCE検査、exact閾値/count-only helper、progress付き高速backend、write-newと資源観測。total kernel710aa69b、accelerator9e294778、C c628e9e8は無変更。

各候補は独立した`run01/<label>/`へ保存する。共有failure flagにより重大異常後の未開始science callを禁止。開始済みの独立jobは結果を保存する。無断retry、追加vector、backend/ρ/範囲変更はしない。

初期観測：64 CPU、available約488 GiB、load約2、確認範囲の同一scientific jobなし。検証済み8worker・batch8を各構成へ再利用し、6構成並列とする。これは今回のworkload/空き資源に基づく実装設定で、恒常的な資源上限ではない。他担当のcompile等を停止・変更せず、共有設定・venvを変更しない。科学job開始前と実行中に資源を観測する。

人工・回帰：新caller5件（人工risk・固定metadataのみ）、既存baseline7件PASS。caller/INPUTS/本範囲を科学実行前にcommitしてSOURCEを固定する。

完了又は重大gate失敗でmandatory STOP。新しいshot探索・他方式risk・価格/E[C]/G/winner・物理的十分N認定へ進まない。
