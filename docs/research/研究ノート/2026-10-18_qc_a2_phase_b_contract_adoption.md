# 2026-10-18：A2 Phase B科学契約Cの正式固定

ユーザーの「A2 Phase B科学契約の正式固定」を受け、[Phase A最終commit a0629ca6](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/a0629ca65142644a38cef545a9ad1244e6387eec) の草案を保持したまま、新しい [正式契約C](../qc_a2_phase_b_contract_adoption_20261018/formal_contract.md)／[machine C](../qc_a2_phase_b_contract_adoption_20261018/execution_contract_C.json) を作成する。

H4保存rank12・N=4・全12 block・固定S2順・trial 0〜3と、d=1/1000 Ha、α_spec=1/10、96-bit sqrt grid、root幅2^-40 Ha^-1、bisections256、record1 TiB/log1 GiBを採択した。理由はユーザーの明示条件であり、実入力結果に合わせた調整ではない。CPU/RAM/wall capなし、実空き資源・非干渉を確認する。既存canonical checkoutとledgerを維持し、保存先変更だけの追加準備を要求しない。

SOURCE S 8d483f9bと入力・mapping・PF・trial・算術/比較条件を [identity台帳](../qc_a2_phase_b_contract_adoption_20261018/identity_manifest.json) に結び付ける。契約メタデータ検査のみ行い、新しいscience source、実係数decode、数値再解析、certificate、authorization/grant/marker、claim消費は0。C採択はA2研究採択や実行承認を意味しない。

次はユーザーが別途Phase B one-shot実行を承認するか判断する。失敗・中断も自動retryなし、部分記録を保持しmandatory STOP。A2 NOT_ADOPTED、A1 CLOSED、K1-F未承認・未実行、他研究は不変。過去の記録・失敗・STOPは書き換えない。
