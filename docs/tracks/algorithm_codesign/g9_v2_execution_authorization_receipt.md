# G9 v2：固定sourceへの明示one-shot authorization receipt

source S：`0ef2b92738750a3c0d187743dd4b7c9b927db802`。authorization-only commit AはSの直接子とし、実行HEADをAへ固定する。
execution branch：`track-b-g9-v2-one-shot-execution-20261010`。contractのbranch欄はsource preparation branchの記録を保持する。

## 利用者の実行指示（原文）

> source `0ef2b92738750a3c0d187743dd4b7c9b927db802` の固定契約で、authorization-only childを作成してG9 v2を一回だけ実行してください。retry=0、旧marker・結果は保持し、終了後はmandatory STOPしてください。

## Scopeとbinding

contract：`artifacts/track_b_g9_v2_api_boundary_preparation/2026-10-10/contract_v2.json`
SHA256：`2b07deb4740b6d987fe88058d05fc6dde5f6970fdeb34b69a4ba495343e88531`。
science source/contract/19-key inventory/target/provider/precision/threshold/capsは変更しない。
Aの変更はauthorization JSONとこのreceiptの2pathのみ。
旧G9 v1 marker/result/source/STOPは保持し、v2 output `artifacts/track_b_g9_p5_native_result/2026-10-10/v2` に別exclusive markerを作る。
run1/retry0、全outcomeでmandatory STOP。prefixや失敗を理由に修正・再取得・再実行しない。
分子/DF/NPZ/GPU/LP/fullv4/実量子shot/trajectory、新grid/backend/precisionは禁止。
正常/失敗後は保存値・provenanceと必要な資料公開のみ。科学的判断・次stageはGPT/利用者へ戻す。

## 結果前確認

固定Sのcritical80 hashes、旧982 protected paths、旧marker hashは一致。v2 output/markerはまだ存在しない。
A公開後にdirect parent・authorization-only diff・remote SHA・clean・runtime identity・launch gateを確認して一回だけrunnerを呼ぶ。
このreceiptに結果や期待winnerを記入しない。
