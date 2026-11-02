# 2026-10-20 H6段階B：保存処理修正と承認済み再実行

ユーザーがc585cef8のSOURCE重複保存不具合の修正と、同じ30 scopeを新契約でone-shot再実行することを明示承認した。旧SOURCE/部分結果/grant/ledger/marker/STOPはそのまま保全。
科学coreと入力・条件・予算を保持し、既存SOURCEの一致確認を含む全出力保存処理だけを修正した。新10人工＋既存27回帰＝37PASS後、SOURCEc0d8ca8ecd3fc29528ca8424f8b8d953d06eff91をsealした。

新30は候補/q/prefix/PF次数変更0、最良総費用変更20、PR/PF方向逆転0、PR低費用30。
旧70をliteral参照した全100は候補変更2（H8非相殺ε=.0005、倍率1/10、strengthened184のp16→p15）、q/PF次数変更0、最良総費用変更50、方向逆転0、PR低費用90/PF低費用10。
全候補順位の厳密逆転180pair・同率変化16pairはscope重複を含む。新baseline34費用は旧保存winner再現とは区別した。

既存712費用exact再現、整数別検査1492費用/30scope/41578pair、CSV/JSON・SOURCE/SHA・入力/runtime・旧112証拠・予算・図coverageを照合してPASS。
native exit0、親wait4/reaped・owned worker0、wall4.90414447337389秒。科学実行は新契約1回、追加retry0。
2図6exportsと100条件の表・CSV/JSON・報告を新packetに保存。旧70/Stage A再解析、signal/sampling/compile/shot/refit/PR統計は0。主表・Figure2・v2本文は不変。

[結果・来歴・限界・監査](../track_a_h6_noncancellation_stage_b_rerun_20261020.md)を参照。
profile/runtime差、S4仮説、初回最終gate欠測、旧48軸の制約、母平均/希少event未保証を維持する。100scopeはunique40 endpointで独立100反復ではない。
次担当ユーザー：公開範囲を判断。Codexは公開承認範囲だけpushする。追加科学は不要、詳細レビュー/freeze未開始、mandatory STOP。
