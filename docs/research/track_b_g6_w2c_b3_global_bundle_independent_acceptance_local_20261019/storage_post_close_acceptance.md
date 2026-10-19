# Storage V2とpost-close join

RAM monotone intentはdurability証明ではない。physical statefile、trusted bytes/rows/chain/SHA、control、partial tail、summary、condition/global receipts、diagnostic、marker、close、delivery、callerを分離した。trusted prefixのみをstream検証し、prefix外の完全に見える行も履歴に採用しない。

正常時は全48pair/F01、16896 counters、Source/contract/runtime宣言、summary、condition receipts、global receipt、診断非成功、fake STOP/spent、storage close→receipt close→marker→final slot→delivery→caller観測をjoinする。marker単独・Sだけ成功・RAM申告だけはbundle成功にならない。最終4MiB＋delivery4096B＋caller4096Bは事前global result予約で、self ACKは外部witnessが揃うまでprojectionに留める。

追加人工6故障はcondition receipt close、global receipt after-write、final write、delivery write/close、state COMPLETE pwrite。final/deliveryが空でも外側の失敗観測とhashを確認し、不完全として安全に拒否する。全RAM/cost証拠を再構成しない。

公開26不完全とfresh6不完全は、故障検出test PASSと両立する。flushをfsync/電源断保証へ格上げしない。caller観測自身のcloseを自己証明する契約でもない。trusted store/caller/安定した保存入力という境界は残る。
