# 初回設定エラーと同一scope内の修正

SOURCE `c5e2e5046f674277778b224329301c04a52eec8f`のrun01は、保存H4 descriptorの復元と局所basis metadata取得後、最初の`make_rte_config(seed=None)`で停止した。元APIはdescriptorだけの利用でも整数seedを要求する。初回は2lawの結果を出力できず、full-path S/平均S/sampling/build/compile/fit/N/Gは0。

元SOURCE・run01/STOP.json・EXECUTION.txtは保全する。修正版は元APIの既定整数値0をmetadataへ渡し、RNG/samplerは起動しない。元式・K2・R2/R128・演算順を変更せず、整数APIの人工testを追加した。これはユーザーが許可したSOURCE固定・必要な技術実装・限定診断のdebugであり、新grantや承認証明は追加しない。

修正版は新SOURCEからrun02へ保存し、run01を上書きしない。同じ唯一のH4 descriptorを再度復元するため、復元処理の起動は合計2回になるが、対象preparationは同一hashの1個、対象lawはR2/R128の2個のままである。S0aの再実行ではない。旧STOPを消さず、追加の物理候補や直接価格へは進まない。
