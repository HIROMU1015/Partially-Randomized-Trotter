# Monitoring・費用・保存の受入範囲

作業基点 `6bffb5b5819918698b522270ff19b255766d3072`。API候補公開HEAD `82896d53fa97235fa848d6c0833eb193af8b99d6`、人工測定SOURCE `78c571005fe766c61b32dc867e006531099ff608`。科学SOURCE `76ca3c4089b201d55da0fede08e1f77733ac38a5` はbytes/ASTだけを読む。これはローカル技術受入であり、科学的採択・本番実行許可ではない。

|項目|取得境界の分類|今回確認できたこと/限界|
|---|---|---|
|parent/all-worker CPU、outer wall|1 既存SOURCE|portの保存outerとpair合計を照合。重複加算しない|
|condition別CPU/wall|1 既存Bundle|固定Bundle回帰に記録。portのglobal outerを条件へ推定配賦しない|
|setup/online|1 既存pair|coarse partition。constructor/preflightのtimed外は全worker CPUとの差でありstage0ではない|
|generator＋audit|1 既存pair|coarse結合。両者の完全exclusive分解は別のfine観測境界|
|angle＋requery|1 既存pair|B acceptedのみ。数学的実angleの費用は未取得|
|serialization/hash/write/flush/close|1 既存pair/Bundle|coarse/fineは代替partition。足し合わせない|
|IPC|1 既存Bundle|親outerに含む。親exclusive全内訳は未識別|
|observer|1/4|call/returnは観測可能。純粋observer単独overheadは識別不能|
|memory/RSS/HWM、I/O|1 既存SOURCE|own-process記録。連続unique global peak RAMではない|
|port bit consumption|2 新adapter|人工draw/counter/valueとeventを独立replay|
|physical storage、STOP/drain、UNKNOWN|1 固定Bundle|trusted trace/physical/RAM/journal結合。port統合保証ではない|
|Full local coefficient/child law、実path分布|3 実API接続後|静的関数境界のみ。実測値null|
|rational bit-operation cost|4 現状識別不能|Fraction C内部はPython callerへ帰属し、bit演算費用に分解しない|
|純粋observer overhead、unique連続peak RAM|4 現状識別不能|観測がない値を0や推定で埋めない|

port n256保存人工観測：parent CPU 0.119676087 s、全worker CPU 39.095133000 s、計39.214809087 s、outer wall21.936534378 s。pair CPU17.446786606 sは内数であり再加算しない。これは今回新たな科学計測ではなく固定人工rawの技術照合値。

port treeの物理bytesはsealed_small 3,971,747、sealed_n256 26,025,290（worker/sidecar/report/caller/fake store含む）。固定Bundle内のresult/log会計とは別座標。portが64GiB/256MiBのglobal保存予算へ組み込まれたことを示さない。候補のS/P1、n256、16896、result64GiB/log256MiBは未採択のまま。

S→P1固定順、host/cache/observerの交絡がある。CPU母平均CI、実科学path率、Full-N31,710,848費用、総量子cost・優位性は求めない。未測定science費用は全てnull。STOP/UNKNOWNとpost-closeの保証範囲は未接続境界資料を参照。
