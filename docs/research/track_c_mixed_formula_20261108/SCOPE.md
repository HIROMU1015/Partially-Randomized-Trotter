# Track C混合分割族：最初の機構検証

基点ce84fa8e（C2-A）、科学入力720416c5系列の保存common.pklを再利用する。
Linear H4 / 1.00Å / STO-3G / DF12 / 元H ground / 8 orbital・4電子sector70。
LD3のD4 block・R218 component、tail-free LD12を対照とする。B2 P1の入力・条件は使用しない。
delta0=.02、18round、energy精度1.5936e-4Ha、alpha総和.05、位相予算.4rad、状態準備費除外を保持する。
common SHA256は68269aa0415f048a0220beb4054daa1bba715c6eb29fbbe9d5ba779baf0346f2。
新SCF/DF生成/physical-ground solveなし。aggregate Dの固有分解はexact-D比較用の新しい演算子計算である。

既知5段族を、既存の片側D sweepと逆sweep・中央reverse/forward S2へ接続する。
coefficientsは2MN(.1931833275037836)、D二重交換子零((3-sqrt3)/6)、R二重交換子零(1/6)。
a=1/4は旧S2半刻み2回の同値対照、a=1/2は旧S2対照であり、新規改善とは数えない。
全三次logで、D内部・外側D/R・追加混合項を分ける。

初期finite-RTE検証はm=0,4,8。公平なstep menu .01/.02/.04/.08/.16/.32、
旧.02時間当たりRTE resolution menu 1/2/4/8/16/32を全partial族へ同じく与える。
実際のstepはT/ceil(T/h)、各R occurrenceのevent数はmax(1,ceil(abs(t_R)*resolution/.02))。
K2の既存P3平均・normalization・truncation boundを再利用し、追加occurrenceを無料にしない。
同じmenuで旧partial S2・partial Y4・Full S2/Y4も比較する。負時間はabsでRTE負担を数える。

初期価格は未取得。shot×(D exponentials+RTE events)は候補絞り込みのsurrogateに限り、RZ/CX等の価格とは呼ばない。
半刻み同値ではない候補に、位相/有限RTE/資源面の継続材料があれば、固定H4の全18round・必要な短価格較正とholdoutへ進む。
改善のない候補を事後的な係数候補追加で救済しない。旧modelの別回路族への無検証移送なし。
旧SOURCE/raw/grant/ledger/marker/STOPは保全し、新しい出力だけへ保存する。
一連の検証を進められるところまで完了・公開照合してmandatory STOP。詳細GPT科学レビューは開始しない。
