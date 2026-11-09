# Track D T1〜H0：局所CX対応と全P0未閉包

2026-11-09 JST。[結果](../track_d_t1_h0_p0_bridge_20261109/REPORT.md)。ユーザーの一括許可に基づき、保存54人工case横断、最小CX frontier bridge、指定3 P0、保存H4 q4_sample00/q64_sample00だけを連続実施した。実行SOURCE2712a73fと公開SOURCEbae6a11bは同treeで、実行原記録を書き換えていない。

70 native列で既存counterと一致。raw適格16のうち全7資源のP0一致10、CX count/depth一致15。1Q再合成とzero処理等の不一致6も保存した。指定3qubit列のwhole P0は0CX、独立二chunkは各2CXで、chunk間のgate情報を捨てると全P0価格を復元できないことを確認した。新8＋既存29の技術回帰と3人工correctnessはPASS。

H4引用条件はLinear H4/1Å/STO-3G/Nα=Nβ2、元H/number-CDF8/元順LD2/S2/s1/r32/K0/h=.1、元H厳密ground仮定・準備費除外、C1 OFF、q4/cut2とq64/cut32。指定8compiled QPYの絶対資源は保存値と一致。raw局所の14/134 CX相殺は両方式共通で、全P0予測やReference固有の追加削減ではない。

判断材料：局所bridgeを増やす価値はあるが、1Q/Gaussian/CSWAP界面が残り、全H4／長q価格評価器は未完成。主A・比較F/Sを変更せず、N／13-round／actual保証を更新しない。次の1Q／native emitter検討は提案に留め、公開照合後STOPしてユーザー/GPTへ返す。
