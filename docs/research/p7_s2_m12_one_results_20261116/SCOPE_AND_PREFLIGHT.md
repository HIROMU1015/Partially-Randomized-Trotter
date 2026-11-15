# p7/S2・16h/7：m12各軸1shotの固定1件

2026-11-16 JST。今回ユーザー指示による新規Q3最大1件。93ab9dbbe610d66eba39ea9299b070a13a867a3eの2shot適合点を基準に、m12だけX/Y各2→1。他roundは不変で、他幅/公式/PR/配分は対象外。

## §10.5差分

関連all-refのp7/m12/Q3履歴、93ab9dbbeの3点比較と原結果、前回局所探索の早期終了、参照可能な同ユーザーの該当jobを限定確認。同configuration・全26N・decoder/target/law/lossの1shot保存/進行中結果は確認範囲で未同定。75f416de6のPR-DFやPauli LD3、別幅の結果は異なる物理候補・lawで代用しない。

保存4/3/2shotは元SOURCE付きで比較・再利用catalogへ登録し、再評価しない。元configuration a6e55777…、total e496358b…、binding d54a41b4…、carrier06a15ee4…の13round signalと1002f1dfeaの26軸価格を維持。16直接軸＋10条件付き閉包軸の元receipt・physical/canonical IDを変更しない。normalization/scalarの再適用なし。

## 固定入力・入口

```text
N_X=N_Y=[59,55,51,47,43,39,35,31,27,23,19,13,1]
allocation_id=P7_S2_16H_OVER7_M12_ONE_01
conditional_cost_RZ=376203472
previous_achieved_cost_RZ=395962678
```

同じ価格行の軸別整数積で費用を照合し、[INPUTS](INPUTS.json)に全列・価格参照・積・差額を固定。Z1＋T1、保存DF10 solver proxy −2.1663874486347634 Ha、deterministic rawのideal IID二項・EXACT_AFFINE_OF_BINARY64_RAW、全事象平方loss・exact τを継承する。

count-onlyはoutcomes3,600、verification38,522,192、transitions116,347,408。固定scopeはρ=1、outcomes4,225、verification55,043,173、非終端frontier8,192、transitions141,559,036。1shot固有の閉包・質量・残余loss・上下界/momentsは新call内で検査し、旧点のPASSを流用しない。

caller `scripts/run_p7_s2_m12_one_fixed.py` は既存single-call supervisor/worker・結果検査・高速DPを再利用する固定入口。旧SOURCEを変更せず、新namespaceとONE選択理由を明示する。新2人工/metadata test PASS。実際のexecute入口を人工supervisorで通し、1case・新保存先・旧run不変を確認した。テスト中の実H4 risk/search callは0件。[TESTS](TESTS.txt)。

caller/INPUTSをcommit/pushしてSOURCEを固定後、新run01へ最大1回。Q3内は既存8 DP worker/batch8、協調stage/segment、原API/partials/資源のworker保存と子終了観測を使う。CPU/RAMへの新制限や非同期frame dumpなし。

1shotが有効上界≤exact τで適合した場合だけ376,203,472 RZへ達成費用を更新し、不適合/INCONCLUSIVE/UNKNOWNは395,962,678 RZを維持する。重大gate失敗・保存不能では証拠を保存しSTOP、無断retryなし。結論は他round固定・X/Y同数の4/3/2/1shotという4配分に限る。十分N/物理RMSE/G_qualifiedの認定なし、AδEは再加算しない。新signal/compile/価格fit/他配分Q3は0件。公開・remote照合後mandatory STOP。
