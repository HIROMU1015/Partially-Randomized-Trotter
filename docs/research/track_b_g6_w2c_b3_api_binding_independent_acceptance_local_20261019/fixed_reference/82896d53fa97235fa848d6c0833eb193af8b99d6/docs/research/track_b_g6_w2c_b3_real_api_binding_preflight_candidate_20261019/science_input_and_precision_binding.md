# 固定入力・precision binding

固定SOURCE：[78c571005fe766c61b32dc867e006531099ff608](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/78c571005fe766c61b32dc867e006531099ff608)。直接基点 `d24cccbecdf0928fe977080dd7b28c591df0089e`。ローカル独立受入 `6bffb5b5819918698b522270ff19b255766d3072` は固定参照のみでmerge/rebaseしない。

今回の範囲は静的読取・AST・hash照合と人工技術検証。実科学入口閉鎖、科学的採択なし、mandatory STOP。サーバーCodex主担当一名による自己検証であり、別reviewerの独立受入ではない。

P5_general_order、p=(1/5,3/10,1/2)、x=5/7、m=5、sigma=+1。root_bits256、probability_bits160、eta/rho各1/10^12、epsilon_axis1/200、primitive_error1/10^6、alpha_axis1/320、familywise alpha1/20を固定参照から保持。A6はtau1/10^6、K_phi32、J32のmath-only。値の再計算、再最適化、保存科学値の新規数値再解析はしない。

technical_contract_candidate.jsonに元reference全体、SOURCE/input pins、runtime executable/stdlib/extension hashesを保持。preflightは実SOURCEとAST digest、入力bytes、runtimeと完全candidate一致を確認する。source_inventory.jsonは固定runtime52 filesと入力10 filesの保全を記録。必須固定資料34件のGit object/SHAはfixed_reference_inventory.json。

G7のroot interval、largest-remainder dyadic law、downward acceptance、有限rational coefficientとproposal/weightの意味論を変えない。F3はcaller-supplied precisionのみで適応選択/再試行なし、P0〜P4と実lawのP5 binding、同event requeryとhalf-angle/alternating atanを保持する設計参照。今回はそれらを実行していない。canonical Bは既存overlayの別監査でありFull F3 preflightと同一certificateではない。
