# 粗幅p7／標準Yoshida4：保存A_SHAPE整数列のQ3最大1件

保存H4 linear1Å/STO-3G/tol-only DF10・B0I p7/標準Y4/PF要求幅32h/5/h_RPE=.1/P0除外。元configuration dd8bfc248fdc67d42a6ae75df7eb0fbfcf34edc5e6ab3f3632e8784b47a7dbe0。X/Y各軸 [37,34,32,29,27,24,22,19,17,14,11,9,6] を固定する。b4296b110の現幅Y4 A_SHAPEから再利用するのは整数列と元評価契約で、現幅の信号や価格を粗幅へ付け替えない。新配分ID B0I_P7_Y4_32H_OVER5_A_SHAPE_SAVED_INTEGERS_01。

信号はf23c3f10eのBOUND_SIGNAL_INPUT.json、元decoder Z1＋T1、DF10 solver proxy、IID deterministic raw、全事象平方loss、exact τ、finite DP/残余/engineering allowanceを維持。INPUTSは278fb5214の粗幅評価入力を基点に、case label・配分ID・整数列・count-only finite bound・現幅比較locatorだけを明示変更した。準備はmetadataのみで、新信号/reference/step/回路build・compile/価格/fit/配分生成0。

専用Python /home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python（3.11.0rc1）を使用。元kernel710aa69b/accelerator9e294778/AST guardをそのまま使う。成功済みPF worker・supervisor・8-worker/batch8 DP・結果保存を再利用する。旧recovery callerの監督部分だけを同ファイルrun_caseへ抽出し、新callerから呼ぶ。旧entryの条件・scientific worker・guardは不変で、旧SOURCE/原結果/UNKNOWN/STOPも保持。新scope/出力は本ディレクトリrun01のみ。

人工・metadata 8 test PASS後にcaller/helper/INPUTS/DELTA_CHECK/TESTS/本計画をcommit固定し、専用Pythonから scripts/run_p7_y4_coarse_a_shape_q3.py execute を一回実行する。out.mkdirの再入拒否と最大1caseを維持。ASTの正規化、期待hash置換、retry、他配分の補完はしない。共有環境・他担当processへ干渉しない。periodic faulthandlerを使わず元stage/round logと親monitor・child exit code/signalを保存。

§10.5：前回preflightを継承し、関連D/B/Aの公表headと差分metadata、同ユーザーjobを再確認。同等Q3は確認範囲で0。現幅A_SHAPEは幅/信号が違い、粗幅B_PRICEはN列が違う。後続Pauli LD3結果・進行PR DF LD7は別候補。未解決項目があれば科学callをしない。詳細DELTA_CHECK。

上界≤元exact τは適合（Q3適合・価格待ち）、下界>τはこの配分でモデル不適合、跨ぐ場合INCONCLUSIVE、input/数値/保存gate不成立はUNKNOWN。結果が不適合なら粗幅で試したB_PRICE/A_SHAPEがともに不適合と記録するが、他配分の不可能性は主張しない。現幅16h/3・粗幅32h/5 × B_PRICE・A_SHAPEの4点を固定locator付きで公開し、今回以外の3点は引用のみ。粗幅費用は常にnull。新規Q3最大1件、新信号/reference/build/compile/価格/fit/他配分各0、commit/push・remote確認後mandatory STOP。
