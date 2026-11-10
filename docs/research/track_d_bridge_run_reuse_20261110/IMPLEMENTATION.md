# run単位のepoch再利用

[固定PLAN](PLAN.md)／[SOURCEと結果](REPORT.md)。新moduleは [track_d_bridge_run_reuse.py](../../../src/trottertracks/track_d_bridge_run_reuse.py)。前回fast、公式1Q合成、CX feed、T0、MPFR160の固定SOURCEは変更しない。

## 分析と選択

前回OneQStreamは各epochで全nativeを分類し、全runをbit-key化する。合成cacheがhitしてもこの走査は残る。CX削除で離れたrunが結合し、非1Q barrier消失後はdisjoint命令のflush位置も変わり得る。位置cacheや固定幅frontierは使えない。全面的な局所worklistには依存graph、順序更新、phase順序の管理が必要となるため、今回の限定目的では採用を拡大しなかった。

新実装は非空の単一wire 1Q runをtuple＋軽量_Run objectとして保持する。1Q epochは全unitを同じ順に読み、pendingを非1Q命令のqubits tuple順、最後にwire昇順でflushする。単一の不変runはmemoからそのまま出力し、内部命令の分類・bit-key生成・合成を行わない。CX削除等で複数runが結合した場合だけ内部を展開して、**前回OneQStream**へ渡す。cache missは旧公式oneq_runsのまま。新しい行列合成や相殺則はない。

## CX groupingが保持する意味

旧CX feedが判断するのは、現在命令がCXか、両wireのtopが同じ同方向CXか、である。非空1Q runは同じ一wire上でCXを遮る。run内部が複数命令でも、topは常に非CXであり、そのrunをCX処理が削除することはない。したがってそのrunを一つのopaque barrierとして旧feedへ渡しても、次のCXの判定は同じになる。空に合成されたrunはbarrierを残さない。

これはrun内の可換移動でも独立chunk価格加算でもない。CXのunbounded wire stack、同方向・両wire隣接判定、相殺連鎖は旧ObservedCXをそのまま使う。残ったunitを元run順に展開することで、最終native列のglobal orderも保存する。classical blockerと最終flushも旧処理に従う。

## memo・phase・cache

run memoは不変tupleに対する一回の評価結果である。結合したrunを旧memoで同一視しない。phase項はmemo hitでも元flush順に記録し、epochのmath.fsum、旧binary64加算順、全logical epochのMPFR160 ledgerを再現する。全wrapper global phaseを相対χと取り違えず、χやactual adjointの命令は原本どおり保持する。

前回の厳密bit cacheも再利用するが、今回は実際に評価したrunだけを直前／現passのcacheへ残す。不変runのmemoはcache keyを作らない。この保持方針によりq8では合成cache missが4回増えた。再合成の規則・精度は変わらず、全roundのwordとphaseが一致した。これは合成回数削減による速度向上ではない。

## 保持量と残る処理

caller入力は変更しない。初期nativeはT0で一度validation/countし、内部の消費済みunit list参照を解放する。run内部とcache、可変長CX node/frontierを保持し、最後に全nativeへ展開してT0で計数する。固定幅で候補を切り捨てない。

各epochで全unitを走査し、CX frontierを再構築する。unit metadata／weight集計／container列挙の走査も残る。一般worklistの更新回数は0（採用していない）。不変run内部は省略できたが、全word反復を漸近的に除去したとは言わない。O(L＋Σ(U_epoch＋D_epoch))という処理分解ができるが、Uはrun/CX等のunit数、Dは変更run展開量であり、最悪O(RL)、記憶O(L)は残る。

新H4 driverやraw QC不要の上流descriptorは追加しない。raw読込／gate.definition cache問題も今回の変更で解消していない。大規模DAG/compiler開発へ進まず、この一方式の限定結果を保全して終了する。
