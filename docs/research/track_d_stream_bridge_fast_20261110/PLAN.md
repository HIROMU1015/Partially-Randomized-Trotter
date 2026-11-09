# 同じ1Q/CX固定点の高速化：固定q4/q8性能検証PLAN

2026-11-10 JST。基点817a1cd54da11ac20dc5a84477bf0f742a567214、既存リソース評価branchを継続する。旧SOURCE／原本／価格／phase／grant／ledger／marker／STOPは変更せず、新namespaceだけを使う。

対象は保存H4 q4_sample00/cut2とq8_sample00/cut4、各Strong symmetric-directional／whole-path Reference X/Yの計8 wrapper。Linear H4／1 Å／STO-3G／Nα=Nβ=2、元H／number-CDF8／元順LD2／S2/s1/r32/K0／h=.1、元H厳密ground仮定・準備費除外、C1追加融合OFF。保存P0はQiskit1.3／opt1／seed17／rz,sx,x,cx／approx1／initially_zero=False／topology-free／既存位相規約。

変更は計算手順と保持方法だけ。1Q flush順・strict-shorter規則・全runの行列積順・同方向CXの両wire隣接相殺・round固定点は維持する。cache missは旧oneq_runsそのもの、CXは旧feedそのものを呼ぶ。入力をT0で一度検査／計数し、中間CXの使わない再計数を省く。1Q出力→CXをstream直結してfolded全listを省き、内部消費済みlist参照を解放する。cache keyはwire／opcode／float64全bitの厳密一致、前pass＋現passだけで無効な過去cacheを保持し続けない。frontierの固定幅打切り、chunk価格独立加算、rule追加はしない。全word走査やO(L)保持が不要とは仮定しない。

人工差分テストと旧回帰を確認する。既存54人工raw原本でも旧新のnative最終word・phase・round・全7資源を比較する。新P0や新人工scientific conditionは追加しない。保存raw8から既存lowererを1gateずつ使うbounded adapterでnative入力を保存し、q4は既存native列とのhash一致、q8は既存raw hash・phaseと一致を検査する。準備SOURCEとnative入力をcommitしてからbenchmarkする。

同じvenv・同じnative gzip input・同じ位相入力・同じ観測器で、8 fixture×旧新＝16独立fresh processを実空き資源に応じて並列処理する。入力decodeも含む時間とその内訳、process peak RSS、処理命令数、round／synthesis回数、cache／frontier／node／container容量を記録する。旧新とも全7資源、最終wordのbit保持hash、旧binary64 phase、MPFR160 phase、各roundが固定保存結果と一致することを要求する。価格やphaseを合わせる補正／閾値緩和はしない。

benchmarkはnative列以降を測り、raw lowering／whole-P0に対するend-to-end speedupとは称さない。前回のW依存frontier観測と異なり、旧新共通のarity依存観測を使う。同条件の対だけから比較し、q4/q8の性能をq1024〜4096へ保証しない。準備段階のbounded gate bufferとraw QC保持も別に評価する。

新sampling／P0 compile／whole-path生成／Hamiltonian・law変更／全入力simulation／q16以上／fit／N／全13-round更新／詳細科学レビューは0。技術不具合時は初回失敗・旧SOURCEを保全し、同scopeの修正だけを行う。native word／phase／固定条件の科学的不一致が未解決なら依存処理を止める。公開はscalar・hash・SOURCE・技術コードだけ、raw/QPY/NPZ／native角列／内部保存先はprivate local。公開ref/tree/bytes照合後mandatory STOP。正式研究判断はユーザー／GPTへ戻す。
