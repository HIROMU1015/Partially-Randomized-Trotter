# Track D C7-1：独立q16 whole-path探索的価格

2026-11-07 JST。[最終結果](../track_d_c7_1_q16_20261107/RESULT.md)／[実行前PLAN](../track_d_c7_1_q16_20261107/PLAN.md)。C7-0後の新しい明示許可scopeでq16/cut8/n16だけを実行し、旧C6bのresumeではない。

科学SOURCE `462f6bff66a1e3fd84a56e32aba83e9aed076257`。人工/回帰41件、raw16・P0 postcheck64・paired X/Y/χ16 PASS、SOURCE/input前後一致。sampling256 occurrence/8192 events、compile64各一回、retry/resume0。旧q2/4・q8 guardを保全し、既存C3/C5 builder、canonical sampler/B2 CAR cache、Stage A checker、C2 compiler、C6b state-action高速化を再利用した。共有環境変更・SOURCE hot-editなし。

両axis/全16本のCX差は−3376で整数候補と一致し、RZ・size・各depthも減少。平均total depth差−25648.4375、RZ差−488.75、size差−3672.75、qubits9→17。これは固定条件の有限観測で、全path/全q定理ではない。凍結C6a予測への一次12量は探索的幅内、別population-line診断はRZ depth/total depthのX/Y計4量で幅外。再fit/幅変更・不利path除外なし、STATISTICALLY_UNQUALIFIED。

raw event/QPY/NPZはローカル保全し、公開はscalar checks/価格/統計/CX残差/SOURCE/hash索引/STOP。320 payload memberのhashを照合。公開SHA/tree/bytes照合後mandatory STOP。次担当ユーザー／GPTが、q≥32で必要な構造bridge・独立検証・校正を判断し、Codexは新しい許可後だけ採択scopeを担当する。N・全RPE費用・actual成功保証・詳細科学レビューへ進まない。
