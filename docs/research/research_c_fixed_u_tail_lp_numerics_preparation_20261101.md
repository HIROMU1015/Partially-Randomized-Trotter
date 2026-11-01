# 研究C：固定U tail LPの数値経路診断・係数保持profileの技術準備

## 1. 完了範囲と未実行範囲

起点は`222330e60179873fb3753264720697677351bfaa`。独立branch `codex/research-c-fixed-u-lp-numerics-preparation-20261101`で、人工LPの原因切り分け、最小adapter、人工・回帰検証、新しい**未承認**実行案を準備した。実装commitは`45e0dc292343a5cb38d69a94563d4e5a6dc757fa`。

**実H4保存配列の数値decode・再解析・新LP・fit・operator/PF/RTE・native・shots・総費用計算はすべて0。** 実入力に対して実施したのはhash・JSON metadata・NPZ header・package versionのpassive確認のみ。前回の`UNRESOLVED_LP`は未解決のまま保全し、今回の人工PASSを実H4の修復成功とは扱わない。重要GPT科学レビューも実施していない。

[前回の科学STOP](research_c_fixed_u_tail_preserving_lp_execution_20261101.md)は保存直鎖H4/1.00Å/STO-3G、number CDF8、sector36、元frame順LD2、h=.1/s1/r128/K0で、HiGHSのOptimal表示に対し、元行列でのdual等式残差・gap・相殺guardが未資格だった。LP1回枠とrun01は消費済みである。そのraw xを候補として採用せず、旧SOURCE・契約・入力・STOPを変更していない。

## 2. 人工問題による切り分け

[再現可能な人工study・27行](../../artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/technical/synthetic_study.json)は、固定seedで新たに作った直交U・対称J、および明示した小行列だけを使用する。保存H4のA/T/b/J/Uや旧LPのNPZは読み込まない。

| 人工ケース | 従来設定 | 今回の係数保持profile | 解釈 |
| --- | --- | --- | --- |
| Invented fixed-U seed7 | Solver success、dual残差4.336780934e-9、既存1e-9基準を超過 | 残差1.272537631e-12、既存全診断PASS | Optimal表示だけでは不十分という状態を人工再現 |
| 同seed19 | dual残差1.535955807e-10、PASS | 3.228528556e-13、PASS | 同じ診断を維持 |
| 同seed71 | dual残差5.071498776e-13、PASS | 同値、PASS | 不要なデータ変更なし |
| 完全冗長等式、RHS摂動0／4.440892099e-16 | 各profileで数値診断PASS | 同じ | 冗長性や微小RHS差だけでFAILが必ず起きるわけではない。摂動問題の厳密feasibilityを認定したものではない |
| 近従属2行、ε=1e-8／1e-9 | 元行列dual残差約.4／4.0、未資格 | dual残差は約2.78e-17まで改善するが、gap約2.07e-8・相殺guard約2.31e-5のため**未資格を維持** | 微小係数保持だけで全問題を解決できない |

Seed7でpresolveを切るとsolver status4、座標を`j=j0+d`へ変更するだけでもdual残差約4.50e-9で未資格だった。座標変更の人工対照では`b-Aj0`を保持し、RHSを勝手に0へ置き換えていない。Presolve無効化・座標変更を今回の科学案へ採用しない。

これらは技術的な機構の再現であり、**前回の実H4の大きな残差7.87が同じ原因で発生したとは確定していない**。H4の原因確定・修正効果には、別途許可された実データ評価が必要である。

## 3. 採用した最小変更とSciPyの警告

新規[adapter](../../src/trottertracks/c_fixed_u_tail_lp_numerics.py)は、既存`c_fixed_u_tail_lp.py`の組立・solve・独立診断、既存runnerの`execute`、SectorCAR／一般K／full CAR／partial PF・K0保持gateをそのまま再利用する。旧SOURCEファイルは変更しない。

変更はHiGHSの`small_matrix_value`を既定1e-9から**1e-12**へ明示することと、その設定処理の警告を固定版で確認すること。これは今回試験したnative経路での最小許容値で、科学的なacceptance allowanceの緩和ではない。HiGHSの説明ではこの閾値以下の行列要素はゼロ扱いになる。[HiGHS option定義](https://ergo-code.github.io/HiGHS/dev/options/definitions/#small_matrix_value)。同ページは開発版の説明であり、固定binaryのidentityや動作証明の代用にはしない。

SciPy1.14.1の固定[wrapper SOURCE](https://github.com/scipy/scipy/blob/v1.14.1/scipy/optimize/_highs/cython/src/_highs_wrapper.pyx)では、`small_matrix_value`が整数用ループ（121～153行）と実数用ループ（155～179行）の**両方**に登録されている。浮動小数値を渡すと最初のループが「Using default」という警告を出すが、次の実数setterで値が設定される。単なる表示から設定失敗とも成功とも判断しない。[取得元・SOURCE hash](../../artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/technical/upstream_source_reference.json)を保存した。

人工sentinel `min |j|, 5e-12*j=5e-9`では、既定値と1e-11設定でsolver status2、1e-12設定でj≈1000を返し、元行列の独立診断を通る。既定値より小さい係数が実際に残ることを確認した。実H4による試し解きは行っていない。

新runnerは2件の既知の`OptimizeWarning`を全文保存する。一方はSciPyのnative option転送、もう一方は上記の整数用重複登録による警告である。実数setterが失敗した場合は同じ文面の警告が追加され得るため、**文面だけでなく発生回数も照合**する。追加・欠落・異なる警告はraw LP保存後の`LP_diagnose`でSTOPし、既定設定へのfallbackや再solveを行わない。広い警告抑制・SciPy本体のpatch・venv変更はない。

`presolve=True`、`highs-ds`、primal/dual solver tolerance各1e-10、元の絶対j座標、独立contractionのb、全256等式行・432不等式、signed j、exact Pauli L1目的を維持する。元A/T/bをスケーリング・射影・削除せず渡し、返却結果のdualは書き直さない。SciPyのmarginalの符号は既存独立診断のまま。[固定版linprog仕様](https://docs.scipy.org/doc/scipy-1.14.1/reference/optimize.linprog-highs-ds.html)。

Native側では1e-12以下の要素のゼロ扱いや内部presolveはなお存在する。したがって全実数LPの厳密解保証ではない。元の未加工行列に対するprimal/dual/gap/conditioning/相殺guardをすべて通った場合だけ数値pilotとして資格付けする。SVDの近null空間をexactとみなしたり、等式を落としてPASSさせたりしない。

Task-local profileは専用runner process内でだけ有効とし、例外時も元moduleのoptions objectへ復帰する。並列CAR評価より前に一度だけLPを行う構成で、同一process内の複数LPの同時起動には対応しない。共有venv／他job／OS設定へ影響しない。

## 4. 人工・回帰検証

[テストSOURCE](../../tests/management/test_research_c_fixed_u_tail_lp_numerics.py)／[実行記録](../../artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/technical/test_results.json)。評価環境は既存のtask専用venvをread-onlyで利用し、依存追加0。NumPy1.26.4／SciPy1.14.1／Qiskit1.3.0などの既存lockを再利用した。

- 新profile・旧LP・Stage B・conditional shotsのpytest **91 PASS**。
- Processを分離した一般K adapter回帰 **13 PASS**。合計**104 test methods PASS**。
- 人工E2Eで全50 unitsのtensor／sector／full CAR／phase／DF drift／PF-K0保持を通し、新候補のCA・shots・総資源をnullに保つことを確認。
- 悪いdualは9-unit未解決STOP、追加native warningはraw保存後にSTOP、U変更は保持gateでSTOP。Input／親STOP identity不一致、既存output、再開を拒否し、全経路でLP1回・retry/resume0。
- Passive importとpreflightは、NPZ数値decode・LP呼出しを遮断した状態でPASS。実fit・native build・sampling・元物理normへのdispatchも人工E2Eで遮断した。

小さな人工行列ではBLASの過剰並列を避けるprocess-local threadpool調整を使用した。将来のfull CAR12因子は既存の動的resource plannerを維持し、空き資源からworkerを選ぶ。CPU/RAM/worker/wallやsolver iteration/timeに新しい人工上限は追加しない。

## 5. 新しい未承認科学実行案

[固定契約](research_c_fixed_u_tail_lp_numerics_contract_unapproved_20261101.json)／[SOURCE109ファイル](../../artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/technical/SOURCE.json)。新entryは[run_research_c_fixed_u_tail_lp_numerics.py](../../scripts/run_research_c_fixed_u_tail_lp_numerics.py)。既定動作はpassive preflightのみである。

| 固定項目 | 条件 |
| --- | --- |
| SOURCE manifest SHA256 | `0c27ca5e99f8b3d72f983f2b9ddfc13fd5939b40305a7e0788901dfa16b90fb7` |
| 契約SHA256 | `a02b6422cf87a34de647b153da65dbf6250d3c2411937d7fbc08babdf002c66a` |
| Scope | `C_FIXED_U_PR_TAIL_LP_NUMERICS_V2` |
| 元export | 保存PR-tail CDF8、SHA256 `3fdb0cb77afc770294cbe7aebf47d8a4529e3567ef9e886895495f4490b7ee21` |
| 物理・PR条件 | H4/直鎖1.00Å/STO-3G、Nα=Nβ=2/sector36、LD2、元順partial S2 h=.1/s1/r128/K0、CA=.00159Ha／DF予算=.00019875Ha |
| 固定対象 | 元U全byte、J head0/1、元g補正κ、scalar、frame順、rank8。Tail2..7の対称Jだけが変数 |
| 新LP／新DF fit | 最大1回／0回 |
| 数値資格 | 旧ALLOWANCES無変更。全元行列でdual残差≤1e-9、gapと相殺guardも既存式内であること |
| Fresh output | `artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/run01`（未作成） |
| Retry／resume | 0／0 |
| 最大科学call | **50 logical units**。旧runの残枠は転用しない |

| Call群 | 最大units |
| --- | ---: |
| 保存evidence・input・export・reference読込、固定input gate | 5 |
| LP組立／1 solve／設定処理＋独立診断 | 3 |
| Candidate export／独立tensor保持 | 2 |
| 新旧arm／law／係数gate | 6 |
| Sector／full CAR context | 2 |
| 新sector6 frame／一般K6 gate | 12 |
| 新旧full256 CAR各6 frame | 12 |
| Full tail／head-phase／DF保持 | 3 |
| 1 h-step PF・補正K0 reference／保存kernel比較／ideal attenuation13round | 3 |
| Qualification／比較 | 2 |
| **合計** | **50** |

実行順序・入力12件・call各種類は契約に固定した。LP未資格ならcomparisonを含む9 unitsで未解決STOP。設定処理の想定外警告、入力／SOURCE／tensor／CAR／phase／DF／PF-K0等のcorrectness不一致では依存callを停止し部分証拠を保存する。数値最適性がPASSでも、後続保持gateが完了するまでは新DFを採用しない。

これは**新しい実行案**であり、旧run01のretry・resumeや旧承認の残枠利用ではない。実行にはユーザー／研究Cによる別途明示的なチャット許可が必要。許可原文保存・承認記録・新grantは不要である。科学SOURCEや条件を実行後の結果に応じて変更せず、再solve・別solver・rank/U/fit探索へ自動拡張しない。

## 6. 保全・残る課題・次担当

[技術postflight](../../artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01/technical/preparation_postflight.json)で、新SOURCE／入力、旧SOURCE6群、旧raw **945件**（LP11、native729、shots92、Stage B69、共通fit37、JSON修復7）のhash一致を確認する。旧SOURCE・契約・STOP・結果は不変。実H4の旧LP問題／raw NPZはhash確認だけで、数値payloadをdecodeしていない。新しい科学結果がないため、既存の科学result manifestやH4の未解決状態を上書きしていない。

技術profileと人工試験は準備完了。ただし、実H4での設定効果・dual相殺・目的改善・tensor/operator保持は未評価である。修正後もLPが未資格になる可能性は残る。その場合、改善不能という結論や、初期値への自動fallback・候補Bへの移行根拠には使わない。元CDF optimizerのUNQUALIFIEDも変更しない。CA／全law保証／shots／native価格／総資源への新しい結論はない。

次担当はユーザー／研究C：この新しい50-call案の実行可否を判断する。Codexは別途許可された場合だけ、固定SOURCE・入力・空き資源を確認しfresh outputでLP1回と条件付き保持検証を行う。今回の担当作業は技術準備のcommit/push・remote SHA／本文／SOURCE照合までで、**公開照合後mandatory STOP**する。
