# 研究C・C1：H6 rank14・保存CDF/XDFの決定論S2限定科学実行

2026-10-25 JST。固定科学SOURCE `87ea924ae28353707bd2eb3adf94231042b67361`。
Branch `codex/research-c-c1-h6-saved-pf-execution-20261025`。

**72/72 logical unitsが完了し、correctness FAILは0。保存DF予算・一般K同値性・physical/CDF/XDFのground/aliasは全PASS。CDF14／XDF14、S2 s=1／4の4候補はすべて52/52 roundの登録PF十分条件を満たし、POTENTIALLY_FEASIBLEとなった。PF誤差は両armでほぼ同水準。合成誤差・最終CA・shots・総費用は未資格／未評価であり、公開照合後mandatory STOP。**

[実行要約](../../artifacts/research_c_c1_h6_saved_pf/2026-10-25/execution_summary.json)／[全52 round CSV](../../artifacts/research_c_c1_h6_saved_pf/2026-10-25/PF_rounds.csv)／[4候補](../../artifacts/research_c_c1_h6_saved_pf/2026-10-25/candidate_summary.csv)／[CDF-XDF点比較](../../artifacts/research_c_c1_h6_saved_pf/2026-10-25/CDF_XDF_comparison.csv)／[gate](../../artifacts/research_c_c1_h6_saved_pf/2026-10-25/gate_summary.csv)／[科学call](../../artifacts/research_c_c1_h6_saved_pf/2026-10-25/calls_summary.csv)／[原本・spectra・kernels・STOP](../../artifacts/research_c_c1_h6_saved_pf/2026-10-25/run01/)／[postflight](../../artifacts/research_c_c1_h6_saved_pf/2026-10-25/postflight.json)。

## 実行条件と来歴

ユーザーが固定契約に基づく72-call限定科学実行をチャットで明示許可した。契約の歴史的unapprovedファイル名・metadataは変更していない。承認原文保存・承認証明書・新grantは作成せず、独立worktree `/tmp/prt-c1-h6-saved-pf-science-20261025` のfresh run01で1回だけ実行した。

直鎖H6・1.00 Å・STO-3G、Nα=Nβ=3／物理sector400、通常number、元g補正κ・scalar・元frame順。CDFは[修正codecの1 fit](research_c_c1_h6_corrected_fit_execution_20261025.md)で返された最終rank14のJ/Uを使用し、初期値へfallbackしない。XDFは元のselected rank14を使用した。ordered symmetric S2は全frame・PR split非適用、s=1／4、δ=0.1、m=0..12、各DFのground-state oracle／状態準備費用除外を維持した。

共通CA=0.00159 Ha、DF reserve=CA/8=0.00019875 Ha。PF誤差は各DF Hamiltonianのexact sector evolutionに対する**無次元のoperator norm**であり、物理DF誤差Haと混同しない。比較する経過時間は常に `t_m=0.1*2^m`。PF十分条件 `error <= sin(pi/12)=0.25881904510252074` は変更していない。

| 固定資料 | identity |
|---|---|
| [契約](research_c_c1_h6_saved_pf_contract_unapproved_20261025.json) SHA256 | `d37088d4012a7b9f99570386a2e13fcc1e9eb6e770a339be29300564cd16afcd` |
| [SOURCE manifest](../../artifacts/research_c_c1_h6_saved_pf/2026-10-25/technical/SOURCE.json) SHA256 | `53d5dce52cf7843e518361f9ed753a00dda91b5bab84213ae111fde0bed94969` |
| 元input SHA256 | `b82ee9864d9c76071944834ae32d3d6eea7e95959708c2e19bca011936bce447` |
| 保存CDF14 export SHA256 | `cd4de687f99d32bd6f644ac33100fb52d4f7543e710e6c273723a0908b87dc45` |
| 保存XDF14 export SHA256 | `559dc43bf923a421ccc62d05c4cefa0bdf39a48ba68824abf3c8f770c04642b1` |
| 元input・XDF SOURCE | `33c380d8aa792e5119bb3adec5dd07d1e1377214` |
| CDF fit・保存DF誤差 SOURCE | `33432e78659b8b3e654e39f1392328cb0ba5c487` |
| CDF結果公開commit | `598aa4de622b32e76b0b98451c4efb1fdaac1ae0` |

SOURCE36 files・保存入力12 hash・evaluation package versionsとlockを実行前後で照合した。固定SOURCE／runnerは無変更。直前の人工21 testsがPASSし、[準備時の人工85 tests＋29 subtests](research_c_c1_h6_saved_pf_preparation_20261025.md)も保全した。新fit・optimizer・rank探索・DF誤差再計算は0。

## 保存DF予算・一般K・ground/alias

| arm | 保存DF norm Ha | 保存guard Ha | norm+guard Ha | CA/8 |
|---|---:|---:|---:|---|
| CDF14 | 1.0689213236556284e-4 | 1.1384527736890936e-6 | 1.0803058513925193e-4 | PASS |
| XDF14 | 1.0247385485275444e-4 | 1.1384527781887395e-6 | 1.0361230763094318e-4 | PASS |

このnorm／guardは対応する保存値の再利用で、今回新たに計算したDF誤差ではない。元g・κ・scalarが同じでもCDFとXDFの物理近似誤差は異なる。既存engineering guardを厳密enclosureへ格上げしない。

一般K I/Z/ZZ展開と独立raw CARのsector同値性は、CDFのmax-entry6.4392935428e-15／F4.0930914185e-14、XDFのmax-entry9.4368957093e-15／F5.2218514130e-14で、固定max-entry1e-10／rtol0のgateをPASS。これはcomplete sector-preserving factorの表現検証であり、未生成のGaussian gates・controlled相対位相・compilerの検証を兼ねない。

| Hamiltonian | ground energy Ha | residual | alias margin | 判定 |
|---|---:|---:|---:|---|
| physical | -3.236066279892337 | 5.5801372694e-15 | 2.2943872500 | PASS |
| CDF14 | -3.2360989175122223 | 8.0341590459e-15 | 2.2943839862 | PASS |
| XDF14 | -3.2360910843201 | 4.7575676677e-15 | 2.2943847696 | PASS |

ground residual<=1e-10と `abs(E0*0.1)+pi/6<pi` を確認した。alias marginは右辺までの正の余裕である。保存spectra／eigenvectorsをexact DF evolutionへ再利用し、状態準備費用や全RPE分枝成功を評価したとは扱わない。

## 全52 roundのPF operator誤差

下表は無次元。全セルが0.2588190451以下で、登録十分条件をPASSした。RTEを使わない決定論モデルなのでattenuation=1／RTE discrepancy=0は構造上の値であり、未知合成誤差を0と認定してはいない。

| m | CDF s=1 | XDF s=1 | CDF s=4 | XDF s=4 |
|---:|---:|---:|---:|---:|
| 0 | 0.000106393375 | 0.000106394885 | 6.6461105e-06 | 6.64620499e-06 |
| 1 | 0.000212087669 | 0.000212090696 | 1.32485964e-05 | 1.32487858e-05 |
| 2 | 0.000418673437 | 0.000418679543 | 2.61538519e-05 | 2.61542338e-05 |
| 3 | 0.000796080176 | 0.000796092618 | 4.97322802e-05 | 4.97330582e-05 |
| 4 | 0.00133166153 | 0.00133168435 | 8.32015307e-05 | 8.32029568e-05 |
| 5 | 0.0015982929 | 0.00159831061 | 9.98694987e-05 | 9.98706056e-05 |
| 6 | 0.00238678267 | 0.00238683998 | 0.000149148172 | 0.000149151753 |
| 7 | 0.00435761246 | 0.00435771311 | 0.00027230501 | 0.0002723113 |
| 8 | 0.00824751269 | 0.00824774809 | 0.000515393029 | 0.000515407741 |
| 9 | 0.0153727353 | 0.0153732873 | 0.000960700873 | 0.000960735372 |
| 10 | 0.0306584953 | 0.0306595562 | 0.00191601737 | 0.00191608367 |
| 11 | 0.061144094 | 0.0611462851 | 0.00382168223 | 0.00382181925 |
| 12 | 0.122226756 | 0.122231116 | 0.00764308172 | 0.00764335476 |

[CSV](../../artifacts/research_c_c1_h6_saved_pf/2026-10-25/PF_rounds.csv)は丸め前の保存scalar、時刻、閾値、各判定を保持し、raw cells52件と照合した。

| 候補 | 最大PF誤差（m=12） | round PASS | 分類 |
|---|---:|---:|---|
| CDF:s1 | 0.12222675584872546 | 13/13 | POTENTIALLY_FEASIBLE |
| CDF:s4 | 0.007643081719996483 | 13/13 | POTENTIALLY_FEASIBLE |
| XDF:s1 | 0.12223111631013221 | 13/13 | POTENTIALLY_FEASIBLE |
| XDF:s4 | 0.007643354759299444 | 13/13 | POTENTIALLY_FEASIBLE |

同じsのCDF／XDFは全13 roundでほぼ同水準。点比較ではCDFがわずかに小さく、XDFに対する相対差の最大値はs1で約0.00359101%、s4で約0.00359096%。これは保存52 scalarの報告上の比較で、追加operator評価はしていない。厳密reference enclosureがないため、この小差から保証付きの優劣や資源順位を決めない。

s4は両armでs1よりPF誤差が小さいが、回路・shots・費用を測定していないので、s4を資源上有利としない。H6では両armのrankが14であり、H4のCDF8／XDF10やH2の費用削減率を転用できない。

## call・資源・保全

| 科学call | 実績／上限 |
|---|---:|
| saved input load | 1/1 |
| saved DF gate | 2/2 |
| sector context | 1/1 |
| sector generators | 2/2 |
| general-K representation gate | 2/2 |
| physical sector | 1/1 |
| ground/alias | 3/3 |
| sector kernel reference | 4/4 |
| PF round assessment | 52/52 |
| candidate summary | 4/4 |
| **合計** | **72/72** |

call journalはattempt72／COMPLETE72／FAIL0。登録十分条件の不適格候補も今回は0で、unknown値を補完して判定したセルはない。新fit・optimizer・rank変更・DF誤差再計算・retry/resume・native build/compile・shots・総費用計算はすべて0。

起動前はaffinity32 CPU、idle約99.44%、利用可能RAM約52.6 GiB、CPU／memory pressure avg10/60/300=0。GPU driverは利用不可で、固定NumPy/SciPy CPU経路を使用した。既存resource_planが4 workersを選び、4独立候補を並列実行した。BLAS/OMP1は各400×400行列処理と4 workersの過剰並列を避ける当該process設定であり、CPU/RAM/wallに恣意的なcapを設けていない。

壁時計10.742204秒、process CPU19.852847秒、peak RSS532468 KiB。context／DF eigenpairsの再利用とmatrix squaringを既存実装のまま用い、共有環境・他ユーザーのjobは変更していない。実測比較なしに速度倍率は主張しない。

今回raw76 files（52 cells、4 kernels、3組のspectra／ground状態等）、計約13.37 MBを保存した。SOURCE36／入力12 hash、旧H6/H2等366 filesのbyte照合PASS。H4を含む旧tracked SOURCE・raw結果・grant・ledger・marker・STOPも改変していない。新SOURCEを書き換えて結果を修正したり、旧runnerや残枠を再利用したりしていない。local evidenceであり、外部独立再現・immutable CIではない。

## 次段階に残る証拠

4候補とも今回のPF条件では除外されず、次段階の合成誤差・精度接続の対象に残せる。ただし、**POTENTIALLY_FEASIBLEは最終CA達成・厳密CA保証・総資源優位ではない。** 次が未取得である。

1. H6実native回路について、κ・scalar相対位相・Gaussianの向き・frame順・actual adjoint・controlled operator／X/Yの正しさと、P0 compiler後の正しさ。巨大8192密行列を前提とせず、既存matrix-free等の適合性を技術準備で確認する。個々の中間Gaussian gateをsectorへ射影する検証で代替しない。
2. 13 round全体の合成誤差を扱う根拠と資格。数値pilotと厳密enclosureを区別し、未資格の誤差を0にしない。
3. 保存DF誤差、今回PF誤差、将来の合成誤差、RPE統計条件を共通CAへ接続する検証。異なる種類の誤差を根拠なく単純加算しない。
4. 両armへの同じ合法的共有・融合・compiler条件で測ったnative費用、資格に応じたshots、状態準備除外の条件付き総費用。今回PF小差から費用差を推定しない。

次の担当はユーザー／研究C。4候補のどれを次のnative・合成誤差接続の技術準備対象にするか判断する。Codexは具体的依頼後に既存資産を優先して準備し、回路・合成誤差・shots・総費用の新科学計算は別の固定範囲と明示許可後のみ行う。**今回の公開照合後はmandatory STOPし、後続科学へ自動進行しない。**
