# compiled価格pipelineの高速化と最小P2計画

2026-11-13 JST、Codex D。既存branch `codex/d-compiled-residual-c-minus-s-20261113`を
P1完了commit `54b64ef72bf507be25970f2d3362d3be091321c3`から継続した。

**技術改善・人工同値性検証を完了。実H4の高速化倍率は未測定。P2科学実行は未承認・未実行。**
[TECHNICAL_HANDOFF](TECHNICAL_HANDOFF.md)にAPI・利用方法・テスト、
[P2_MINIMAL_PLAN](P2_MINIMAL_PLAN.md)に精度・law・rare・有限実行案を記載した。

## 確認した既存SOURCEと再利用

| 役割 | 実際に確認したSOURCE・箇所 | 今回の扱い |
|---|---|---|
| P1完了結果 | `54b64ef72bf507be25970f2d3362d3be091321c3`、d_p1_efficient_czz REPORT/HANDOFF/EXECUTION_RUN02 | 既存の性能内訳・科学観測を引用。raw価格の再集計・fitなし |
| P1実行 | `20fe11942369253361068dd2b00f98b26350ba8d`、run_p1_efficient_zz.py/p1_efficient_zz.py | 固定C1 loader・新lowering・人工回帰を再利用。旧runner/PLAN不変 |
| 残差基盤 | `af6f27d7a6467bb355686421acd067544edcbbd2`、compiled_residual.py、旧REPORT/SCIENCE_PLAN | 同pathのC−S・役割分離・未資格値を維持。旧P2/P3を自動採用しない |
| C全26 E[S] | `9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f`、h4_structural_schedule_results REPORT/HANDOFF | identity・元区間・source付き再利用条件を確認。値の数値再解析なし |
| C law調査 | `d72dc4e6228bb380f4d1d000e5e030794fce7164`、finite_k_sampler_law_alignment REPORT | ideal/CDF/53-bit/PCG64とcohort誤差を分離。実law診断は実行しない |
| B価格接続 | `bf515a976f1987ad6a19c61d14689b27b0894542`、rpe_s3_readiness REPORTとD/B adapter | 未資格E[C]の扱い、law不一致/null、旧loweringとのidentity分離を維持 |
| C1 Backend | `1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7`、r_price_phase1_science.Backend.context/price_requests/structural | pinned blobを読込み、拡張classの基底と人工baselineに使用 |
| Accounting | C1 BASE `8457c443554af5fadaa3ba36f97f6316ca3c5050`、h4_df10_price_v1.Accounting/track_d_native_resources | 元の展開・basis相殺・Zfoldを継承。RZ取得に不要なdepth更新だけを除く |
| compose | carrier `155529e20a8fac39e00786c40b57830b71203c4f`、h46_long_time_native.compose/schedule | 元関数をそのまま使用。人工fixtureのGaussian loweringも同SOURCE |
| provider | `c31ef424175c02379eafebe8fc6c7548c1d9aabe`、df_rte_qiskit/ax2b_native_df_v5/rte_compiled_cost/h6_hash_reuse_v1 | 実ファイルと既存公開関数を確認。編集なし |
| 過去の性能改善 | `e75e1b217` streamed_circuit_fingerprint REPORT、`c71bb8227` b2_fingerprint_pipeline REPORT、既存h46_b3_fast_fingerprint.py/tests | canonical streamingとcall内ゲート再利用の実装・限界を確認し、適合する既存fast writerを再利用 |

既存streaming成果は別の人工入力・compiler seed7の検証であり、今回のC1/seed17経路の
性能・同値性証拠を代替しない。新たにpipeline接続だけを追加し、serializerやcompilerを作り直していない。

## ボトルネックの調査

P1報告のm6 1 pathはprice全体3,139/3,187秒、X/Y compileはそれぞれ約4分、
worker peak RSSは2.86/2.81 GB。元報告のfile時刻からの概算は、構造S約6分、
compose＋X wrapper（hash含む）約22分、照合とY wrapper約12分、最終照合約3分。
これは**既存報告の引用**で、今回rawや時刻を再解析して得た値ではない。

SOURCE調査で確認した処理は次のとおり。

- C1は同じ本体のSをX/Yについて2回計数する。Accountingが使うNativeCounterは
  RZだけでなくCX・3種のdepth frontierも全gateごとに更新するが、price_requestsが読むのはstructural_rzだけ。
- composeは全pieceでtailをbuildし、部分PF/tail回路のNativeEvolutionとfingerprintを生成して
  全体へcomposeし、最後に全evolutionをfingerprintする。Pythonゲート生成・走査が繰り返される。
- wrapperは両軸で_gate_only走査、evolution再fingerprint、Qiskit composeを行う。
  Backendも各wrapperをcompile前後にfingerprintする。
- 既存h6_hash_reuseは現graphのcontent guardを再走査してからdigest/bytesを再利用する。
  guard側の再帰展開・JSON化にも時間が必要。SHA256そのものを弱める解決ではない。
- Qiskit compose/copy・tail basis append、event.to_dict/JSON出力、回路の保持は残る。
  今回はgate列を書き換えず既存関数を使うため、これらの重複copyを直接変更しなかった。

新実装は、(1)元AccountingのRZ-only projection、(2)finish前状態の分岐によるS本体の共用、
(3)既存fast canonical writerのC1接続、に限定した。回路意味・制御・phase・測定・gate順序と
compiler設定は維持する。X/Yを個別に取得し、cache/checkも維持する。

## 人工同値性・回帰

新9項目＋既存lowering10＋fast hash8＋残差13＋mean price15＝**55 distinct項目PASS**。
作者自身の技術確認で、独立科学レビューではない。

新テストは任意の8-wire行列と明示event列を使い、実H4 input・保存trajectory・samplingを使わない。
旧Backendと高速版のevent bytes/digest・構造出力・X/Y wrapper fingerprint・compiled回路fingerprint・
6価格指標・測定が一致した。cache/compiler binding・変異検出・新lowering要求も確認した。

初回の合同pytestでは54 PASS/1 FAIL。失敗は残差adapterの「科学libraryが未import」検査で、
同じprocessに人工Qiskitテストを混在させたことが原因だった。
[原記録](TEST_COMBINED_IMPORT_CONFLICT.txt)を保全し、テストを書き換えず
[専用processで13 PASS](TEST_ISOLATED_RESIDUAL.txt)を確認した。
provider初期化順序をcontext優先に修正した後、[pipeline/lowering/hashの27 PASS](TEST_PIPELINE.txt)を再確認。
mean priceの15 PASSは初回記録を再利用した。
Qiskit1.3の大量のDeprecationWarningは既存property参照によるもので、設定・機能は変更していない。

## 人工性能比較

[BENCHMARK.json](BENCHMARK.json)に最終実装の全観測・SOURCE SHA256・内訳を保存する。
初期化順序修正前の測定は[BENCHMARK_INITIAL](BENCHMARK_INITIAL.json)として区別した。
baselineはpinned C1のprice_requestsそのもの、改善版は拡張class。
両方とも新controlled ZZ・同じ明示event列、Qiskit1.3.0/NumPy1.26.4、rz/sx/x/cx/opt1/seed17。
old cacheは空の人工一時directoryで、両軸を実際にcompileした。

各条件3反復、毎回fresh spawn process、旧/新の実行順を交替。
wallはimport/fixture構築を除くprice全体、peak RSSはそれらを含むprocess lifetime high-water。
tracemallocやCPU/RAM/affinity制限なし。自分の計測worker同士の競合を避けるため順次計測した。
全反復でevent digest・S/C・両軸fingerprint・compiler・6指標が一致した。

時間は中央値（範囲）、RSSは中央値、MB=10^6 bytes。

| 人工case | 明示event位置 | 旧wall秒 | 新wall秒 | 倍率 | 旧peak RSS MB | 新peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|
| q2/R4 | 8 | 3.345 (3.336–3.348) | 1.586 (1.561–1.596) | 2.11 | 380.08 | 373.37 |
| q8/R32 | 256 | 12.126 (12.046–12.251) | 4.890 (4.824–4.912) | 2.48 | 395.53 | 390.42 |

q8/R32の段階時間中央値（秒）。compose/wrapperには内部の検証hashと走査を含む。

| 段階 | 旧 | 新 |
|---|---:|---:|
| 構造S（X/Y） | 1.130 | 0.324 |
| compose（部分/全evolution hash込み） | 3.270 | 1.416 |
| X wrapper（evolution走査/hash込み） | 0.886 | 0.369 |
| X fingerprint/cache key | 1.580 | 0.206 |
| X compile＋6指標取得 | 0.483 | 0.592 |
| X mutation check | 0.715 | 0.207 |
| Y wrapper（evolution走査/hash込み） | 0.896 | 0.348 |
| Y fingerprint/cache key | 1.452 | 0.214 |
| Y compile＋6指標取得 | 0.448 | 0.447 |
| Y mutation check | 0.852 | 0.355 |

段階中央値の和はwall中央値とは異なる。baselineのJSON出力等はwallには含むが上表には分離していない。
新側で分離したevent.to_dict/JSON出力/digestの中央値は0.156/0.193/0.034秒。
compileを速くしたという結果ではなく、Python側の計数/hashが主な改善である。
RSS削減は小幅で、全回路copyの除去や大規模メモリ問題の解消を示していない。

本測定は人工入力のlocal観測である。実H4の時間/RSS/倍率、長q/Rでの実行可能性は未測定。
特に人工caseのDの符号やorder2頻度を科学的なlaw観測に用いない。

## 最小P2の判断

平均compiled価格に必要なのは、同じlawでのE[S]＋E[D]と誤差予算であり、残差全機構の解明ではない。
全26 E[S]は再利用し、必要点の直接paired samplingを第一候補にする。
提案精度は2%×保存E[S]、使用価格全体95%confidence（採択待ち）。
cohort/law/rare/S数値区間を分離し、実用PRNGモデルと厳密保証も区別する。

次の有限案は、P1保存m0/m6の各1 replay（4 compile）＋fresh各4 path（16 compile）、
**新sampling8 path・最大20 compile**。校正/holdout/fitは不要な直接法pilotで、精度認定ではない。
その後のnは実測分散と必要精度から別の有限案として決める。
rareは平均価格認定前にtail上界で予算を検査し、必要な場合だけ層別評価を追加する。
長roundへmodelが不可避なら最小の校正・fresh holdoutを用意し、最大使用点の確認なしに
m0/m6やm7から全13roundへ外挿しない。詳細は[P2_MINIMAL_PLAN](P2_MINIMAL_PLAN.md)。

## 保全・残る作業・STOP

旧SOURCE/input/raw/result/grant/ledger/marker/KEEP/STOP、P1 run01/run02・PLAN/RUN02_PLAN、
既存providerとC1/S0a/K0/K1は変更していない。新規の科学実H4 sampling/build/compile、
保存科学結果の数値再解析、fit/校正、全13round平均C認定、N/RMSE/G、新8次PFは0。
新ファイルと索引への技術項目追加だけを今回のcommitへ含める。

人工改善が得られたため開発範囲をここで止める。composeのpieceごとの中間hash、
wrapperの再走査、Qiskit既定copy、event serializationとcall単位O(L) fingerprintは残る。
X/Y並列化は回路受渡し・RSS増との実測が必要であり、path間並列の既存APIを維持する。

- **ユーザー**：P2案の精度・target law/PRNG・必要domainと、将来の有限科学実行範囲を判断する。
- **Codex D**：今回の成果をcommit/push、remote SHA・tree・主要成果物を照合してmandatory STOP。
  P2へ自動進行しない。
- **Codex C/B**：現時点で追加実行なし。既存E[S]と価格identity/未資格状態を維持する。
