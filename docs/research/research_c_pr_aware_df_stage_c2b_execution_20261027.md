# 研究C：PR-aware DF C2-B・保存確率表の限定科学実行

2026-10-27 JST。Repository：`HIROMU1015/Partially-Randomized-Trotter`。

**固定SOURCEを変更せず、fresh C2B-run01を一度実行した。35/35 logical units COMPLETE、correctness FAIL・retry・resumeは0。** 両armの216 atom表が既存`make_tail`でexact再生され、IID一様53-bit格子モデル下のcategorical TVと全26roundの条件付きraw-law上界を取得した。PCG64のIID性、CAR/atom/角度・normalizationのreference bridge、全law P0 synthesisと費用supportは未資格のまま。Actual CA、保証shots／total、C1 evidenceは更新していない。

## 1. 固定SOURCE・契約・入力

| 項目 | 固定値 |
|---|---|
| 科学実行commit | `3c1bb2d9ce46c29205594a42b090c0811fbafd27` |
| Branch | `codex/research-c-pr-aware-df-stage-c2b-execution-20261027` |
| Scope | `C_PR_AWARE_DF_C2B_K0_TABLE_IID_GRID_V1` |
| 契約 | [固定C2-B契約](research_c_pr_aware_df_stage_c2_contract_draft_20261027.json) |
| 契約SHA256 | `735ee857e549abd086d5085120cddc023708fec71329886921eb2a0dd3aa0150` |
| SOURCE manifest SHA256 | `2cfb05ea3764c220fdc5ffdc3eae1f1bc54bfc107b45a6ffacac0c6a09e3b45b` |
| 保存Baseline tail JSON SHA256 | `c1ead156a101a77470d005c3fd5e9a231470d342c088a944bca015421c57e81c` |
| 保存Proposed tail JSON SHA256 | `b8ac37bb622982d8347a8b52e8ba03cf2177b2b112d57e7af9cfc706baf0e72c` |
| 出力 | [C2B-run01原本14ファイル](../../artifacts/research_c_pr_aware_df_stage_c2/2026-10-27/C2B-run01/) |

保存直鎖H4・隣接1.00 Å・STO-3G、通常number CDF8のBaseline／Proposed、Nα=Nβ2／sector36、元順LD2、元g補正κ・核scalar一回。Partial symmetric S2、δ=.1／s4／h1/40／r16／K0、m0..12、own-DF ground oracle／状態準備費用除外、共通Gaussian/P0を維持した。CA=.00159 Ha／DF reserveCA/8、既存phase/failure条件はmetadataとして継承し、今回その適格性を再評価していない。

ユーザーのチャット上の明示許可に基づいて実行した。歴史的なdraft名・`UNAPPROVED_SCIENCE_CONTRACT_DRAFT` metadata、旧契約・技術STOPは変更せず、承認証明書・承認原文保存・新grantを作成していない。元入力とJ/Uはhashだけの照合で、NPZ/QPYはdecodeしていない。

## 2. 全gate・call実績

| 分類 | COMPLETE |
|---|---:|
| saved load | 1 |
| C1 SOURCE／status identity | 1 |
| table replay | 2 |
| choice CDF/grid/TV analysis | 2 |
| 条件付きraw-law round：2arm×13 | 26 |
| arm qualification／null C1-preview | 2 |
| output summary | 1 |
| **合計** | **35** |

[Journal](../../artifacts/research_c_pr_aware_df_stage_c2/2026-10-27/C2B-run01/science_calls.jsonl)のattempt35件とCOMPLETE35件はkind/labelごとに一対一。FAIL・inflight0、retry/resume0。[STOP](../../artifacts/research_c_pr_aware_df_stage_c2/2026-10-27/C2B-run01/STOP.json)はfailed=false／mandatory_stop=true。Overall結果は`C2B_CONDITIONAL_IID_TABLE_ANALYSIS_COMPLETE`で、全実装誤差のGOという意味ではない。

SOURCE／契約／保存入力・C1 provenanceはPASS。保存216 atom/armのID・係数・符号・basis・component probabilityの再生が両arm PASS、K0 order=[0]／probability=[1]と登録条件もPASS。両表ともpositive ideal atomのgrid mass消失は0件。これはこの保存表・格子モデルの結果に限り、別分子や未登録表のunderflowが一般に起きないという主張ではない。

新fit／optimizer／rank探索／追加sampling／回路build-compile／DF-PF-RTE operator計算／shots・総資源再計算は0。NumPyの`choice(size=0)`による受理検査では乱数wordを消費していない。

## 3. Table TVと最大roundの結果

| arm | TV η（上向きbinary64表示） | m12の条件付きraw complex-signal誤差上界 |
|---|---:|---:|
| Baseline CDF8 | 3.647073528232258e-15 | 1.912116885969834e-9 |
| Proposed CDF8 | 3.226176237345202e-15 | 1.6914454871252413e-9 |

ηのexact値も保存した。

- Baseline：`83691142786380208221/22947478886433488248878099997917184`
- Proposed：`24742425482656375478151/7669272743455809267621450291594919936`

対象は**保存binary64由来atom係数をexactに正規化した分布P**と、固定NumPy1.26.4のCDF写像へIID一様53-bit wordを与えた分布Qの距離である。元の分子積分に対する確率誤差、実PCG64のjoint law、B1 raw CAR referenceとの差をすべて含む距離ではない。

第一確率丸め、sampler再division、連続CDF、finite gridの各段階TVをexact rationalで別記した。第一確率表→sampler表の正規化済み分布間TVは両arm0だった。直接P対Qのηを使用し、段階TVをさらに足して二重計上していない。[2表の結果CSV](../../artifacts/research_c_pr_aware_df_stage_c2/2026-10-27/C2B_tables2.csv)／[全値とexact rational](../../artifacts/research_c_pr_aware_df_stage_c2/2026-10-27/C2B-run01/result.json)。観測標本から推定したTVではなく、宣言したCDF／格子写像の有限区間計数である。

## 4. 全26roundの条件付き上界

各eventがIIDで、両lawに同じexact unitary trajectory mapを使い、matrix elementの絶対値≤1であることを条件とする。登録した式`min(2,2*n_m*η)`、`n_m=16*4*2^m`を使用した。ここでn_mはkernel回数ではなく総event数。m12では262,144 event。Attenuation Aを掛けたり割ったりしていない。Synthesisは別項のまま。

下表は表示桁を丸めた値。上向きbinary64値とexact rationalの全桁は[26round CSV](../../artifacts/research_c_pr_aware_df_stage_c2/2026-10-27/C2B_law_rounds26.csv)に保存した。

| m | event数 | Baseline条件付き上界（概数） | Proposed条件付き上界（概数） |
|---:|---:|---:|---:|
| 0 | 64 | 4.6682541161e-13 | 4.1295055838e-13 |
| 1 | 128 | 9.3365082323e-13 | 8.2590111676e-13 |
| 2 | 256 | 1.8673016465e-12 | 1.6518022335e-12 |
| 3 | 512 | 3.7346032929e-12 | 3.3036044670e-12 |
| 4 | 1,024 | 7.4692065858e-12 | 6.6072089341e-12 |
| 5 | 2,048 | 1.4938413172e-11 | 1.3214417868e-11 |
| 6 | 4,096 | 2.9876826343e-11 | 2.6428835736e-11 |
| 7 | 8,192 | 5.9753652687e-11 | 5.2857671473e-11 |
| 8 | 16,384 | 1.1950730537e-10 | 1.0571534295e-10 |
| 9 | 32,768 | 2.3901461075e-10 | 2.1143068589e-10 |
| 10 | 65,536 | 4.7802922149e-10 | 4.2286137178e-10 |
| 11 | 131,072 | 9.5605844298e-10 | 8.4572274356e-10 |
| 12 | 262,144 | 1.9121168860e-09 | 1.6914454871e-09 |

全26行は`CONDITIONAL_IID_TABLE_BOUND`。Raw complex-signal単位であり、Ha、normalized signal誤差、DF誤差、PF誤差、合成誤差と混同しない。未知の合成誤差を0へ置き換えず、C1のshotsや精度条件と組み合わせた再判定もしていない。

## 5. 資格が得られた部分・残る未資格

| 項目 | 今回の状態 |
|---|---|
| 保存atom→componentの既存処理の再生 | 両arm exact PASS |
| 固定CDF境界と53-bit格子のbin確率 | Exact rationalで取得。SOURCE／library意味とIID一様wordモデルが条件 |
| P対Qのcategorical TVと26round伝播 | 条件付き上界を取得 |
| 実PCG64 streamのIID性／shot・occurrence間の独立性 | 未証明。Seed分離を証明としない |
| Atom係数とB1 raw CAR tail、λ/tau/atan/normalization/phaseの対応 | 誤差enclosure未取得。今回operator再評価なし |
| Native emitter・whole-kernel P0・shellの全law synthesis | 未資格。有限標本最大残差で埋めない |
| 全law費用support／母平均CI | 未資格。旧32標本/armは増やしていない |
| C1 evidence／actual CA／保証shots・total／資源優位 | UNQUALIFIED／nullを維持 |

C1-previewでは各armの13roundすべてのactual synthesis/law入力がnullで、`error_evidence=None`、guaranteed shots/totalもnull。C1の保存約83%削減を、今回の結果だけで実装同CAの保証付き優位へ昇格させない。B1／DF／referenceは元のbinary64 numerical pilotの資格のまま。

今回の表レベルのroundoff/grid上界は得られたが、それを採択済み全実装law boundとして旧C1へ注入していない。PCG64とIIDモデル、reference bridge、P0 synthesisはそれぞれ独立に残っている。大きい表上界が出た場合の追加探索や条件変更も行っていない。

## 6. 環境・資源・保全

[実行前preflight](../../artifacts/research_c_pr_aware_df_stage_c2/2026-10-27/C2B_execution_preflight.json)で固定commit・SOURCE76・保存入力12・fresh output不存在と依存版を照合した。Python3.11.1、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、threadpoolctl3.6.0、symengine0.13.0。既存venvをread-onlyで使用し、task-local cacheを使用した。

CPU affinity32、MemAvailable約52.7 GiB、CPU/memory pressure avg=0。登録2表を既存resource_planの2 taskで並列処理し、CPU/RAM/worker/wallの人工上限は設けていない。GPUや大行列を必要としない有理数と有限表の処理であり、speedup測定はしていない。共有venv/cache・他job・OS設定は変更していない。

Runner wall `1.8606958873569965 s`、CPU差分 `10.083745888 s`、peak RSS `356820 KiB`。[Resource記録](../../artifacts/research_c_pr_aware_df_stage_c2/2026-10-27/C2B-run01/resources.json)。出版・remote照合等を含む全作業時間ではない。

[事後照合](../../artifacts/research_c_pr_aware_df_stage_c2/2026-10-27/C2B_execution_audit.json)でSOURCE76／保存入力12、旧SOURCE・script・PR-aware証拠1768ファイルのbyte保全を確認した。原本14件のSHA256、35callの完了対応、2表／26roundのschema、actual nullと禁止処理0を記録した。旧A/B1/B2/C1とC2技術STOP、grant/ledger/markerを変更していない。CSVは既存出力のserializeだけで、TV/PF/RTE/shot/resource関数を再実行していない。

[準備時の人工99 PASS](research_c_pr_aware_df_stage_c2_qualification_readiness_20261027.md)は同じ固定SOURCEの保存テスト結果であり、実行前後に科学SOURCEを変更していないため重複実行しなかった。今回の実データgateと事後照合はすべてPASS。

再現用command（消費済みrun01へ再実行しない）：

```bash
env PYTHONDONTWRITEBYTECODE=1 \
 XDG_CACHE_HOME=/tmp/prt-c-pr-aware-df-stage-c2b-execution-20261027/.cache \
 MPLCONFIGDIR=/tmp/prt-c-pr-aware-df-stage-c2b-execution-20261027/.cache/mpl \
 /tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python -B \
 scripts/run_research_c_pr_aware_df_stage_c2.py \
 --contract docs/research/research_c_pr_aware_df_stage_c2_contract_draft_20261027.json \
 --output artifacts/research_c_pr_aware_df_stage_c2/2026-10-27/C2B-run01 --execute
```

## 7. 次の最小作業とSTOP

固定H4・CDF8二arm・LD2・δ.1/s4/r16/K0の保存表に対し、登録IID格子モデルのcategorical law上界を取得できた。この限定範囲の計算は完了した。

次は**研究C／ユーザー**が、残るwhole-kernel P0 synthesisとCAR/atom/angle/normalization reference bridgeのどちらの最小技術調査を先に進めるか、またIIDモデルを条件付き研究モデルとしてどの範囲で使うか判断する。**Codex**は具体的な次指示後に既存SOURCE／proof候補を再利用して対象を絞り、新しい実分子数値評価は別に固定した範囲への明示許可後だけ行う。追加samplingやC1再解析を自動で開始しない。

本報告は固定契約の実行と証拠整理であり、重要な詳細科学レビューやC2-A/Cの採択・実行を含まない。結果・SOURCE・入力・call/STOPをcommit/pushし、remote SHAと成果物を照合した後に**mandatory STOP**する。
