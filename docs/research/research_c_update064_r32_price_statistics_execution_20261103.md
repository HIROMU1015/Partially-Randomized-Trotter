# 研究C・update064/r32：独立native256校正による母平均費用差の近似統計検証

2026-11-03 JST。固定科学SOURCE `9837eac59b94113bf057fcc311a27b35e60343f5`・2,374-call契約を無変更で一度実行した。**256/256独立prospective校正が全correctness検査を通過、2,374/2,374 logical calls COMPLETE、FAIL/retry/resume0。主判定は`NOMINAL_MODEL_PR_LOWER`。** 固定実装lawの独立擬似乱数replicationモデルにおける95%近似t区間で、同一update064 J/U強Dに対する条件付き総RZ差は負に分離した。点推定はD比**4.48164%低費用**、比の近似区間は**[.946594634483, .963772600308]**。Validated保証やactual CAの認定ではない。公開照合後mandatory STOP。

[全結果](../../artifacts/research_c_update064_r32_price_statistics/2026-11-03/run01/result.json) ／ [主統計](../../artifacts/research_c_update064_r32_price_statistics/2026-11-03/run01/primary.json) ／ [実行集計](../../artifacts/research_c_update064_r32_price_statistics/2026-11-03/execution_summary.json) ／ [Calls](../../artifacts/research_c_update064_r32_price_statistics/2026-11-03/run01/science_calls.jsonl) ／ [STOP](../../artifacts/research_c_update064_r32_price_statistics/2026-11-03/run01/STOP.json) ／ [postflight](../../artifacts/research_c_update064_r32_price_statistics/2026-11-03/execution_postflight.json) ／ [固定契約](research_c_update064_r32_price_statistics_contract_unapproved_20261103.json) ／ [技術準備](research_c_update064_r32_price_statistics_preparation_20261103.md)。今回のユーザーの明示的な次段階への実行許可に基づく。提案時の`UNAPPROVED_SCIENTIFIC_PROPOSAL` metadataとファイル名は履歴として保持し、承認証明・承認原文保存・新grantは作っていない。

## 1. 固定科学条件・SOURCE

直鎖H4・隣接1.00 Å・STO-3G、通常number CDF rank8、一般signed対称J・実直交U、Nα=Nβ=2/sector36、system8/full256/controlled512。元g補正κ・核scalar一回・元spin/frame順。update064のJ/U、元順LD2、partial ordered symmetric S2、s1/δ=h=.1/r32/K0、m0..12を固定した。Dは**同じupdate064 J/U**のκ＋全8 frame ordered symmetric S2/s1。

共通CA=.00159 Ha、DF reserve=CA/8=.00019875 Ha、既存RPE βtotal=π/6・βstat=π/12、failure=.01/各round軸=.01/26。Own-DF ground-state oracle、状態準備費用除外。価格校正は保存RPE ideal shots/Wとshellに条件付け、RPE/DF/PF/RTE/groundを新条件で再評価していない。

P0はQiskit1.3.0/opt1/seed17、rz/sx/x/cx、approximation degree1、qubits_initially_zero=false、既存Gaussian/MPFR160 declared phase、global phase fitなし。Head相対融合・tail exact basis_key共有、fenced serial会計を維持した。

| 固定対象 | 値 |
|---|---|
| 今回の科学SOURCE | `9837eac59b94113bf057fcc311a27b35e60343f5`、tree `d3054c0ace699eee7ef31bb520510c84bf2bcdd6` |
| SOURCE manifest | SHA256 `8f1864e900f5c1eae5e9453bc8fe4c3c4fdce02852c6f60278be213a31248c5d`、96件 |
| 契約SHA256 | `af507d6961440ece2d11a4fdfb8474f33d3c3bc39b41435bae72e0d608c77883` |
| update064 candidate SHA256 | `53d8f5027ce9e61bc72f6943c0268666bfcf87515154850a9a99326687b26ed3` |
| 候補結果commit/最適化SOURCE | `ea54b9b8ba42ff993d803ec6f5c57e4e622fdee6` / `025d495166311a58ec9c24c43f07ed6e40f65f42` |
| 保存下流結果commit/SOURCE | `cca6b487c8e1ed3362f8825e95c937ebbcc87755` / `928fda0a6df260b66eba1bbea63b51f2734e7b54` |
| h/g/constant snapshot SHA256 | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |
| A統計engineの固定取込み元 | `62fa4c0144b9e36f9d96b7dff67628051946632d`。Bのadapter/引数拡張は今回SOURCEに固定 |

実行branch `codex/research-c-update064-r32-price-statistics-execution-20261103`、専用worktree `/tmp/prt-c-update064-r32-price-statistics-execution-20261103`。準備SOURCE・候補peer・元成果物を変更せず、独立fresh run01へ一度だけ実行した。

## 2. Estimand・固定会計・独立holdout

Xは固定update064/r32/P0の一つのcomplete controlled h-step kernelのRZ数。`μ_impl=E_impl[X]`を推定する。実装lawはuniform64-bit seedからPCG64を初期化する既存law。実際の固定hash seed群を独立擬似乱数replicationとする統計モデルであり、数学的独立性やideal lawとの距離の証明ではない。

`C_PR = 13381920 × μ_impl + 52528`

`C_D = 105038112319`、`Δ=C_PR−C_D`、`R=C_PR/C_D`。

保存ideal shotsは15008、W=13381920、shell RZ=52528、Dは同064の固定kernel価格/費用（D shots13706/kernel RZ11026）。旧update005 Dの105532782860を分母に使っていない。Dは今回の古典的PR校正の第二のランダム標本ではなく、一標本t区間のaffine変換で比較した。Dの科学的/合成/丸め誤差を0と認定したものではない。

固定N=256、32×8組、master2026110303、cohort `update064_r32_prospective_v1_batch00..07`、axis `UPDATE064_R32_PRICE_HOLDOUT_ONLY`、source/candidate/axis/sample/occurrenceへ64-bit seedを束縛。新256 seedsの重複0、005探索128・064探索32・A固定予定640の計800 keysとの重複0。異なるkeyで同じevent列になった観測を棄却していない。旧key照合時のPRNG draw0、seed非重複は独立性certificateではない。

旧update064 native32は探索/設計用、Aのupdate005校正も別law/別候補として、いずれも今回の主推定にpoolしていない。Aの実行中worktree・結果・cache・lockやAの統計結果は読んでいない。r32/有限update064を選定後に固定した一比較であり、全DF/r探索の母平均最小性やfamilywise保証ではない。

## 3. 主解析：95%近似t区間

主解析は固定Nの`xbar ± t_(.975,255) s/sqrt(256)`、信頼水準95%、半幅/D≤1%の登録精度目標。W>0/D>0なので総量・差・比へ同じ端点を変換した。主対象はideal-weight RZ一件だけで、価格alpha=.05をRPE failure=.01と区別する。

| 主統計 | 結果 |
|---|---:|
| 新holdout N | 256 |
| Kernel RZ mean | 7497.4765625 |
| Kernel RZ SE | 34.2338706876 |
| 条件付き総RZ点推定 | 100330684089.25 |
| 条件付き総RZ SE | 458114918.832 |
| 同J/U D固定総RZ | 105038112319 |
| 差Δの点推定 | -4707428229.75 |
| PR/D点比 | 0.955183617395 |
| 総RZの95%近似区間 | [99428513537.4, 101232854641] |
| 差Δの95%近似区間 | [-5609598781.61, -3805257677.89] |
| 比Rの95%近似区間 | [0.946594634483, 0.963772600308] |
| 区間半幅 | 902170551.862 |
| 半幅/D | 0.858898291% |
| 精度目標 | PASS（≤1%） |
| 登録判定 | **NOMINAL_MODEL_PR_LOWER** |

差の上端は−3.80526e9 RZで0未満、比の上端は.9637726で1未満。精度目標と後述のscreenを満たすため、**宣言した近似IID/tモデル内でPR母平均条件付き費用がDより低い、という登録判断**になった。近似区間の被覆率をvalidated guaranteeとは呼ばない。正規IID母集団という理論条件を検証・採択済みとしたものではなく、擬似乱数replicationとt/CLT近似に依存する。

μ_implをideal-law期待価格へ移送する保証、全law synthesis/implemented-law信号誤差、actual CA、実ハードウェア費用、一般的PR優位は別の未資格事項である。価格の区間が負に分離したことだけで、その資格を更新していない。

## 4. 分布screen・データと選択の扱い

| 登録screen | 観測 | 条件と結果 |
|---|---:|---|
| abs moment skewness | 0.0124623791236 | 閾値2以内、clear |
| 最大delete-one mean shift | 6.13146446078 kernel RZ | 値は既存Aの変化量式 |
| shift×W / CI半幅 | 0.0909481768472 | 閾値.25以内、clear |
| 観測分散0 | false | zero-variance保留に該当せず |
| 削除/置換/winsorize | 0 | 全256件を保持 |
| 旧cohort pool | 0 | 新holdoutだけ |

Screen clearは正規性、独立性、未観測tail、CI被覆率の証明ではない。Observed min/maxはsupportではなく、true support・非漸近価格保証はnull。高価格標本を都合よく除去せず、再標本化・追加入力・r変更・追加Nは0。Equivalence/noninferiority marginは未採択のままnullで、この結果から同等性を認定していない。

Calls journalで、**全8組のcalibration完了と全256件validationの後にprimary_statisticsが一度だけ実行**されたことを確認した。途中CI・符号/精度判定なし、データ依存打切り/追加なし。旧32件と今回256件の差の有意性やA/Bの母平均比較は新たに推定していない。

## 5. 価格と副指標：記述だけの集計

同じ校正meanを全13roundへ共有するため、総SEは`W×SE_kernel`であり、√13でさらに縮めていない。r32のmicroeventはcomplete kernel価格内に含まれ、Wへrを重ねて掛けない。256は古典価格校正数であって量子RPE shotsの倍率ではない。各interrogationのhead/X/Y shellを一回計上する既存serial会計を保持した。

| 指標 | 新256 kernel mean | Kernel SE | ideal-weight総量 | 総量の記述SE |
|---|---:|---:|---:|---:|
| rz_count | 7497.4765625 | 34.2338706876 | 100330684089.25 | 458114918.832 |
| rz_depth | 2584.62890625 | 8.57657240101 | 34587349781.125 | 114771005.744 |
| cx_count | 3359.8046875 | 10.5332589464 | 44960637543.75 | 140955228.56 |
| cx_depth | 2337.15625 | 4.56021086477 | 31275637965 | 61024376.9754 |
| total_depth | 5657.14453125 | 18.6910640956 | 75703553097.625 | 250122324.443 |
| circuit_size | 14733.390625 | 69.0544540981 | 197161152224.5 | 924081180.384 |

Assumed raw-signal-budget laneは保存W13447004、shell RZ52766、shots15076を固定し、同じ新meanを入れた記述総RZは`100818650091.84375`、記述SEは`460342996.072`。All-law synthesis≤.001A/law誤差≤.001Aは未検証の仮定のままである。

他5指標とassumed laneには、別の有意差判定・同時CI・全指標支配の主張を付けていない。主判定はRZ/ideal-weightに限る。

## 6. Native correctness・保存物・calls

256件すべてで、固定general-K/κ/scalar、signed law/identity、raw/full-CAR reference、P0 compiled、実際に生成したraw/compiledのadjoint、complex X/Yを既存pipelineで検査した。Controlled512の**全入力列・全出力**を対象に、component allowance1e-10、signal2e-10、rtol0、phase fitなしを維持しすべてPASS。単一trajectoryのcorrectnessから全lawのsignal誤差上界を推定しない。

保存candidate/DF/ground/aliasを同じ候補として再利用し、J/U/rank/LD/s/r/K/compiler/物理条件を変更していない。新ground eigensolve/DF norm/PF-RTE/RPE shots/fit/U更新は0。Science pipelineとcontractは無変更で、失敗標本の置換・retry/resume0。

| 部分 | 実行/上限 |
|---|---:|
| metadata/seed/accounting＋saved candidate準備 | 23/23 |
| 共通full-CAR/Gaussian/shell | 35/35 |
| 8組×（9種×32件＋calibration1） | 2312/2312 |
| 全件validation・主統計・副記述・summary | 4/4 |
| 合計 | **2374/2374 COMPLETE** |

9種はsample/build/reference/raw gate/compile/compiled gate/actual-adjoint pair gate/XY/metrics。全32件×8のevent・sample ID・source/candidate・cohort/seed束縛をpostflightで照合しPASS。量子shots実測0、old pooled0、first_failure=null。統計的未資格でのSTOPとcorrectness FAILを区別する設計を維持したが、今回は精度/screen/符号の登録条件を通過して正常完了した。

新run01 rawは**2859 files / 799194187 bytes**、QPY1024、full256 reference NPZ256。最大file1048862 bytes。Raw/compiled/actual-adjoint QPY、reference、events、metrics、source/contract、Calls、STOPを保全した。結果を出した後の新条件での統計再解析・配列decode・科学callは0で、以降はJSON出力の報告整理とbytes/Git照合だけを実施した。

## 7. 資源・実行前後の保全・非干渉

内部wall `3620.74615579` 秒、parent CPU `60140.8891212` 秒、peak RSS `2493068` KiB。外側timeは**1:00:21、CPU1660%、swap0、exit0**。[実行log](../../artifacts/research_c_update064_r32_price_statistics/2026-11-03/execution.log)。CPU32 affinityで各native32を並列化し、BLAS1のprocess-local調整と既存Qiskit排他を維持。8組は同一mutable builder/cacheを共有するため逐次で、価格を見た逐次選択ではない。CPU/RAM/worker/wallの人工上限を追加せず、sourceを実行途中に変更していない。速度比の主張はない。

開始前のread-only process照会ではA runnerは可視範囲になかった。これはAの結果を確認した意味ではなく、A liveworktree/output/cache/lockや統計結果は参照していない。CPU pressure/memory pressure/空き資源を確認し、実行時空きメモリ約52GiB、途中も約50GiB・memory pressure0。GPUはdriver通信失敗のため固定CPU/SciPy/Qiskit経路を使用。共有venv/global設定・他jobの停止/凍結/変更0。

SOURCE96件・local input16件・ea54 peer3件と契約hashを実行前後で照合し一致。run内SOURCE/contractコピーも固定原本と一致。元科学SOURCE/入力/raw/契約/grant/ledger/marker/STOPは変更0、peer HEAD/clean状態も保全した。旧raw全hashや旧価格再解析を繰り返していない。準備の82 unique人工/回帰PASSは固定SOURCEの技術証拠として参照し、今回科学実行時に再テストしたとは記載しない。

共有索引・Evidence Map・研究概要・validation manifestは、元依頼どおり後続統合担当へ引き渡す。本書の近似価格meanの資格を、理想law/actual CAの証拠資格と一緒に更新しない。

## 8. 判断範囲・次の担当・STOP

新規独立holdout256件は、**固定update064/r32実装lawの価格母平均を保存ideal RPE会計へ接続した条件付き一比較について、同J/U強Dより低価格という近似モデル内判断を支持**した。これは単なる旧32件の再集計ではなく、旧cohortと分離したprospective校正である。一般的な最適DF/最適r、目的関数の因果効果、分子間汎化、全law合成/実装law誤差、理想law母平均への移送、validated CI、actual CAは未認定。

**ユーザー／研究C**は、このnominal価格検証と残るsignal/law/実CAの資格を区別して、研究上の次の判断と正式レビューの要否を決める。**統合担当**は、SOURCE・固定条件・近似区間・未資格事項を伴ってEvidence Mapへ接続する。Codexは追加校正・r探索・J/U更新・科学レビューを自動開始しない。**今回の固定実行は完了。公開SHA/tree・本文・source/input/旧成果物照合後mandatory STOP。**
