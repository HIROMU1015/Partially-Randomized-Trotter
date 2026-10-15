# H6/H8 Native DF：metadata補完とモデルA/Bの実データ評価

利用者の「実行してよい」「続きを行って」に基づき、固定208候補の理論費用を評価した。
H8の7 preparationは過去の保存hashと全件一致し、H6は保存archiveの7 rootを再利用した。
これは固定分解・有限lawの技術評価であり、新しい科学的採択、母平均の保証、原稿改訂ではない。

[結果・全表・binding入口](../../artifacts/resource_applicability/track_a_native_df_theory_evaluation/2026-10-16/README.md)。
旧[metadata受領報告](track_a_native_df_h6_h8_metadata_completion_20261016.md)と未認可draftは当時の記録として保持した。

## SOURCEと対象

- 起点commit `b0c7a2855482175e438ffebe6c821ce36b0b2860`。独立branch `track-a-native-df-theory-evaluation-20261016`。
- 実行SOURCE `7e59df8ac3c356007d85bfd9218ce1a224a40d52`。source/testsを先にcommitし、その40文字SHAと6 source blobs、265 provider files、39保存入力を実行manifestへ固定した。
- 科学providerはH8の `c31ef424175c02379eafebe8fc6c7548c1d9aabe`。H6の `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7` と共通6 sourceがbyte一致することを別receiptで確認。両SOURCE全体の同一性は主張しない。
- 理論counter `native_df_gate_model.py` は既存 `d2431b8c2a385af20a3d996cf8074b7a826a6e6b` から変更0。

|対象|H6|H8|
|---|---|---|
|系・geometry・basis|linear H6 / 1 Å / STO-3G|linear H8 / 1 Å / STO-3G|
|spin modes / sector / DF rank|12 / 400 / 19|16 / 4900 / 28|
|T / prefixes|.8 / 5,10,15、full19|.8 / 7,14,21、full28|
|登録候補|旧92＋B0I12＝104|旧92＋B0I12＝104|
|snapshot SHA|`99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005`|`d4c4346127e97c241439ae1c4ae010962b0456b19f461cfd3d9b06e7d4a50955`|
|preparation処理|archive `ad52f9f6…`を記述readerで読む。再生成0|保存snapshotからfaithful4＋discard3を補完。旧hash7/7一致|

q=1,2,4,8、PRのr=1,2,4・K=2,4・R=qr、S2/full S4は既存登録集合のまま。
H6の探索8・独立確認32/union8、H8の探索8・独立確認32/union6、B0I実測を別scopeへ接続した。
[保存candidate/SOURCE/paired group索引](../../artifacts/resource_applicability/track_a_native_df_metadata_completion/2026-10-16/saved_candidate_connection_index_v1.json)のphase・execution_sources・未確定seed sourceは元のまま保持した。
全候補で同じ保存wrapperの数値fingerprintを再構築した検査ではない。

## 結果

- 208固有候補 × A/B × cosine/sine＝**832理論wrapper行**、PR finite law **144件**。
- 既存5精度×factor1/10のshot数を導出し直さず、同一候補の条件・metricへ接続した **10,480行**。不適格shotと未測定compiled費用は欠測のまま保持した。
- 新SCF/DF分解/state solver・signal/reference・trajectory sampling・native full wrapper生成・transpile/compileは0。H8のpreparation/Gaussian記述補完だけを認可範囲で行った。
- 旧結果の費用平均・shot・選択済みwinnerを更新しない。新PR/PF比・候補ranking・paired SEの導出は行っていない。

固定した参考6候補のcosine one-shot RZは次の通り。費用順に選択した表ではない。
PRはsupplied finite lawの期待値、deterministicは固定分解の命令数。

|系|候補|モデルA RZ|モデルB RZ|保存compiled RZ（標本点平均）|保存scope|
|---|---|---:|---:|---:|---|
|H6|`B0I_p10_q1_2nd`|35,054.000000|33,446.000000|20,292.000000|H6_B0I_FAMILY12|
|H6|`B1_p19_q1_2nd`|63,395.000000|61,787.000000|37,561.000000|H6_EXPLORATION92|
|H6|`B2_p10_q1_2nd_r1_K4`|36,568.077235|36,566.884389|22,004.875000|H6_EXPLORATION92|
|H8|`B0I_p14_q1_2nd`|88,004.000000|85,092.000000|53,448.000000|H8_B0I_FAMILY12|
|H8|`B1_p28_q1_2nd`|168,850.000000|165,986.000000|104,229.000000|H8_EXPLORATION92|
|H8|`B2_p14_q1_2nd_r1_K4`|90,822.281607|90,818.538775|56,601.250000|H8_EXPLORATION92|

A/Bはcompiler非依存の `explicit_rz_sx_x_cx_v1` 分解であり、Qiskit optimization1の出力と一致する量ではない。
state preparation・measurement・routing・回転のFT合成費用を含めない。保存RZ/CXと比較する際はこの範囲差を残す。
深さやT/Toffoli数の理論値は算出していない。

## 期待値の計算と意味

元のfinite-RTE lawで `tau=lambda_r*(T/q)/r` を評価し、component確率と全偶数次数確率をbinary64のまま取得。
その値をFractionへ写して厳密に正規化し、正規化係数も全件保存した。
これは与えた数値lawの条件付き期待値であり、超越関数や真の確率に関する誤差証明ではない。
希少なK4次数も省略していない。一般identity位相は従来の **0〜1 P / closed occurrence** の幅を維持した。
実catalogでその幅が縮む場合も、元counterの規則だけを用いる。

DPは同じ `(basis_id, source_hash)` の確率質量とcentral costの線形和をまとめる。
分布、符号、identity barrier、閉じたoccurrence、phase条件は変えない。
元DPとの全次数・shared basis・identity・signed component・rare order等の人工11件が一致した。
100万grouped transition capを保持し、近似fallbackは0。今回の最大遷移数は14616。
この上限は**実行したgrouped遷移数**であり、元のcomponentごとの遷移数とは同じ会計ではない。

モデルAはGaussian再利用なし。Bは厳密に同じbasisの隣接境界を相殺する仮想構成。
実nativeはPF prefixがA相当・RTE tailがB相当の混合政策である。
均一Bを実compilerで実施済みの高速化と扱わず、今回の均一A/B表をnativeの完全な費用予言とも扱わない。

## 保存compiled値への接続と残る制約

各候補・scope・精度・factorの既存 `N_real/N_imag` を使い、
`G_theory=sum_axis N_axis E[C_theory,axis]` をRZ/CXについて評価した。
`G_compiled` は旧D0の保存axis meanから再表現し、保存 `G_exact` との一致を検査した。
B0Iでは原SOURCEの保存2軸metricを参照した。新しいcompile平均を計算していない。

5,736行が保存compiled値まで接続され、残り4,744行は保存shot条件が不適格。
条件別状態は [summary](../../artifacts/resource_applicability/track_a_native_df_theory_evaluation/2026-10-16/result_summary_v1.json) に保存。
全行を `CONDITIONALLY_ALIGNED_METADATA_NOT_FULL_NUMERIC_ENVELOPE` とし、過剰な同一性認定を避けた。
H6は旧basis variantとcurrent archiveの違い、元profile、seed sourceの未確定行を保持する。
代表metadata一致を全探索wrapperのbit同一性へ拡張しない。
理論期待値と探索8/確認32のcompiled標本平均の差には、構成・compiler・有限標本の差が含まれる。
母平均の優劣や希少eventの費用保証、科学的採否は今回認定しない。

## 資源・検証・保全

既存private venv（Python3.11.0rc1 / NumPy1.26.4 / SciPy1.14.1 / Qiskit1.3.0 / Nature0.7.2）を変更せず使用。
旧full profileの固定receiptと、現在のcompiler/profile digestを保持し、freshには15重要fileのbytes/SHAとversionを確認した。
34,273 fileの全hash campaignは反復していない。旧compiler出力のbit同一性は未認定のまま。

1計算process/内部thread1＋stdlib observer1、nice10、CPU affinity変更0、AS16GiB/RSS4GiB、observer soft AS512MiB、headroom16GiB込みadmission32.5GiB。
phase/total wall capはnull。出力128MiB・log各16MiB、disk reserve2GiB/inode4096、own exclusive output/controlとone-shot lockを使用。
保存shotへの接続処理も別のown-child監視下で一度実行し、1.003秒でexit0。
終了時は両childのwait/reap・pidfd/pipe closeを確認した。自動retry0・次stage0。
wall 1230.366 秒、peak RSS 519.02 MiB、最大観測interval 1.003676 秒、OOM増分0。
CPUの予約は行わず、許可affinity内でOS schedulerを使用した。GPU query/use・CuPy import0。
quota binaryは利用不可、/homeはext4でquota mount optionなし。利用者quotaの独立証明は未取得として記録した。
fresh FS/inode・有効memory・host/cgroup pressure/OOMを確認し、計算中も1秒間隔で別processが監視した。

限定人工17件PASS（grouped DP11＋資源gate/保存会計6）。旧42件等のcampaignは反復0。
主担当の実装・技術照合であり、別担当の独立reviewと称していない。サブエージェント0。
旧source/results/grant/ledger/carry、既存rootのdirty/untracked、共有環境・venv・他job・原稿を保全した。
NPZ・raw runtime/cache・private絶対path・credential・内部SSH情報は公開しない。

## 公開とSTOP

source/tests commitと結果資料commitを分離。公開対象だけを明示stageし、non-force pushを試す。
失敗時は認証設定を変更せず、ローカルcommitと実行可能なpush commandを報告する。
資料公開またはpush待ちの報告でmandatory STOPし、追加計算・レビュー・科学的採択へ自動進行しない。
