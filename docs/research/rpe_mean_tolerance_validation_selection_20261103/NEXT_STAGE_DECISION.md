# 次段判断：mean予算の局所残余と決定論的周辺平均の照合

2026-11-03 JST。**今回は設計まで完了。新しい科学実行・数値再解析・重要科学レビューは開始しない。**
V2のCONDITIONAL_SUCCESS、run02のUNDETERMINED、R1費用順位UNDETERMINEDを維持する。

## 1. 今回の結論

1. 必要なmean精度を恣意的な数値には固定しなかった。既存N/α/epsilonを保持する十分条件は
   `tau + u_num < min_axis[epsilon − sqrt(2 log(2/alpha)/N)]`。
   これは統計誤差へ割り当て済みのepsilonの残余を使う条件であり、唯一の必要条件・最大許容差ではない。
   具体の残余・tau・追加nは未計算。
2. 第一候補はPartial m17/q131072/r32/K2の両軸。保存radius減衰、未検証r32、累積normalization、
   最終energyへの関与を合わせて選んだ。全roundの新スコアや確率を計算した順位ではない。
3. 推奨はsamplerの宣言lawから作る決定論的平均と、係数/Taylor参照・保存meanの局所照合。
   小さい差を識別する前に大量samplingを追加する案は推奨しない。
4. 比較用候補は最初のr-transitionであるm8/q256/r2/K2。第一案に自動追加しない。
5. 8人工tests PASS。実H4の新sampling/state-action/演算子構成・raw再解析・compileは0。

詳細：[精度要件](RPE_MEAN_ERROR_REQUIREMENTS.md)、[round選定](ROUND_PRIORITIZATION.md)、
[方式比較・全supportの平均式](VALIDATION_METHOD_COMPARISON.md)。

## 2. 提案する最小科学検証

### 問い・固定入力

「保存physical stateと既存selected m17 tupleについて、samplerのorder/component確率・
event phase/作用順から直接平均したcomplex信号は、保存attenuated event-meanと一致するか。
その数値的な差を、既存RPEの固定N/α/epsilonの残余に照らしてどう解釈できるか。」

linear H4/隣接1.00 Å/STO-3G/legacy DF12、8 qubit/4電子sector70、L_D3、δ=.02、
q131072/r32/K2、extract_identity_phase、coefficient_atol=1e-12。
stateはBの保存physical ground stateをそのまま使う。新しいE0・固有状態・DF表現を生成しない。

| 入力 | 固定参照 |
|---|---|
| H snapshot | carrier4591b8d75a2a6a55c74a51225f0c61648023c8eeの`artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz` |
| H file SHA256 / H hash | 13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b / 56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5 |
| 保存state | B commit e22d7718c47358ff650dbc04e8e72f6ab1c8bf41、`artifacts/rpe_fixed_request_measurement_b/2026-11-03/run01/physical_state.npz` |
| State file SHA256 / fingerprint | 793300c86417f52569289ce86980c5f9a87292e130199355a0f4f7056225c9c4 / a07d6c7fdbbb23a035f9fc54b9956ee79e557a2185e29b1870748b771958be90 |
| 対象μ・N/α/β/epsilon | V2 commit78c49d0b5f1d71e63b5959e78813237923bbc648のCSV、ld3/round_index17の1行とその両軸 |
| Preparation | run02と同じhash89145d8324bf139eaec67265f3134a5c1a3cebb7ed97792d54ed3221d31f8919、218 retained component |

保存N411/axisはRPE設計の入力であり、この決定論的検証のshot生成数ではない。
alpha=.016957089147912167/axis、epsilon=.15244325851620388、保存βは元行のまま。
この検証のsampling・RNG draw・仮想shotは0で、seed/nの新選定は不要。

### 結果を見る前に固定する手順とcall範囲

1. **保存scalarの局所再解析**：この1行・2軸だけでHoeffding半幅s_aと残余R_17を計算する。
   出力は保存epsilon、s_a、R_17の符号・値。μを使って都合のよいtauを後付けしない。
   この演算自体が新しい科学的再解析なので、今回実施していない。
2. **数値contextを1組準備**：保存H/stateをload・照合し、DF preparationは1回。
   保存218成分から、probability重みと係数重みのsector平均Aを各1個、F/Rを各1個構成する。
   basisごとにgroup化/cacheし、同じbasisを成分ごとに再構成しない。
3. **独立した集約式を比較**：実際のorder_probabilitiesとcomponent再正規化を読む式(3)のM_lawを1個、
   既存paired-Taylor参照を1個構成する。tau=lambda_R*δ/32、内側r32・外側q131072を固定し、
   最終complex期待値を両経路各1個求める。Re/Imを両軸の結果として読む。
4. **比較を記録**：law対Taylor、各経路対保存μの座標別・complex残差、数値identity、
   数値不確かさの扱い、R_17との関係を記録する。新しいenergy成功率や費用を計算しない。

この範囲は1 tupleであり、m8・Full・残り17roundの新mean計算・新しいalias/branch確率評価を含めない。
H/state新生成・対角化・PF fit・r/K/N/α/β再最適化、sampling、controlled/Hadamard build、compile、価格再計算は各0。
DF preparation内部のGaussian gate定義とsectorの数値operator/state-actionは必要な科学callとして明示する。

### 精度と判定

初期数値形式は既存sourceと同じcomplex128/float64。
保存matrix比較の`atol=1e-10`を**数値再現の診断**として継承する。
これは新しく導いたRPE許容mean差でも、厳密なu_numでもない。
長qの丸めを一stepの許容差のまま無条件で移送せず、最終mean残差を報告する。
高い精度が必要なら、閾値判定・符号・研究結論に影響する場合に限って技術的に精度を検討し、
条件を凍結した新版で比較する。全面的な高精度/interval基盤はpilotの前提にしない。

| 出力判断 | 事前規則と解釈 |
|---|---|
| NUMERICALLY_CONSISTENT | 両経路と保存μが既存1e-10の診断内。選択セルの数値接続の支持に限る |
| INCONSISTENT | identityが正しい条件で診断を超える差。原因調査対象であり、元V2の保存モデル計算を消さない |
| UNDETERMINED | 入力不一致、非有限、数値不確かさ不足、残余非正、残余との比較を信頼できない場合 |

RPEに十分という追加解釈には、`observed_difference + u_num < R_17`、
u_numの根拠、既存V2の半径/固定格子/入力/IID/alias等の条件を明示する。
float64二経路の一致や、二つの精度の差だけをrigorous u_numにしない。
u_numを条件付きの数値仮定に留める場合は、その条件付き性を出力へ残す。
R_17が小さい／非正でも、診断をPASSへ寄せるためNやepsilonを変更しない。
保守的残余で結論が出なければ、robust二項領域の別検証案を返す。

### 性能

主要workloadは小さいsector行列の構成・powerで、全trajectory列挙ではない。
H/stateを共有しbasisとAを再利用、少数の独立経路は必要なら並列化し、matrix powerは反復二乗を使う。
起動時にCPU/GPU/RAMと共有負荷を確認してbackend/threadを選ぶ。恣意的なCPU/RAM/wall-time上限は置かない。
一つの小さいtupleなので、GPUや多数processのstartupよりdense BLASが適切な可能性もある。
実測なしに速度倍率は主張しない。共有環境・H8/他ユーザーjobを変更しない。

## 3. 再利用と必要な最小実装

| 既存箇所 | 再利用／差分 |
|---|---|
| run02 `run_fresh_iid_a_v2.py` | H/state/sector/基底順序・numeric half・identity検査。contextのr1固定をr32入力へ一般化する必要がある |
| `rte.py` | RTEFiniteDistribution、finite_taylor_operator/operator_moments、component/eventの型・規約。literal-law平均は独立した短い追加部分 |
| `finite_rte_signal_validation.py` | basisごとのtail group化とsector permutation。validator全体はH/eigensolveを行うので呼ばない |
| V2 `run_saved_v2.py` | 保存tuple/予算の取得形式、必要なら後続二項検証の再利用候補。全V2解析を起動しない |
| 元trajectory/measurement adapter | SOURCEの作用順と±1平均規約の対応を読む。今回はsamplerを呼ばない |
| `test_design_identities.py`、既存test_rte等 | 小さいsupport・非可換toyで新集約と旧参照の対応を確認。分子fixtureの一括実行は不要 |

`research_direction_full_opt2_extension_analysis.py`はgated cost再最適化のpipelineであり、
mean検証のrunnerには流用しない。共通の保存入力の系譜だけを使う。
[V3](../pr_rpe_v3_cost_transfer_preflight_20261103/REPORT.md)、
[R1](../pr_rpe_v3_r1_shared_calibration_sensitivity_20261103/REPORT.md)、
[MINIMAL_NEXT_ACTION](../pr_rpe_v3_r1_shared_calibration_sensitivity_20261103/MINIMAL_NEXT_ACTION.md)の
価格共有誤差・長q外挿は別問題で、R2/R3再開の根拠にならない。

## 4. 既存研究判断への影響と担当

一致すれば、選択したm17の「保存μが宣言されたsampler lawの周辺平均」という条件を数値的に支持できる。
式(2)の残余との比較も成立すれば、固定N/αのそのroundについて、追加系統差が十分小さいという条件付き接続を強められる。
一つのroundの一致では全18roundの実装平均、物理状態準備、全controlled回路、実backend/noise、費用順位を認定しない。
不一致ならV2の保存平均に対する計算は保持し、その測定実装への適用条件を未解決として切り分ける。
元V1/V2/V3/R1・B・A run01/run02のSOURCE/raw/statusを変更しない。

**次担当案**：Codexが本計画の最小runnerと人工テストを準備して公開SOURCEを固定し、
ユーザーがそのSOURCEと上記1 tupleの保存scalar再解析・DF/sector平均作用評価を明示許可する。
現在の設計依頼は、その新しい科学実行の許可ではない。実行許可の証明書・grant・台帳は要求しない。
重要な科学レビューの開始は独立した許可事項である。

許可後の意図しない技術的中断は、既存ユーザー指示の範囲で原記録を残して修正・再実行できる。
正常完了後の不一致・未判定を理由に、対象・seed・N・閾値を変更して試行を繰り返さない。
科学結果の保存・公開照合後はmandatory STOP。別round、追加sampling、robust確率評価等は自動開始しない。

## 5. 今回の完了状態

4文書と8人工testsを作成し、run02/V1/V2/V3/R1/Bの公開SOURCE・保存値・既存実装を技術調査した。
原データは文字列/metadataの引用だけで、新しい残余・確率・branch・費用の数値評価はしていない。
既存manifest・科学source・結果を変更せず、関係する文書索引だけへ新しい計画の入口を追加する。
この推奨を研究の正式採択・独立科学レビューの完了と扱わない。
公開branchのremote SHAと成果物を照合して今回の作業を終了し、mandatory STOPを維持する。
