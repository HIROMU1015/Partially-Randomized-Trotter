# RPE候補A：fresh-IID測定接続の実装準備

2026-11-03 JST / LOCAL。**adapter・限定runner・人工テストを準備した。
今回、実H4のsampling/state-action/仮想shot、保存signalの科学的再解析、回路build/compileは実行していない。**
科学実行には[固定計画](SCIENTIFIC_EXECUTION_PLAN.md)の別承認が必要である。

## 既存証拠と再利用

入口PROJECT_MAP、LOCAL AGENTS、共通指示2026-10-25の§8.1・§8.4・§10を参照した。
共通指示はLOCAL元checkoutにあり、この親treeにあるGit固定文書と偽って扱わない。
前段の研究概要・規範資料と、以下の固定後続資料を区別して用いた。

| 資料 | 固定版・役割 |
|---|---|
| [候補B報告](../rpe_fixed_request_measurement_b_20261103/REPORT.md)／[scope](../rpe_fixed_request_measurement_b_20261103/EXECUTION_SCOPE.md)／runner/test／RESULT | 結果e22d7718c47358ff650dbc04e8e72f6ab1c8bf41、科学SOURCE4d7c455db3ed7d3e4bc9e2e1fd4dbeba9f46b272。指定6request・12軸の条件付き平均だけの証拠 |
| [既存接続計画](../rpe_18round_measurement_connection_20261103/MINIMAL_SCIENTIFIC_VALIDATION_PLAN.md) | e22d7718c47358ff650dbc04e8e72f6ab1c8bf41。Aは保存q1/q8、各軸N510/N454の第一実験 |
| [V1](../pr_rpe_v1_e0_state_radius_20261103/REPORT.md) | e520e6e396f7e0def9e522ea1cff58eb10ddfa8f。physical/control、半径の未保証 |
| [V2](../pr_rpe_v2_branch_success_20261103/REPORT.md) | 78c49d0b5f1d71e63b5959e78813237923bbc648。保存physical attenuated平均とIID仮定下のCONDITIONAL_SUCCESS |
| [V3](../pr_rpe_v3_cost_transfer_preflight_20261103/REPORT.md)／[R1](../pr_rpe_v3_r1_shared_calibration_sensitivity_20261103/REPORT.md) | cc60beb51930d4311d2ded84440682e583ccafc7 / 8fde197c64da671f1d428f0d3e8fdb762dc21930。長q価格外挿・費用UNDETERMINEDを保持 |

旧科学コードの参照carrierは4591b8d75a2a6a55c74a51225f0c61648023c8ee。
これは所在commitであり、元M06-F実行のclean SOURCEではない。
旧initial/fresh32のdirty provenanceと停止条件は変更していない。

| 関数・SOURCE | 再利用と差分 |
|---|---|
| rte.py sample_rte_events / iter_sample_rte_events / _make_event | **変更なし**。偶数order、component分布、phase、角度、B_Kを保持。samplerを作り直さない |
| failure SOURCE _seed_for_shot | **変更なし**のFunctionDefを限定読込。新しい公開namespaceとtrajectory/measurement別masterを外側で固定 |
| _trajectory_signal / _apply_event_to_state | **変更なし**。D_f→event→D_rをq回作用。Bと同じsigned componentとscalar規約 |
| 候補Bのstate/sector処理 | 保存stateを新solveなしで読む。basis/endian規約、constant/extracted identityの扱いを再利用 |
| old _explicit_fresh_iid_validation | whole runnerは再利用しない。ground-state solveとq共通r・無条件clip・half回路buildを含み、今回の保存state再利用には不適合 |
| Hadamard wrapper / native controlled assets | Bおよび既存small testsを静的参照。Aは理想state-actionの確率過程に限定し、shotごとに回路をbuildしない |
| reconstruct_rpe_phase | 仕様と既存testを参照。このAのm0/m3は非連続roundであり、decoderを呼ばずenergy/branch成功率を評価しない |

## 新コードと測定の意味

[fresh_iid_adapter.py](fresh_iid_adapter.py)は固定設定の転記、stream key、毎shotのsampler呼出し、
条件付き測定、事前固定診断を接続する。全trajectoryに共通するmutable RNGは置かない。
同じ(axis,q,shot)を二重に配ること、異なるstreamのseed衝突、欠測・重複shotはSTOPする。
**独立抽選でも同じevent列は自然に発生する**ため、event列の値が等しいことを再利用と判定しない。
factory/sampler呼出し契約の検査は数学的IIDの証明ではない。

trajectoryとmeasurementのstreamは役割を分け、cosine/sineにも別のkeyを使う。
価格標本のpaired cos/sinとは別の実験であり、その標本やseedを流用しない。
新namespaceは公開用に新規設定したもので、旧非公開seedやprivate rawを取り出していない。
旧explicit検証のmeasurement RNGはaxis単位だったが、Aではshot単位の別roleへ分ける。
新しい再現性配置として固定したもので、母分布を変更したり旧RNGをreplayしたりしない。

trajectoryのunitary signal z_ωについて、cosineはRe z_ω、sineはIm z_ω、
P(+1|ω)=(1+μ_axis(ω))/2。U=exp(-itH)、S†→H、bit0→+1/bit1→−1の規約を保つ。
helperは既にevent.phaseとdeterministic scalarを含むため、追加phase/B補正を加えない。
保存targetはphysical_df_ground_state / attenuated_event_mean_signalだけ。
corrected/control labelは拒否する。

非有限signal、unit disk逸脱、meanの[-1,1]からの許容外逸脱を拒否する。
meanが境界から1e-12以内だけ逸脱した場合に限り数値roundoffとして境界へ戻し、
raw/used mean、p、修正有無を全shotに残す。無条件clampでPASSへ変える設計ではない。

## 数値contextと並列化

[run_fresh_iid_a.py](run_fresh_iid_a.py)は通常起動／`--describe`ではmetadataだけを返す。
`--run`は処理選択であり、ユーザーの科学実行承認を証明・代行するものではない。
今回`--run`と実H4のprepare_contextは呼んでいない。

将来の実行は保存Hをprepareし、218 retained componentのsector operatorを一度だけ作る。
旧basis_change_unitaryのgate列をnumeric Operatorとして合成し、number sectorへ制限してcacheする。
deterministic halfには既存global/RZ/RZZ primitiveを使う。
旧helperが毎componentで256次元matrixを再構成する重複と、halfのtime-evolution回路buildを避ける
ための最小拡張であり、分布・PF公式・精度・積順序を変えていない。
ただし旧DF prepare内部にはGaussian gate定義の構築がある。これは将来の科学的入力準備に含め、
controlled evolution/measurement回路のbuildと区別して計画へ明記した。

独立shotを固定indexのchunkへ分ける。seedはchunk/worker番号や完了順に依存しない。
number-sectorのimmutable行列をworkerごとに一度渡し、job内BLAS1threadとprocess並列を組み合わせる。
worker数は利用可能CPUと実際のmemoryから選び、恣意的なCPU/RAM/wall-time上限を設けない。
実行前の空き資源・quota・非干渉確認は必要。speedupは測定していない。

## 人工検査と限界

[test_fresh_iid_a.py](test_fresh_iid_a.py)は**18件PASS**。
人工2成分・2次元stateのsampler、exact event列挙とnormalization、seed再現性・役割分離、
fresh呼出し・自然な同一列、符号・確率、roundoff/不正値、欠測、固定配分、
三段階の平均、UNDETERMINED/不一致判定、serial/process一致を検査した。
人工小回路ではnumeric basis/halfを既存primitiveによる明示回路とscalar込みで照合した。
mock中断はH4 contextを置換し、science/RNGを起動せずSTOP記録を確認する。
旧RPE/科学test全体は実行していない。

既存Python3.11.0rc1 / NumPy1.26.4 / Qiskit1.3.0をread-onlyで使った。
新しい科学的結果ではなく、主エージェント自身の技術検査である。
記録は[VALIDATION.json](VALIDATION.json)。Bの保存stateはbyte hashだけ確認し、今回NPZ数値decodeもしていない。

未保証なのは、実H4の新numeric context、実際のsamplingと保存μとの整合、PRNGをIIDとみなす仮定、
全sampler支持での実回路意味論、厳密半径、alias、全18round成功率、実準備/backend/noiseである。
Bの局所6requestを全trajectoryの回路correctnessの証明へ昇格しない。
V1/V2/V3/R1とBの正式status、Fullの半径問題・欠測fingerprint、M06-F/R2/R3のSTOPを維持する。

今回の完成点はSOURCE・入力・seed・判定を公開commitへ固定するところまで。
次の担当と別承認は[NEXT_STAGE_DECISION](NEXT_STAGE_DECISION.md)を参照。
公開・remote照合後mandatory STOP。新科学samplingを自動開始しない。
