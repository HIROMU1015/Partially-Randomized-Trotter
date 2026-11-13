# κ型Full-Hの有限科学検証計画（未承認・未実行）

**ここに書いた段階はどれも自動では開始しない。** signal/normalizationの数値評価（S）と、実回路接続の短いcompile（P）は独立した選択肢で、同時に実行する必要はない。m12の直接compileや全13roundの価格取得は含めない。κ、K、grouping、decoder、誤差予算は採択しない。

## 1. κ候補と選定理由

| 候補 | 役割 | 根拠 |
|---|---|---|
| κ=2 | 主候補（S-1） | E22の停留条件Bκ²−2Bκ−2A=0で、純RTEは一定費用A≈0なのでκ*≈2。既存のT2.0最良（H4 κ_eff 2.03、H6 2.11）と未実行のR2-ADAPT（2.03〜2.06）も近い |
| κ=1, κ=4 | 任意の括り（S-2） | κe^{2/κ}は1:7.39、2:5.44、4:6.59（解析モデル）。最小点付近は平坦で、Bのdecoder riskやCの価格係数でずれる可能性がある。signalが安いので感度を見る材料になる |

K=2（R448と同じ）、grouping DYADIC（q=2^m）、h=binary64 0.1、M12は固定する。これらを同時に変える計画はない。

## 2. 対象round・T・q・R・K

全13roundのq/R/s/τ/κ_eff/包絡、configuration ID、canonical candidate ID、physical IDは [KAPPA_CANDIDATES.json](KAPPA_CANDIDATES.json) に固定した（metadataのみ、未評価）。κ2の要点：

|m|T|q|R|s|physical ID|
|---:|---:|---:|---:|---:|---|
|0|0.1|1|9|9|FULL_H_2a9a68a11d08af234a8c|
|3|0.8|8|71|568|FULL_H_4de6130f0d3a66e5d695|
|6|6.4|64|566|36,224|FULL_H_4ec8d73571760a6830b8|
|12|409.6|4096|36,223|148,369,408|FULL_H_4bcadbdccdb7e96aefb5|

## 3. identityとlaw binding

κ2 configuration `54ac3eb38de5f90a77e9921100576c89f6c870f68384f4c7d9efe61b2c80c07e`（κ1 `f4fec35121ee4e7ceefdfc75957f8e2b9d03b3e30e112edcc999f85b0a5c43c2`、κ4 `e442b9a566c15ec6a20caae3e4774f6f5512f0263fd901868e7e6d85e66d13ed`）。kind `PAPER_TOTAL_ROTATIONS`、definition `PAPER_V2_E21_…`、event_mapping `D_FULL_H_EXPLICIT_PHYSICAL_LAYOUT_V1`、law `ideal_finite_RTE_law`/`PAIRED_FINITE_TAYLOR_COMPONENT_ORDER_LAW`、RMSE policyはR448/PRと同じA S0 policyである。生成は既存のC `configure/prepare`・B `generate/_validate`・D `physical_schedule` で行い、同じbuilderで保存R448 IDと13 physical IDが再現することを確認した。

## S-1：κ2のsignal/normalization（数値評価のみ）

- **SOURCE**：数値2 moduleはC `5cc7c07e`（`3a8b37f1`、`c3ae31f3` でもbyte一致）を変更しない。callerは `run_truncated_fullh_signals.py` の様式で、scopeとmanifestをκ2 configurationへ差し替え、下記の追加gateを入れた小差分とする。**現時点では未実装**。人工loader確認の後、新commitで固定し、その完全SHAを実行許可の対象にする。
- **入力**：INDEX `7a8bc7f1` のsnapshot/receipt/context（hash照合）、Full-H preparation（9b83 summary転記）、保存reference pair 13組（`bba15083` PHYSICAL_INPUTS、R448と同じdescriptor）。新reference・scalar抽出・SCF/DF/state生成は0。
- **件数**：context 1回、非scalar spectrum 1回、finite normalizer 13回（異なるτ）、polynomial/power 13回、mean 13件（raw/correctedの組、X/Y座標26）。sampling・回路・N・価格は0。
- **gate**：既存gate（complex128、state norm 1e−12、Hermiticity/再構成 1e−9、reference pair 1e−10、raw/corrected closure 1e−9、Raw∈[−1,1]、b>1かつ有限、clampなし）に加え、次を検査する。
  - configuration ID、canonical ID、physical ID、q/R/sが計画と一致すること。
  - 0<logB≤(λT)²/s＋`normalizer_one_ulp_log_sensitivity`。
  - correctedとreferenceの位相・半径（既存のlocal診断）を記録する。
- **負荷**：R448実行ではB0I＋Full-Hの26 meanが4.15 s。κ2もevent列を作らないので数秒程度と見込む（未計測）。CPU/RAM/workerに人工上限を置かない。
- **出力**：fresh directory（例 `docs/research/d_full_h_kappa_signal_results_20261114/run01/FULL_H_K2_QDYADIC_KAPPA2/`）。`round_m.json`、`RESULT.json`、`B_SIGNAL_INPUT.json`、SOURCE/INPUT_PROVENANCE/COVERAGE/STOP。
- **S-2（任意）**：κ1とκ4の各13 mean（normalizer 26、mean 26）。κ2がCOMPLETEになった後、同じbatch内で続けて実行する。承認はκ2と別に選べる。

## P-1：新loweringでの配線確認（短いcompile）

- **cell**：κ2 m3、T=0.8、q8/R71/K2、s=568、`FULL_H_4de6130f0d3a66e5d695`（canonical `RMSE:201ebc6f…`）。m0（9 event）より成分の種類を十分含み、既存Full-H direct（H4最大s=4,096）より小さい。
- **path**：fresh normal 1 path（index 0）、cohort `D_FULL_H_KAPPA2_NEW_ZZ_M3_WIRING_V1`、outer 0〜7のseed 8個は元Full-H `trial_seed`（`DF_FULL_H_LIGHT_20261109` family）から計算済み（JSON `proposed_wiring_cell`）。D S1のm0 seedを同じ抽出で再現することも確認した。
- **SOURCE**：Full-H `9b83b96b`（`setup/context/prepare_full_h/full_oracle/trial_seed`）、native `af284d85`（`compose/cost_trial`）、provider `c31ef424`、新ZZ `01e068af:efficient_controlled_zz.py`。callerは**未実装**。fresh PUBLIC/PRIVATEを設定し、composeの範囲だけ新ZZを有効化し、`cost_trial((row,cohort,0))` を1回呼ぶ小関数とする。旧 `main/run_cost`（n9、費用会計）は起動しない。
- **件数**：context 1（内部でprepare(ham,0)＋prepare_full_h、保存T=.8 reference読取、oracle）、sample 1 path（8 request）、compose 1、wrapper 2、**compile 2（X/Y）**、statevector最大7。
- **検査**：index0の既存gateを全て使う（low control、high-control leakage、独立occupation oracle vector、X/Y Hadamard期待値1e−9、compiled vector 1e−7）。加えて次を記録する。
  - 新ZZ builderが有効だったこと（`lowering_id`）、旧 `RZZ.control(1)` 命令が0であること。
  - scalar=c_H＋Full-H抽出identityが総Tで一回だけ適用されたこと。
  - one-body成分が実際にsampleされたこと。
  - 軸別RZ/CX/depth/size、compile時間、peak RSS、full fingerprint。
  - C guard用のsource-bound implementation sidecar。
- **位置付け**：新loweringでκ layoutの実回路が正しく作られることの技術証拠である。law平均、CI、model係数の資格はない。D S1のF-H案（R448 m0）とは選択肢で、両方を既定にはしない。

## P-2：長時間価格モデル用データ（任意、Cのmodel scope）

BのN/比較margin、Cの必要精度を受けてから設計する。参考として最小構成の例を示す（n、精度、採用閾値は未定）。

| 種類 | cell | 目的 |
|---|---|---|
| 単一group | T=0.1・q=1・R∈{566, 4,528, 36,223}（κ2 m6/m9/m12の1 groupと同じτ） | per-event傾きbとgroup内定常性 |
| 全round holdout | κ2 m5（q32/R283、s9,056）、m6（q64/R566、s36,224） | a＋b·s＋c·(q−1)予測の独立確認 |

例えばgroup cell n=2、m5 n=2、m6 n=1なら18 compileになる。単一group cellはどのconfigurationのroundでもないので、Cの価格modelでcalibration recordとして受ける形式を先に決める。R=36,223とm6は既存direct（H4最大4,096 event）の約9倍で、時間・RSSは未知である。P-1とm5規模の実測後に並列度を決める。m7〜m12は直接compileせず、検証済みでないmodel scenarioとして扱う。

## 実行と依存

- 順序の推奨：S-1 →（S-2）→ BによるN/risk評価（Bの別許可）。並行してP-1を実行できる。P-2はB/Cの精度目標が出てから。
- 担当：S-1/S-2はCの既存caller様式で、C又はDが実行できる（ユーザー指定）。P-1はD。P-2はD（取得）とC（fit/検証）。
- 正式Gには、κ候補のRaw → Bの適格N → Cの全13round E[C]（m10〜12はmodel）→ 同じ比較policyでのB cost、の全てが必要。

## STOP条件

- SOURCEのbyte、input hash、configuration/physical ID、q/R/s、reference descriptor、preparation、scalarのいずれかが不一致。
- 数値gate、logB包絡、Raw範囲のいずれかが失敗。非有限値。
- compile側：index0のいずれかのgate失敗、新ZZが無効、旧RZZが残存、scalar不一致、one-body非sample、OOM又は継続不能な資源不足。
- 上記の場合は部分結果・理由・環境を保存してbatchをSTOPする。retryしない。別seed・round・κ・Kへ切り替えない。閾値を緩和しない。片軸だけで済ませない。UNKNOWNを0やPASSにしない。
- 一時的なPSI・CPU競合だけでは止めない（記録する）。他ユーザー・他担当のprocessや設定には触れない。
- 完了時もmandatory STOP。S/Pの結果からκ・K・groupingを自動採択せず、N・価格・Gの評価へ自動で進まない。
