# 既存研究：成果・停止理由・再提案しない範囲

[入口](README.md)／[候補](candidate_comparison.md)／[固定証拠](evidence_index.json)。以下は既存記録の引用・要約であり、科学判断の再実施でも状態変更でもない。旧記録の「採択待ち」と後日の正式採択を混同せず、後者のdecisionを優先する。

## 正式状態と到達点

| 系列 | 保存すべき成果・証拠の種類 | 追加投資を妨げた点・既知法との関係 | 現在の正式状態／今回の扱い |
|---|---|---|---|
| 一体項再配分・PF分割 | ground交換子恒等式、1-RDM応答の閉式、固定H4比較・独立数値監査 | 同情報・同cacheのcompact/grouped BCH対照が同じ応答を生成可能。固定構成の登録費用改善も未達。生成器が未実装だったことは独立差ではない | 現固定構成 `CLOSED`。一般の分割改善の不可能性とはしない。再配分を別名で提案しない |
| Compact BCH、H-E1、M-a/M-b | 既存恒等式への還元、固定回路のcounterterm/correctorの限定監査 | scalar補正、外挿、processed PF、既知BCHへの還元。新しい有利な生成規則が特定されなかった | H-E1現定義終了、M-a/M-b `STOP`、PF小改変 `PAUSED`、旧D `HOLD`、K1未実施 |
| SOS-PFの分割生成 | PSD/SOS、DF/Gram/BLISS等との対応、K0/K0.5監査 | 独立して有利なPF-native分割則・有限step実用利益が未成立 | 当該分割探索 `CLOSED`。K0の技術的未確定性と正式終了を区別する |
| 相対形式A2 | RF0/RF1/EW2の条件付き有限step保証、SPEC-01〜07のgenerator/checker、固定DF実入力certificateと強対照比較 | 現在のPSD cap、相対係数・有限剰余と適用域では十分step優位を得なかった。d/精度/trial走査での救済根拠なし | 現行実用化・十分step改善経路への追加投資を見送り。A2全体 `NOT_ADOPTED`、A1 `CLOSED`、K1-F未承認。鋭いcap探索・cheap-certificateへの目的変更を今回行わない |
| Exact Encoding × Controlled Evolution | scalar、sector、controlled相対位相の必要条件を整理。限定DF-square構成を文献照合 | 試した固定N shiftは既知DF shiftとBLISSの特殊化。普通のzero-rotation削除を新compilerと呼べない | 限定構成 `REDUCES_TO_KNOWN_METHOD` として区切り。encoding一般の不能性や④全体終了ではない |
| N1 表現・Gaussian completion/cluster | 近似biasを元精度へ戻すselector、exact fallbackとbasis費用を評価した既存結果 | 表現の近似利益だけでは全task費用の優位にならなかった | N1 `STOP`。representation変更一般を未調査とはしない |
| N2 signed charge・算術 | signed fit、compute–phase–uncompute、bit幅、Hamming-weight/catalyst setup・償却まで既に検討 | fit改善とcontrolled算術回路の改善は別。既知adder/phase-kickbackへの置換だけでは新しい差分にならない | N2 `HOLD`。weighted square算術やphase-gradientを新候補として再包装しない |
| N3 sector/occupation縮約 | 特定classのSchur/Fantope/SDP/SOS等による縮約certificateと保存pilot | 一般二体・元energy・solver/準備費用を含む独立利益が未成立 | N3現系列 `CLOSED`。sector行列の小ささを安い量子回路の証拠にしない |
| THRIFT・interaction picture・mixed可解block | 固定templateのstep削減、Gaussian/条件付きSU(2)・小Lie toy、既知THRIFT/Patel等との照合 | 関数算術・和の指数・control・準備を戻すと現templateの投資根拠不足。時間非依存depthとsystem-size効率性は別 | THRIFT現template `CLOSED`。一般THRIFTの不能性ではない。追加可解block探索を開始しない |
| H1〜H4 | 先導固有値補正、mixed ratio/filter、Gaussian frame、copiesと絶対phase accessの静的監査 | H1恒等式・H3既知frame等へ還元。H2は分母/leakage/取得費が未解決。ground copiesはHを符号化するdensity matrix oracleとは違う | 現探索 `PAUSED`、H2個別 `UNDETERMINED_DATA_REQUIRED`。direct energy測定の一般NO-GOは導かれていない |
| C1〜C4等の旧候補 | macrocycle、branch付き外挿、Schur/Feshbach downfolding、cost-aware samplingの固定監査・限定結果 | catalog内の否定結果、既知Richardson/importance samplingへの対応、高body・normalization・state/branch取得費 | 旧状態を維持。未解決の費用があることだけを新規性・再開理由にしない |

正式状態は基点の [PF採択decision](../../../artifacts/qc_pf_exploration_state_adoption/2026-10-15/decision.json)、[SOS-PF採択decision](../../../artifacts/qc_sos_pf_a3_state_adoption/2026-10-16/decision.json)、[Exact Encoding採択decision](../qc_exact_encoding_direction_adoption_20261017/decision.json)、[A2最新方針](../qc_a2_current_route_handoff_20261018/decision_record.md)を参照する。N1/N2/N3/THRIFTの後続正式状態はこれらと[既存統合引継ぎ](../qc_research_integrated_gpt_decision_handoff_v1_20261014.md)による。一般論への過剰なNO-GO拡張をしない。

## A2の負の結果を使う範囲

既存結果が対象とするのはH4／1.00 Å／STO-3G／保存DF rank12の全12blocks、total N=4・保存8modes、one_body→block0〜11の正stage S2、trial modes0〜3、d=0.001 Ha・α_spec=0.1である。PR-tail split/δ窓の試験ではない。旧報告のRF1/EW2十分step約 `1.71e-4 Ha^-1`、構造付き交換子対照約 `1.69e-2 Ha^-1`、対照約99倍という既存概数を引用し、再計算していない。

独立実装checker受理・`ENCLOSED`は固定binary64係数Hamiltonianの認証結果であり、元電子積分への厳密保証、真のPF誤差限界、QPE/native総費用の比較ではない。実行約343秒の大部分は強い対照の生成・検証だったという[既存費用記録](../qc_a2_current_route_handoff_20261018/achievements_and_limits.md)も保持する。全時間を相対側だけに課金しない。今回この値の再集計・寄与分析は行わない。

## 他研究と残る問題

①PR資源評価、②PR内部改善、③M1-S、別repositoryの高次PF研究は対象外の状態を維持する。M1-Sのvisibilityや整数shot余地を、この新測定法の実験結果・ground状態保証・総費用改善に転用しない。既存のPF/DF/sector・strict controlled phase基盤は再利用可能な部品であり、完成した絶対ground-energy pipelineではない。

過去の範囲調査で残った「copiesを使った元Hの直接energy測定の取得費・全費用」は、H2 filter ratioやH4位相参照とは異なる狭い問いである。しかしfermionic shadows、joint POVM、粒子数保存RDM推定自体は既知である。今回残すのはその中の低rank共分散処理の方法差だけであり、旧PAUSED系列の解除ではない。

## 取り下げた新候補のスクリーニング

| 調べた方向 | 文献・既存結果 | 順位付き候補に残さない理由 |
|---|---|---|
| 一般shadows、joint fermion測定、粒子数保存だけの改善 | 2208.08964v2、2402.19230v3、2402.19349v2、2606.27254v1、2606.30601v2 | 機構そのものが既知。個別RDMの保証やquadratic boundを全energyの保証へ置き換えない |
| 既知測定ensembleの係数最適化・fragment再配分 | 2006.15788v1、2105.12207v1、FFF 2208.14490v3 | 係数依存sampling、adaptive選択、共分散最適化と一体再配分は既知。単に組み合わせる案を第2候補にしない |
| MPS/DMRG guiding state、log-depth/fusion、低T合成 | 2307.01696v2、2404.16360v3、2603.11156v2 | 準備が重要という動機は正しいが、本調査では既知MPS準備・最適化と異なる生成則を特定できなかった。一般分子の低bond・overlapは無料でなく、normal-MPS/fusion条件もある。核系の実測値を化学へ外挿しない |
| DF diagonal arithmetic、charge/HWP/phase-kickback、Exact Encoding | N2・限定encodingの正式記録 | 既存終了・保留理由を覆す具体的根拠を得ていない |
| qubitization/QROM/BLISS中心の新simulation、block-encoded Zeno | 既存D1・encoding文献照合 | 今回D1解除の明示採択なし。比較背景に留め、新規対象として順位付けしない |

これは世界全体の不可能性・論文価値の最終判断ではない。今回取得・読取した資料の範囲で、独立した具体的改善を主張できるだけの差分がなかったという技術的選別である。
