# 研究C・C4：独立研究の一時終了とC1への成果引継ぎ

2026-10-24 JST。状態：**C4_TEMPORARILY_CLOSED_USER_ADOPTED**。

ユーザーは、研究CのC4「Rustiq接続・単純角度結合による回路資源改善」を独立研究方向として一時終了し、重点をC1「圧縮DFとnative回路費用の共同改善」へ移す方針を今回正式承認した。前段の「終了は未採択」は当時の履歴として残し、本書を新しい研究判断の記録とする。

一時終了の対象は今回の限定探索。Rustiq一般のNO-GO、非可換Pauli network合成の不可能性、Gaussian合成全般やPR研究全体の終了ではない。C1の具体的方式は未採択であり、今回許可するのは[静的な接口・実装資産調査](research_c_compressed_df_interface_survey_20261024.md)まで。C4/R2追加実行、Gaussian singleton再開、C1のfit・実装・科学実行を許可しない。

## 1. 研究目的と固定証拠

C4では、既存のordered PR中央領域へ位相・順序を保持してRustiqを接続できるか、得られたcompiled費用差が既知の単純角度結合を超えるかを確認した。以下は既存固定報告の引用で、今回再計算・保存値の数値再解析をしていない。

| 固定資料 | 保持する証拠・役割 |
|---|---|
| [Rustiq限定実行][R1]、`f95148e3f176efc6b3fd81a93d043ccd20255343` | B0/B1/R1の登録local/full/actual adjoint/raw・compiled X/Y/compiled operatorがPASS。science SOURCEは`b546c4e4f5d6645fd198b9ec3f85e3cd644ac4e9` |
| [単純角度結合比較設計][DESIGN]、`b7b70f6a5e3ee4ae4a55b215b49de0390e5732e5` | 元run/basis/phase固定、隣接同一controlled Pauli角だけのB2と公平な比較条件。R2を必須にしない |
| [B2実装準備][PREP]、`0c9d0a480db95e4bbd57a1cda02f7fb24c2b2162` | B2 opt-in、math.fsum、fusion ledger、保存reference reader、非科学的unit/mock 41件の過去PASS。B2科学SOURCEもこのcommit |
| [B2限定科学比較][B2]、`c3511345ab24bb19a77fb9f17a8675e4e805a8a3` | B2全登録gate PASS。全8 wrapper・全8 compiled指標でB2=保存R1。1 attempt、retry/resume0、追加reference/Rustiq/旧route再実行0 |
| [Gaussian singleton一時終了][GAUSS]、`69f8e1cac23a4924ab81aebe399657a705024f6a` | 別方向の終了判断と資産の保全。今回その研究を再開せず、共同合成・統計・同精度比較の境界だけを引き継ぐ |

**上記数値の共通scope：** 保存F1の4 mode・2粒子・occupied(0,1)、density-square合成Hamiltonian、`L_D=1`・残差2fragment、`T=1/5,q=2,δ=1/10,r=2,h=1/20,K=2`の一点。分子geometry／AO basis／分子DF rankは非該当。SS M2/M3/M4・PP M0の4case、X/Y両axisを含む8 full measured Hadamard wrapper。Qiskit1.3.0、native `rz,sx,x,cx`、opt1、seed17、approximation1、qubits_initially_zero=false、free topology。行列絶対tol1e-10・complex XY tol2e-10・rtol0・phase fittingなし。これは固定公開されたローカル証拠であり、CI再生成・外部独立再現ではない。

## 2. 正結果・負結果・未検証範囲

| 区分 | 正確な結論 |
|---|---|
| 接続の正結果 | R1とB2はいずれも対象scopeのcontrolled相対位相、actual adjoint、full X/Yとcompiled正しさgateを通過した |
| 既知の実装改善 | SS3caseでB0/B1に対しRZ・CX・depth等が減少した。B2でも同じ削減を完全に再現できた |
| 限定的な負結果 | PP M0に改善なし。全8 wrapper・全8資源指標でB2=R1であり、この比較でB2を超えるR1の追加compiled削減は観測されなかった |
| 独立研究の評価 | 既知の角度結合を超える新アルゴリズム上の優位性は今回確認できず、一時終了する。既知法の有用な実装改善が得られたことは残す |
| 未検証・未同定 | Rustiq内部機構の一意同定、異なる入力・非可換network・compilerへの一般性、matched古典timing、law期待費用・必要shot・全round総資源・分子での改善 |

[前段の報告][R1]にある8 wrapper無重み合計RZ `800→780`、CX `402→382`、total depth `988→948`をそのまま引用する。今回新しい合計・割合・順位は計算していない。設計した有限集合の1-shot費用であり、law重み付き期待資源、認証shot数、全round総資源や一般分子の改善率ではない。

限定結論は、**固定小規模合成fixtureで観測されたRustiq適用後のcompiled回路費用削減は、既知の単純な同一Pauli controlled角度結合でも再現可能だった**、まで。全R1内部処理が角度結合だけだったことや、別構造でも追加利益が存在しないことは証明していない。

## 3. 再利用資産と使用条件

| 固定資産（c3511345 tree） | C1その他への引継ぎ | 維持する制約 |
|---|---|---|
| `src/trottertracks/pr_dictionary_validation/rustiq_adapter.py` | B2 `fusion_groups`、既存planner・emitter、route identity、fingerprint、ledger。今後の既知の強いbaseline | B2はopt-inのまま。typed generator・元word/support/frame/run/macro一致、隣接のみ、元singleton/phase保持。C1の新表現へ自動適用済みとはしない |
| `rustiq_execution.py`と既存CLI | stage barrier、fresh承認照合、local/full/adjoint/wrapper/cache、saved target接続、call journal | 固定4case用の実行scope・旧grant・retry/STOPを新研究へ流用しない。C1へは再利用可能な検証部だけを別条件で接続する |
| `reference.py`／`gates.py` | 独立CAR/exterior参照、絶対operator比較、control0/1、unitarity、phase保持、実X/Y、official-definition lowering | 旧referenceやPASSを新近似Hamiltonianのreferenceとして使わない。scalarとsector/full-spaceの違いを新契約で明示 |
| `circuits.py`のactual adjoint／Hadamard wrapper | 完成回路からactual inverse、state preparation・readout・measureを含む公平な評価 | 現fixture用D側と汎用分子DF側の接口は別。C1のbasis・register・新Hを照合してから使う |
| `tests/management/test_research_c_rustiq_adapter.py`／`test_research_c_rustiq_execution.py` | 人工境界・符号・angle・cache・ledger・承認/失敗保全mock | 今回test再実行なし。mock PASSは新表現の科学的同値性PASSではない |
| `artifacts/research_c_rustiq_limited/2026-10-24/attempt01/` | 保存B0/B1/R1、source/input/profile hash、独立reference、QPY/NPY、原STOP | 原本を上書きせず、歴史的対照として引用。新分解の価格・誤差へ移送しない |
| `artifacts/research_c_rustiq_angle_fusion/2026-10-24/attempt01/` | B2と保存対照の対応表、元plan/fusion ledger、gate、raw IR、timer、manifest、原STOP | 資源同値性の固定証拠。C1の新しいsampling law・finite target・認証Nの証拠ではない |

新しいC1 baselineには、B2が**合法に適用できる場合に同じ機会を両方式へ与える**。そのための型/境界・phase確認は将来の実装条件であり、今回はコード変更・移植をしていない。B2を新しい提案手法の独立寄与と呼ばず、known optimizationとして扱う。

## 4. C1への引継ぎと再開条件

C1では元integralsから圧縮表現への近似、scalar/one-body、sector、general diagonalとsquare型、Gaussian basis・中央回転・D/R split・tail normを先に分離する。fragment数やqubitization λだけをnative PRの優位と扱わない。C4からは強いbaseline、位相保持wrapper、実compiled費用、単純最適化の対照を置く習慣を引き継ぐ。C1方式・rank/精度契約は[別調査](research_c_compressed_df_interface_survey_20261024.md)の判断事項であり未採択。

C4の再開を**検討する**条件は、単純B2で説明されない具体的構造・新しい外部証拠、順序と位相を保持した追加改善の仮説、取得可能な入力/実装、期待情報価値を示せること。B2=R1だった旧4caseを再び走らせるだけ、文献benchmark倍率だけ、scopeや許容差の変更だけを理由に再開しない。新しいbaseline・対象・SOURCE/input・科学call・STOPを固定し、実装と科学実行を別途承認する。これは将来の再検討条件で、再開承認ではない。

## 5. 正式状態・次の担当・保全

- **今回採択済み：ユーザー**によるC4の独立研究としての一時終了、C1への重点移行。C1の具体的方式選定や実装・科学検証の正式採択は含まない。
- **今回実施：Codex**による固定資料の引用・静的調査、終了/引継ぎとC1調査文書、研究状態metadata・索引・当日noteの追記、専用branchの公開照合。
- **次の判断：ユーザー・研究C・統括**がC1の候補と対象/誤差/sector/評価scopeを選び、必要な実装準備を別依頼する。研究Aのlaw/shot、研究BのPF誤差方針を研究Cだけで採択しない。
- **次のCodex作業（未承認の提案）：** 選定済み外部実装と保存integralの版・来歴を固定し、既存DF下流への最小adapterと人工mock、未承認科学契約を準備する。新fit/build/compile等はさらに別明示承認が必要。

旧SOURCE・raw・結果・grant・ledger・marker・STOPは保全し、過去の科学status/未採択・STOPを書き換えない。原workspaceを変更しない。今回の新科学call・数値再集計・Hamiltonian生成・fit・回路build/compile・新adapter実装は0。公開とremote照合後に**mandatory STOP**し、C4/R2/C1実行・Gaussian singleton再開へ自動進行しない。

[R1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f95148e3f176efc6b3fd81a93d043ccd20255343/docs/research/research_c_rustiq_execution_20261024.md
[DESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b7b70f6a5e3ee4ae4a55b215b49de0390e5732e5/docs/research/research_c_rustiq_angle_fusion_baseline_design_20261024.md
[PREP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c9d0a480db95e4bbd57a1cda02f7fb24c2b2162/docs/research/research_c_angle_fusion_b2_preparation_20261024.md
[B2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c3511345ab24bb19a77fb9f17a8675e4e805a8a3/docs/research/research_c_angle_fusion_b2_execution_20261024.md
[GAUSS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/69f8e1cac23a4924ab81aebe399657a705024f6a/docs/research/pr_gaussian_singleton_closeout_handoff_20261024.md
