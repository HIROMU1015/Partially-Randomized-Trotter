# Track A（O1）M1：技術調査・最小検証準備報告

2026-11-06 JST、SERVER 2（GPUA100-gleap02）。基点は`aeb9d1847f9d500f2389db12b384bac6087117a9`、専用branchは`codex/track-a-o1-m1-validation-preparation-20261106`。成果物は本書と[M1_COMPARISON_SETUP.md](M1_COMPARISON_SETUP.md)、[M1_EXECUTION_READINESS.md](M1_EXECUTION_READINESS.md)の3文書のみ。

**保存入力候補と既存APIの再利用境界を特定した。H4-A/Bは未採択、必要接続は未実装、新科学実行は未承認・未実行である。** 最小の次段候補は、一つの共通物理splitについてP/S二端点のfull H・T・V・辞書和・scalarの対応を数値確認すること。今回その数値確認はしていない。

## 1. 得られた具体的な準備

| 調査対象 | 確認事実と準備状態 |
|---|---|
| P0/F1・F2保存入力 | public資料にphysical/SS/PP/bundleの科学入力hashとschemaはあるが、rawはprivate保全物で今回未取得。固定Git treeと取得資料で利用可能なSERVER 2の原本pathを確定できなかった。不存在とは断定しない。P0のoccupied(0,1)は分子元H groundの資格ではない |
| F2保存価格 | 修正版結果commit `dc7378ab34015287215bcf4ce52147e46d94ecd9`の`docs/research/pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/EXECUTION_REPORT.md`はprep X0X1込みと明記。SOURCE `9468c7089727f5648d9f59cddbaebe0a42f5f0a0`の`src/trottertracks/pr_dictionary_validation/circuits.py::hadamard_wrapper`とも対応する。`C_full,no-prep`に転用できず、単純減算もしない。旧operator/mean PASS、N=null、既存STOPを保持 |
| H4公開入力 | 保存anchor `4591b8d75a2a6a55c74a51225f0c61648023c8ee`の`artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz`を固定。2,157,448 bytes、SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。ZIP/NPY headerと保存receiptを照合でき、数値配列はdecodeしていない |
| DF rankとstate | tol-only1e-8、requested rank=None、保存実rank10。legacy12/研究C8と異なる。`state`/`DF_state`は各256成分だが、生成SOURCE `0c329b30a81accb35c212be8defa66fee0dd0f12`の`src/trotterlib/h4_molecular_inputs.py`は元full/DF fullそれぞれから生成する。元H検証候補は`state`であり、DF_stateに置換しない |
| 分子のmatched比較 | 既存案Aは共通T_eff/V_square、案Bは共通T_bare/V_normal。各案内のP/S二端点を比較する候補。Aは既存一体保持・全square tail経路に近い。Bは一体収縮D̂のI/Z接続が追加で必要。旧Pauli prefix27対DF prefix1は同じHD/HRではなく、この比較を代替しない |

保存metadata・出典の完全path、memberのshape/dtype、state/hash、P0入力hash、scalar/sectorの対応は[M1_COMPARISON_SETUP.md](M1_COMPARISON_SETUP.md)にまとめた。過去receiptの小さいfull-H差・state residualは保存結果の引用であり、新T/Vの一致、数学的厳密ground、元H energy精度の新保証にはしていない。

## 2. 再利用できる部分と残る技術差分

既存`DFHamiltonian`、DF partial-S2/tail、Pauli residual作用、finite-RTE law/mean、controlled event・actual adjoint、no-prep Hadamard wrapper、fixed-signalとRPE会計を再利用候補にできる。P0の2fragment/4arm APIをrank10分子selectorへ一般化せず、選んだ共通T/Vのexplicit P/S入力と既存経路の境界だけを将来接続する。新selector、混在内点探索、再DF、新state生成を提案・実装していない。

no-prep wrapperはあるが、現公開入口の短round・DF型条件への接続差が残る。固定TのN APIにはbias・allowance・log_Bと根拠、全RPEには半径・誤差・失敗配分と全roundの価格が要る。既存の単位半径specializationや旧18round成功を新分割へそのまま移せない。F2の統計関数の存在も新条件のN資格にはならない。具体的関数名・固定SOURCE・既存testと制約は[M1_EXECUTION_READINESS.md](M1_EXECUTION_READINESS.md)に示した。

## 3. 比較条件と未決定事項

M1の機構比較では元H、HD、HR、元H state、scalar/identity、保持programと情報アクセスを揃える。P/S双方でglobal collect、合法なbasis/control最適化、actual adjoint、full no-prep compileの機会を与える。異なるlawに同seedを付けただけで統計的paired標本とは呼ばない。

task全体の強い対照には同じ情報でのUWC・決定論PF・既存最適化の機会を含める。UWCが表現・splitを変える比較は、同じHD/HRを隔離するM1機構比較と区別する。λや単回価格だけから総PR＋RPE費用改善を主張しない。本調査で新しい総費用、順位、優越性は得ていない。

次の判断は以下のとおり。

- **ユーザー**：P0 private原本の回収を要するか、公開H4-A/Bのどちらを進めるかを選ぶ。両案は未採択で、Aの方が既存接続との差は少なく、Bはbare一体/二体分割への情報価値とD̂接続作業がある。
- **ユーザー**：共通targetと数値DF差のallowance/誤差転送、identity/scalar、保持blockを選び、operatorのみかmean/価格/全RPEまでかの科学範囲を明示許可する。近似DF energyへtargetを変更したり、無断の残差補償を加えたりしない。
- **Codex**：許可後に限り、選ばれた入力・共通splitに必要な最小接続と適切な既存testを実施する。新科学実行と重要科学レビューの許可は独立している。今回はどちらも開始しない。

## 4. 実施範囲・公開・停止

既存O1四文書、3環境統合台帳/再利用map、文献調査を継承し、必要な固定SOURCEと保存metadataだけを照合した。実施した確認はGit object/path、sourceの静的読解、公開JSON metadata、archive bytesとheader、文書の参照/変更範囲である。科学module import、保存配列の数値decode/再解析、H/residual/state生成、simulation/sampling/RNG、shot計算、回路build/compile、fit、algorithm実装、科学test実行、重要科学レビューは行っていない。

公開対象はこの3文書のみ。旧SOURCE、raw、結果、O1文書、台帳、固定契約、STOP、停止familyを変更しない。専用branchのremote commitと3文書の再取得内容を確認した結果は最終チャットで報告する。

**mandatory STOP：文書の公開・再取得照合で今回作業を終了する。A/Bの採択、科学実装・実行、旧run再開、M2との結合を自動開始しない。**
