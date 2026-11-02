# A2 認証情報・SOURCE対応表

[preflight報告](practical_preflight_report.md)の追跡表。SOURCEは固定 `db1819587df36ee8833b37227ca0083faa0408ec` を実読した。文書のK0/K0.5/RF/EW2は[evidence index](evidence_index.json)の指定commitを正本とする。既存SOURCEは変更・実行していない。

## 1. 表の読み方

- **構成法あり**：既存の紙上手順。数値certificateを取得済みという意味ではない。
- **数値APIあり／未実行**：該当処理がSOURCEに存在する。正当な入力・精度・実環境で機能することを今回検証していない。
- **証拠未確認**：本class・同一H/sector/分割/順序へ接続した証拠を今回の限定調査で確認できなかった。
- **認証手順不足**：既存数値値・helperをcertificateへ格上げする、丸め・残差・同値性等の手順が不足する。

「理論的に値を定義できる」「保存値がある」「量子入力として準備できる」「認証済みである」を別に記録する。以下の全てについて新しい数値を求めていない。

## 2. 量ごとの対応

| 必要量・条件 | 実際に確認したSOURCE/APIと既存資料 | 取得状態・certificateとしての有効性 | 追加で必要な技術作業（今回未実施） |
|---|---|---|---|
| **Hの表示・PSD分割**：元H=LI+ΣF_j、全F_j PSD、共通sector | `df_hamiltonian.py: DFHamiltonian, df_hamiltonian_from_integrals, select_blocks`。K0 §6.1のpositive/signed平方＋一体shift | 表示API・既知構成法あり。signed λ、元H対応、PSDの実入力証拠は未確認。`select_blocks`は全Hを返すとは限らない | 完全fragment一覧、順序、符号、Hermiticity、scalar・元integral対応、PSD証拠を同一identityへ接続 |
| **sector**：全fragmentで共通不変 | `PhysicalSector.number_sector/spin_sector/from_h_chain`、`_build_excitation_table`。K0 §6.4 | 基底列挙・投影APIあり。N保存の構造はあるが、spin sectorのfragment別保存証拠は未確認。基底外skipは証明にならない | N/各spin条件とg行列の保存則、basis/JW ordering、projectionと回路の対応を検証 |
| **L**：実装するPSD分割の下界 | K0 §6.1 Eq.(18)・signed shift。`df_trotter/decompose.py: diag_hermitian` は`np.linalg.eigh`、`df_diagonal`等も数値helper | 構成法と数値APIあり。PSD分割に添付した認証Lは未確認。数値eig出力自体は下界証明ではない | 一体sector下限とsigned平方capの証拠、scalar復元、丸め/モデル誤差を含む下界ledger |
| **U**：同一H・sectorでE0≤U | K0 §4.3。`_h_chain_integrals_session_cached`の`hf_energy`、`solve_df_ground_state`と`DFGroundStateResult` | 数値HF/solver機能あり・未実行。今回保存energy値を確認・使用しない。HF metadataも近似固有値も、同じDFモデルへの検証可能な変分上界とは限らない | 明示trial stateのsector/norm/expectation、各誤差と元H対応の検証付き上界。数値solver残差受理からUを認定しない |
| **W=U−L**：共通モデルの認証区間 | K0 §4.3、EW2監査のtruth-free置換 | L/Uの証拠が未取得のためWも未取得。exact FCIやground vectorを無料のcertificateにしない | L/Uの出所・単位・丸め方向・共通対象を固定後、検証可能な区間として形成 |
| **f_j**：0≤F_j≤f_jI | K0 §§4,6,7。`df_partial_randomized_pf.py: _block_operator_norm_upper_bound`、`df_trotter/ops.py: build_df_blocks/build_one_body_gaussian_block` | 数値helperは一体のΣ|eps|、平方の|λ|(Σ|eta|)²を返す。exactな入力・スペクトルなら既知のbound形だが、binary64 eig/sum・shift後F_jの証拠はない。`df_fragment_weight`の|λ|‖g‖F²は**partition ranking proxy**でf_jではない | eig/行列残差と安全な区間演算、shift後のPSD norm、sector条件を証拠化。異なる量をラベルだけで流用しない |
| **Λ=Σf_j**：同じ分割からの認証和 | K0/RF0の定義。`_pf_phase_rate_upper_bound`は別用途のfloat和 | 計算部品に似た処理はあるが、A2のΛ証拠は未確認。同helperはPF weightsとenergy−shiftを含むphase tracking量で、truth-free Λそのものではない | certified f_jを集約し、分割/順序/scalar identityを記録。保存phase-rateや経験normで置換しない |
| **K_SOS**：K0の同順序係数辞書Bと最大行和 | K0 §4の辞書・Eq.(9)。O(M³)算術、O(M)行和保存という既存紙上手順。RF1での利用は固定RF/EW2監査 | 構成法あり。調査したSOURCE内に本class用の専用certifier・保存証拠は確認できず。ground φや多体対角化は不要だが取得費ゼロではない | K0の係数・添字/重複・単位・全fragment・算術誤差を保つ実装/証拠手順。実際の生成・bit計算費を記録 |
| **Cbar2**：同じS2のC2を上から認証 | K0.5 §3.4およびEW2比較§§1–3の強化対照。`pf_decomposition.py`で順序、Gaussian block APIで係数の入口 | 理論的取得経路あり。同じ分割の認証nested-commutator norm値は未確認。float phase-rate/経験PF係数/真のbiasから逆算しない | 固定順序の辞書、可換性・sector・係数残差を保ったnorm証拠。相対側と同じ構造情報・取得予算。低energy等の使える強い対照も除外しない |
| **state access**：EW2の厳密ground支持、または別に正当化された近似state仕様 | `solve_df_ground_state`、`_save_df_ground_state_npz`、cache key。`rpe_hadamard_interrogation.py`は`state_preparation_included=False` | 古典solver/cache APIあり。`max(100*tol,1e-8)`等のsolver受理は厳密ground証明・no-gap距離保証ではない。量子準備API/費用・厳密ground copiesは未取得 | state品質と対象taskを科学的に決め、overlap/支持/誤差・準備成功率と実装費を別会計。今回はその新保証を構成しない |
| **絶対原点・normal ordering** | `df_trotter/decompose.py: _parse_low_rank_result/df_decompose_from_integrals`、`two_body.py: interaction_operator_from_df_model`、`df_hamiltonian_to_model` | scalarを持つ経路はある。spin-orbital/full値とspatial/correction-onlyは規約が違う。truncation値をscalarに足さない既存対策を確認 | 外側c、onebody correction、tensor係数、nuclear scalar、extracted identity、PSD shiftを一度ずつ台帳へ記録。実数値照合は別承認 |
| **モデル誤差・有限精度**：元Hと実装モデルの対応 | `df_hamiltonian_from_integrals`metadata、`DFModel.hermitize`、`_as_hermitian`、`diag_hermitian` | truncation指標・対称化・数値eighはあるが、元H→DF→回路→cert全体のvalidated enclosureはない。ファイルhashは数理certificateではない | raw入力、切捨て、対称化、固有値/回転残差、trig/合成/制御誤差を明確化。厳密保証と条件付きengineering評価の選択は未採択 |

## 3. 回路・保存・補助証拠の位置付け

| 実ファイル | 確認できた範囲 | このpreflightで証明したことにはならないもの |
|---|---|---|
| `df_trotter/ops.py` | DFBlockのsigned λ、Gaussian、RZ/RZZ、controlled scalar PhaseGate。Nature経路とdense fallback、Givens指定経路 | A2のPSD adapter、installed stackのcorrectness、scalable fallback、native合成誤差/cost |
| `df_trotter/circuit.py`・`pf_decomposition.py`・`product_formula.py` | energy_shift・固定PF schedule・正二次S2の部品 | 全PF labelへのRF保証、絶対phaseを捨てた比較の正当性 |
| `df_partial_s2.py`・`df_partial_s2_repeated.py` | deterministic-only経路、controlled対角/phase、反復builder | randomized RTE平均へのEW2適用、完成したground-QPE/readout/access保証 |
| `df_rpe_resource.py` | 既存compiled providerのmetadataでもstate preparationを含まない | A2用総費用を測ったこと、過去PR/M1-S費用のそのままの移植 |
| `rte_connected_cluster_cost_validation.py` | DF snapshot writer/loader、schema/hash検査のSOURCE | loader未実行。添付snapshotのPSD、元H対応、認証L/U等 |
| 履歴DF `.hamiltonian.npz` | ファイルhashとZIPメンバー名のみ | 配列内容・λ符号・PSD性・sector適合。次期実験入力の採択でもない |
| `tests/test_df_hamiltonian.py` | 第4戻り値をconstantへ加えないtest定義等がある | 今回のtest PASS、古い保存artifactへ修正が反映されたこと |
| `n3_occupation_certificate_audit.py` | 別系列のrational LDL/PSD witness checkerとsaved検査がある | 分子DF/SOS・全丸め/交換子/costのcertifier。N3を再開していない |

SOURCEへの固定リンクと各functionの読取位置は[evidence index](evidence_index.json)に集約する。未確認の関数を作って「実装済み」と表記しない。ここには新規SOURCE・契約・数値certificateを含めない。**Mandatory STOP。**
