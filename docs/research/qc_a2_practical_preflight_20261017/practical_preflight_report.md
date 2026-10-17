# 有限step相対形式保証A2：実用性・証拠取得可能性preflight

2026-10-17依頼。文書の作成・公開日時は[evidence index](evidence_index.json)と[publication receipt](publication_receipt.json)のUTC記録を正本とする。

**技術分類：`INPUT_CLASS_FOUND_CERTIFICATES_INCOMPLETE`。** 固定粒子数のHermitian一体項＋符号付きDF平方という既存入力classと、その論理的な指数化部品はある。しかし、同じ元Hamiltonian・sector・分割・PF順序に対する認証情報を接続した入力は確認できなかった。次の科学検証を直ちに実行できる状態ではない。これはA2の採択、科学的GO/NO-GO、数学的一般不可能性の判定ではない。

既存EW2監査の `VALID_SYMBOLIC_ADVANTAGE_CONDITION` は維持する。今回新しい証明、数値代入、certificate生成は行っていない。[認証情報・SOURCE対応表](certificate_source_matrix.md)と[判断用ハンドオフ](decision_handoff.md)を併読すると、取得済みの部品と不足する証拠を区別できる。

## 1. 固定した読取範囲

最初に[EW2監査 `db181958…`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/db1819587df36ee8833b37227ca0083faa0408ec/docs/research/qc_relative_form_ew2_step_audit_20261017)の数学・比較・handoff・証拠索引・receiptを確認した。RF0/RF1は指定 `051df466…`、K0は `58d86727…`、K0.5は `01bdc56a…` の正本を用いる。今回その証明や先行研究レビューをやり直さない。高次PF側との直接重複は前回監査の記録を参照し、別repositoryを再監査していない。

SOURCEの正本は今回の基点 `db1819587df36ee8833b37227ca0083faa0408ec`。実ファイルを静的に読み、入力生成・保存、sector、DF変換、Gaussian/controlled S2、数値norm helper、state cache、既存SOS関連資料の範囲へ絞った。SOURCEをimport・実行していない。関数の存在は、数値correctness、実環境のAPI互換性、認証付き実装の存在を意味しない。

本資料に記す外部手法の性質は、上記固定監査の読取結果を引き継ぐ。出版版・最新依存パッケージ・既知保証全体を新しく独立確認したとはしない。全読取先のcommit・blob・bytes/SHA256と読取範囲は証拠索引に記録する。

## 2. 一つの具体的な入力class

対象候補は、既存 `DFHamiltonian` の

\[
H=cI+Q+\sum_\ell\lambda_\ell A_\ell^2,
\qquad Q=a^\dagger q a,\quad A_\ell=a^\dagger g_\ell a,
\quad q=q^\dagger,\ g_\ell=g_\ell^\dagger
\]

という表示である。ここでは一体係数行列qと、後述の一体演算子のsector下限を区別する。実・spin-freeな分子DF構成を優先し、各fragmentの保存則を確認した固定N、または固定Nα/Nβ sectorを用いる見込みがある。新しい分子・geometry・rankを選んだものではない。

`df_hamiltonian_from_integrals` はspin-orbital tensorから表示を作る既存入口であり、`build_df_h_d_from_molecule` は既存のH-chain用便利関数である。後者のSCF/integral生成は今回呼んでいない。任意の分子について実装済み便利APIがあるとはしない。

### PSD化について既存K0から使えること

[K0 §6.1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/technical_audit.md)の構成をそのまま参照する。

- λℓ≥0なら、Hermitian Aℓの平方項はPSD。一体Qは、**認証されたsector下限**q_minを得て `Q−q_min I` とする。非負DFの場合の下界は既存式の `c+q_min`。固定Nなら占有数に対応する一体固有値和、spin sectorなら各spin blockの和、という既知の取得方法がある。
- λℓ<0の項はそのままPSDではない。K0の `|λℓ|(uℓ² I−Aℓ²)` という既知の外側shiftには、同じsectorで `Aℓ²≤uℓ² I` の証拠が必要。scalar下界も対応して変わる。今回はuℓ、PSD fragment、Lのいずれも生成していない。
- これらは既存fragmentの**外側scalar shift**である。K0 Eq.(19)どおりscalarを復元すれば元のPF回路・実biasは同じ。改善候補は同じ回路の保証と認証stepであって、新しい時間発展算法を得たわけではない。
- 保守的なshiftはLを低下させ、WやΛを増やし、相対保証の許容stepを小さくし得る。実入力で悪化の程度を測っていない。shift後のf_jをshift前のhelper出力と同一視しない。

このclassの構成条件は既知資料で説明できる。一方、保存した特定入力がこの条件を証明付きで満たすことまで確認したわけではない。一般SOS/BLISSのGram PSD表示から、安価なGaussian平方指数化が自動的に得られるとはしない。複素Gramの実化、残余一体項、平方内scalarのadapter不足はK0 §§6.2–7のまま。閉じた分割生成・一体再配分系列を再開しない。

### sectorは基底を抽出しただけでは足りない

`PhysicalSector.number_sector` / `spin_sector` は古典計算の基底indicesを作る。固定Nはnumber-conserving一体項・平方と整合する。固定Nα/Nβには各gℓと一体項のspin-block構造等の確認が必要であり、全Hの対称性だけでは足りない。S²保存を確認済みとはしない。

`_build_excitation_table` は基底外の遷移をskipする。従ってsector上の数値行列が作れたという事実だけから、各fragmentの共通不変性や、物理回路のsector保存は証明できない。`PJP=0` と、指数化のsector内同値性を混同しない。JW回路は基本的にmode数のqubitを使い、古典的sector圧縮は量子sector encoding・state preparationの実装ではない。

## 3. 元Hamiltonian・scalar・DFモデルの接続

今回の最も重要な不足は、次の別々の意味を一つのcertificateへ接続する手順である。

| 経路 | 静的に確認した規約 | 認証のために残ること |
|---|---|---|
| `DFHamiltonian` | `constant + one_body + signed squares`。分子入口はMolecularDataのconstantを受け渡す | 核反発等を含む元scalarの由来・単位・原点、元integralとの対応を入力provenanceで固定する |
| `df_hamiltonian_from_integrals` | spin-orbital入力。DF correctionをone_bodyへ加え、4番目の戻り値は`df_truncation_value`としてmetadataへ保存。constantへ加えない | 切捨て指標の定義・sector normでの適用条件と、丸め・Hermitizationを含む元Hへの証拠は別に必要 |
| `df_decompose_from_integrals` | spatial chemist入力、spin展開と係数1/2、correction-onlyの`DFModel`。入力constantはこの関数内ではモデルへ入れない | spin/spatial、tensor順序、係数とnormal ordering、外側scalarを一致させる。こちらのmodelは切捨てmetadataを保持しない |
| `_parse_low_rank_result` | 4-tupleの第4scalarをconstant correctionとして解釈しない | 既存対策をバグと誤認しない。外部APIのversion/signatureと他の戻り形式は将来固定が必要 |
| `df_hamiltonian_to_model` | `one_body_correction`/`constant_correction`というfieldへ**全onebody/全constant**を入れる既存bridge | correction-only modelとfield名だけで混用しない。外側constantやonebodyを二重加算しないadapter契約が必要 |
| `interaction_operator_from_df_model` | spatial blockを再展開しCAR normal ordering。外部constantとmodel.constant_correctionを加える | 上のbridgeをそのまま別規約へ渡してよいわけではない。既存callerの誤りを確認したという意味ではない |

`DFHamiltonian.hermitized` の `_as_hermitian` はskewがtolを超えても例外にせず対称化して返す。回路側も`DFModel.hermitize`、数値`eigh`等を使う。**Hermitianな出力を作る処理は、入力が厳密Hermitianである証拠や、その変更の残差予算ではない。** 今回それを修正・実行せず、認証接続の不足として記録する。

`select_blocks` や `build_df_h_d_from_molecule(...block_indices=...)` は部分Hamiltonianを返せる。A2で元Hを対象にするときは未選択項を消したH_Dだけへ保証を付けて済ませられない。全fragmentとscalarを同じ分割に含める必要がある。

元電子Hamiltonianを対象とするか、保存binary64 DFモデル自体を正確な対象とするかは未採択。前者にはDF切捨て・入力丸め・Hermitization等のモデル誤差を別予算へ戻す必要がある。metadataにtruncation値があることだけでこの予算が閉じるとはしない。後者を選んでもsector・PSD・丸めを含む証拠の取得は必要であり、分子への厳密な保証と呼び替えない。

## 4. 既存の保存入力はあるか

履歴中のDF snapshotとして、次を**ファイルidentityとZIPメンバー一覧だけ**確認した。

`artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz`

3589 bytes、SHA256 `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b`。`schema_version / hamiltonian_hash / constant / one_body / lambdas / g_matrices / metadata_json` の.npyがある。writer/loaderも存在する。**.npy payloadは読まず、λの符号、配列値、rank・sector実適合、PSD、norm、energyを確認していない。** このファイル名の条件を次期実験へ採用したものではない。

同名JSONはキー構造のみ確認した。schemaとhashは保存DF representationの追跡に役立つが、L/U/f_j/K_SOS/Cbar2の証拠や、元分子Hとの厳密同値性を与えない。古いSOURCEで生成したsnapshotへ現行のscalar修正が遡及適用されたと仮定しない。

既存PR-2には旧入力再現不一致STOPと別rebaselineの履歴がある。古いnpzの存在を「認証済み分子入力」の証拠にしない。他研究のheld-out配列を開いたり、再解析したりしていない。`saved_oracle`という名前だけでも不十分で、確認したC1 specは非分子の2-qubit synthetic Pauli対象であり本classの入力証拠にはならない。

## 5. 回路部品で可能な範囲

`build_df_blocks` / `build_one_body_gaussian_block` は一体係数行列を数値対角化し、Gaussian回転と対角phaseを作る。`df_squared_diagonal_primitives` はλの符号を保持するRZ/RZZとglobal phaseを持ち、`append_diagonal_primitives` はcontrolled時のscalarをancillaのPhaseGateへ移す。`QiskitDFPartialS2CircuitBuilder` は無制御basis回転とcontrolled対角部、constant・extracted identity phaseを組み合わせる。したがって、**既存平方への外側PSD shiftをscalar台帳と接続する論理的経路はある**。

ただし、A2用PSD shift・L/f_j証拠を入出力とするend-to-end adapterは確認できなかった。新しいadapterを今回実装していない。一般の`(c+A)^2`を直接受けるDFBlock APIもない。

PFは全項決定論の固定順序・正stage S2である必要がある。`iter_pf_steps`と`2nd`、決定論onlyのpartial-S2経路には対応する部品がある。random-tail平均やM1-S信号へ同じ固定unitary保証を移さない。高次PFの負stageにRF0/RF1を自動適用しない。

`U_to_qiskit_ops_jw`にはQiskit Nature経路がある一方、環境によってfull-Fock dense `logm/expm/UnitaryGate` fallbackもある。Givens指定経路は別に存在する。依存version、複素Gaussian対応、fallback選択と実装残差を確かめず「スケールする認証済みnative backend」とは呼ばない。

反復controlled回路のbuilderはあるが、既存RPE wrapperはstate preparation・backend executionを含まないとmetadataに明記する。**固定S2のQPE powersと窓・alias/readout保証を持つ完成A2実験は未実装・未検証。** 反復回路を生成できることだけでEW2のground-copy前提やQPE成功保証を満たさない。

## 6. 認証情報の現在地

| 量 | 今回確認できた取得経路 | 現状 |
|---|---|---|
| L | K0 §6.1の一体sector下限＋符号別scalar shift | 構成方法あり。現在の分割と一致する認証値・台帳は未確認 |
| U | 同一H・sectorの検証可能な変分上界という既存方針。HF metadata/数値solverは候補材料 | 数値energy/residualはcertificateではない。正規化・期待値・丸め・モデル対応の認証手順が不足 |
| W | 認証L/Uの差 | 両端未取得なので未取得。既知FCI真値から無料で作らない |
| f_j・Λ | 一体/平方のsector上界。既存float norm helperはある | mathematical boundの形と、外向き丸め込み証拠を区別。shift後PSD分割の認証値は未取得 |
| K_SOS | K0の係数辞書・最大行和。ground期待値や多体固有値は不要 | 紙上アルゴリズムあり。同じ分割・順序・certified f_jを受ける専用認証実装・証拠は確認できず |
| Cbar2 | 既知S2 nested-commutator対照＋同じsectorのnorm情報 | strong baselineの方法あり。同じ入力に対する認証値・tightさ・取得実費は未取得 |
| state access | numerical eigensolver/cache、HF初期ベクトル、sector matrix-free計算 | 古典vectorを量子ground copiesへ変える準備・品質証拠・費用はない |

「未確認」は、指定基点・調査したpathで本classと接続する証拠を確認できなかったという意味で、repository全体や将来の構成不能を主張しない。数値解法・test定義・他系列の小規模exact witness checkerは存在するが、この不足を埋めたとはしない。詳細は対応表に記す。

## 7. 情報取得と回路の費用構造

以下は既存資料・実装構造の読取であり、新しい測定・benchmark・改善率ではない。

| 工程 | 既存の構造・価格を支配し得る部分 | まだない情報 |
|---|---|---|
| 入力・分割 | integral/DF生成、tensor変換・normal order、Hermitization、保存I/Oとhash | frozen原入力から元Hまでの誤差証拠、実費、適切な再利用条件 |
| PSD/L/f_j | 一体行列の対角化や検証付き区間、signed shift、sector保存検査 | 外向き丸め・固有値enclosure・identity検証の実装とbit長・費用。粗い上界の実用性 |
| U・state | variational expectationの証拠。多体vectorを使う既存solverのsector次元は組合せ的に増加 | truth-free Uの取得価格、quantum準備の成功率・品質・inverse/再準備の単価 |
| K_SOS | K0 raw辞書はO(M³)演算、行和streamならO(M)保存、密行列ならO(M²) | これは算術回数。厳密/区間のbit計算、係数生成・検証・再利用費は別で未測定 |
| Cbar2 | 同順序S2の三重交換子とnorm上界。係数行列・可換性・sector情報が利用候補 | 展開の膨張、tightなbound取得の実費。相対側だけ厳しい取得基準にすることも、対照だけ粗くすることもしない |
| 一fragment指数化 | K0のgeneric論理価格はGaussian O(n²)、対角RZ O(n)、RZZ O(n²)。S2は単純には2M−1回 | 境界融合・zero省略・接続制約・controlled分解・精度依存合成費。論理slot数はT数ではない |
| QPE・readout | 固定S2のinteger powers、scalar相対位相、窓とbranch、readout失敗率の共通会計 | power列・state品質・alias・合成/丸め/モデル誤差を含めたtask仕様と実費 |
| qubit/workspace | 既存JW mode registerとcontrol部品。sector matrix圧縮はqubit削減実装ではない | QPE register、state-prep・native合成workspace、全stackでの価格 |
| 償却 | 同一証拠はH/sector/分割/順序/精度条件が一致する場合に再利用候補 | 検証費と生成費の分離、再利用回数・更新条件・実測。hash一致だけで数理証拠の有効性は作られない |

同じ資源座標で、前処理・certificate検証・回路生成と、state準備・各powerのcontrolled S2・readoutの反復を含めるという前回監査の費用枠を維持する。古典時間・メモリ・T/CX/depthを換算規則なしに一つのcostへ足さない。ground preparationを各queryへ不要に重複課金することも、無料にすることもしない。

比較は同じ元H、sector、PSD分割、PF順序、情報access・取得予算、成功率・精度配分で行う。EW2の比較先は前回のcenterした強いCbar2対照を少なくとも保持する。可換性、追加構造、適用可能な低energy保証が同じ情報で使えるなら両側へ許す。RFの収束半径を対照の全有効領域へ機械的に課さない。

## 8. preflight判定と残る判断

classと論理回路部品はあるので `NO_SUITABLE_EXISTING_INPUT_CLASS` ではない。必須文献は取得できており `UNDETERMINED_MISSING_EVIDENCE` でもない。しかし、元HとDFの対応、sector、scalar、L/U/f_j、K_SOS/Cbar2、stateの証拠が一つの入力へ閉じていないため、`PREFLIGHT_READY_FOR_SCIENTIFIC_DESIGN` とはしない。**`INPUT_CLASS_FOUND_CERTIFICATES_INCOMPLETE`** が今回の結論である。

後続で数値比較を設計する前に、対象モデル、認証に求める証拠水準、state access、対照の範囲を科学的に選ぶ必要がある。Codexが行える不足adapter・証拠生成手順の準備も、今回のSOURCE実装禁止を超えるため実施していない。次の限定検証が意味を持つ条件と、ユーザー判断・実行承認の分離は[handoff](decision_handoff.md)にまとめた。

独立作業領域・新規専用ディレクトリのみへ記録。旧SOURCE・raw・契約・原本・authorization・grant・ledger・marker、親worktreeと他研究を保全する。研究状態・全体索引を採択済みへ更新しない。新しい科学計算、保存値再解析、証明・保証、科学SOURCE、科学実行契約、authorization/grant/marker、新しい重要科学レビュー開始は各0件。**公開・保全確認後mandatory STOP。**
