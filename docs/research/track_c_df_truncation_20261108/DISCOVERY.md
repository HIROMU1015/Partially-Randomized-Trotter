# 既存成果・SOURCEの適合調査

対象はLinear H2/H4/H6、1.00 Å、STO-3G、RHF積分に基づくDF表現。
本研究の結果は同じdirectoryのREPORTを正本とする。これは著者による実装・適合調査であり、独立した科学レビューではない。

## 入力の再利用

| 分子 | 採用した既存入力 | 今回の評価空間 | 転用しないもの |
|---|---|---|---|
| H2 | `5385c6ef01879ee8a70734bfc0d5ae8999dd6186` の `artifacts/research_c_c1_h2_h6_science/2026-10-25/H2-run01/{input,XDF_full}.npz` | Nα=Nβ=1、4次元。4 system qubits | 旧Pauli価格、旧T=.8のshot、別compilerのnative資格 |
| H4 | 基点 `c5a3c102199fe97f4d5c21837c059ae2684f5ef2` のpublic MODEL replay | N=4、70次元。8 system qubits | DF12・LD3・218 componentsを他分子へ移すこと |
| H6 | 同じ5385c6の `H6-run01/{input,XDF_full}.npz` | Nα=Nβ=3、400次元。12 system qubits | CDF14/selected XDF14のown-DF groundと旧CA=.00159の資源値 |

H2は保存full XDF3、H6は保存full XDF18のJ/Uと元frame順を使用する。H4は元のDF12とgeneration orderを維持する。H2/H6の数値入力の生成条件は同commitの`input_identity.json`と元SOURCEに遡れる。新しいSCF・DF fitは行っていない。

H2/H6の元Hは保存h/g/nuclear constantから既存のunthresholded spin liftと疎CAR作用で組む。H6ではfull Fock空間の疎演算子をsectorへ制限し、4096次元の密行列は作らない。回路検査は8192成分のcontrolled statevectorを用い、native gateを途中でspin sectorへ射影しない。

元の数値行列には微小な反Hermitian丸め残差がある。最終対象はそのHermitian部分であり、H4の既存保証と同じ規約である。原配列・保存SOURCEは変更しない。最初のgeneric資格評価から再利用する行には、対象行列とnormalizationの移送diskを付ける。

H2/H6のfull DFから再構成した元Hとの差の数値ノルムは約6.79e-16 / 1.73e-9 Haだった。H4は約5.67e-14 Ha。この観測値だけで保証とはせず、最終的には元Hのground入力に対する信号を直接囲む。連続積分Hamiltonianの無条件包含とは区別する。

## B0/B0Iから再利用する知見

確認した主な一次証拠：

- `fbfbfd21a252a4a32f52e3b40e50acd91b3c626a` のB0比較SOURCE、`best_by_precision.csv`、`data_archives.json`、`batch01_evidence.tar.gz`内の`p027_q1/target.json`。
- `src/trotterlib/h4_pauli_b0_comparison.py`、`h4_pauli_b0_execution.py`、関連account関数。
- `docs/research/pauli_jw_resource_evaluation_summary.md`、`pauli_h2_h4_h6_paper_evidence_synthesis_20261101.md`。

旧B0/B0Iは保存abs係数降順・label tie-breakのPauli prefixを保持し、constantを別枠で常に残す。したがって同じprefix/qのB0とB0Iは重複した独立方式ではない。旧H4では106構成を調べ、精度に応じて保持prefixとqを選んだ。固定prefix27の安価な回路だけでfamilyを代表しない点を今回も採用した。

元rawは保存binary64 operandsの積、未再正規化のstate、元full-H referenceとのsigned signal bias、数値包含幅、B=1とshot headroomを区別していた。p27/q1にはcosine biasによる不適格判定が保存されている。旧`actual_rho=null`、`formal_certificate_accepted=false`を新しいnative保証へ格上げしない。

旧研究は主にT=.8のPauli-JW信号課題、5種類の信号精度、opt1 compilerである。今回のDF・18roundの価格、N、精度達成値には流用しない。再利用するのは有限prefix探索、constant保持、元H state/referenceへの直接信号比較、biasとNと単価を同時に見る評価方針である。

## 関連研究との重複範囲

- Git refs/worktree metadataとGitHub branch searchで、同じDF入力・18round・精度・compiler Aによる今回の3方式比較は取得できなかった。これは会話履歴や別環境の非公開runが存在しないことの証明ではない。
- O1 preflight `aeb9d1847` は辞書変更M1とcore残差補償M2の調査。元Hを保つ研究計画であり、今回と同じTruncated PFの18round結果ではない。
- 研究CのH6 conditionfix `d166c75c407f4b2f74f239018fe4bbc5cf9dd390` はCDF14・LD4・own-DF系列のfit/数値条件修正であり、保存full XDF18の元H groundを用いた今回の比較とは異なる。原研究のrun・契約・STOPは再開しない。
- XDF8/9、CDF7、旧H6 CA資源は表現・ground・精度・compiler資格が異なる。入力や関数の参考と結果の転用を区別した。

他サーバーへのSSH、他者へのメッセージ、agent履歴の横断検索、他Trackの研究再開は行わない。非公開の他チャットの実行状態は未確認として残す。

## 主な再利用関数と必要なadapter

| 既存SOURCE | 再利用・変更の境界 |
|---|---|
| `track_c_closed_p3_inputs.load_model` | H4 public replay。private原本pickleは公開しない |
| `c1_stage1.physical_sector/corrected_kappa` | H2/H6の元Hとone-body correction。保存積分を使用 |
| `df_trotter.ops.U_to_qiskit_ops_jw` | 保存Uから定義付きGaussian gatesを生成。opaque unitary synthesisは使わない |
| `c1_general_k_adapter.general_k_coefficients` | 同じI/Z/ZZ展開式。切り捨て定義と丸め会計のためadapterでは保存binary64からexact rational係数を作る |
| `track_c_d_internal.schedule` | 既存S2/Yoshida4、D4、Dtunedの作用順。H4の係数を他分子へコピーしない |
| `track_c_mixed_formula.cubic_generator` | 分子・splitごとのD2/D4係数からDtuned候補を生成。finite hの成功は別に検証 |
| `track_c_compiler_a.compile_reference` | 同じQiskit1.3.0、rz/sx/x/cx、近似commutation・angle cutoffなし |
| `track_c_compiler_a_18round` | 局所interval lowering。D/R本体のXX/P contextとreadout contextを検査 |
| `track_c_source_guarantee` | spectral residual、polar distance、binary powering、phase disk。H2/H6と任意の切り捨て係数へ接続 |
| `track_c_bottleneck_basis.partial_ops` | H2/H6から新しいsingleton/pair QRを作る。旧625件catalogueは移送しない |
| G7 `CanonicalGenerator`、Cの`CachedCanonical/DigitalSampler/certify/angle` | 同じClosed P3 law・256-bit proposal・weighted estimator。新sampling familyではない |
| `track_c_closed_p3_joint_opt_alpha.discrete_allocate` | 同じ36軸のdirected十分trial式、exact α和、整数局所改善 |
| `rpe_four_round_phase_validation.reconstruct_rpe_phase` | phase/qだけを渡す。E0をdecoderへ渡さない |

新adapterは`track_c_df_truncation.py`と`track_c_df_truncation_pr.py`。前者はDF prefix/小係数候補、元H信号、modular価格、数値保証、独立経路の照合を接続する。後者は分子固有のsplit/BCH/QR、同じG7、weighted範囲、価格のtraining/holdoutを接続する。

## 回路と誤差の境界

DF blockはB†、controlled diagonal、Bの順に実装し、identity/constantはcontrolの相対位相に残す。identityだけ残るblockはGaussian変換を省ける。これはFull/Truncatedに共通の正確な合成であり、元Hamiltonianの別のenergy補正ではない。

I/Z/ZZ係数と角度合算はexact rationalで定義し、実native angleへの一回の丸めを会計する。既存GraySynthはparity topologyにだけ用い、数値angleのmodulo処理を避ける。prefix/middle/suffixまたはD/R/D junctionを個別に実compileし、そのnative moduleを連結する。RZ/CX/sizeは実装の加法値、depthはserial上界である。

H2/H6のGaussian誤差はone-particleの区間作用とexterior powerから囲み、微小なspin leakageも含める。spin-sector外へ掛かる対角作用には係数L1を使う。QR採用は新しい列/source/native certificateに限り、未登録の大きいsupport unionはfull basisへ戻す。

Closed P3ではY=(全event weight積)×Hadamard outcomeを使う。十分range LでY/Lを[-1,1]へ囲み、rho、beta、Nを求める。通常の無重みHadamard平均と同一視しない。量子eventのacceptanceは1、zero-fillは0。有限bit law、root/angle、weight丸め、π生成、SOURCE/native/数値diskを分離する。価格MCのrare/no-rare条件付けは同じ256-bit lawに対する整数処理であり、量子sampling familyの変更ではない。

最終的な科学的資格・負結果・価格・データ分離・STOPはREPORTおよびSOURCE/raw索引を参照する。
