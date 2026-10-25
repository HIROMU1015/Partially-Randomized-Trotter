# 研究C・C1：H6修正codecと限定round-trip評価の準備

2026-10-25 JST。起点 `53f45cacf70e23abe911d78f6f005fce5ce4c26d`。
Branch `codex/research-c-c1-h6-codec-preparation-20261025`。

**列符号整理＋実Schurによる修正codecを実装した。人工新規54＋既存回帰80 tests・29 subtestsの最終検証がPASS。det=±1、厳密な−1、πの両側の近傍、一般対称J・複数frame、ffsim形式との互換性を人工データで確認した。保存H6の新しい数値decode・評価は0。rank14／18の20-call経路と未承認契約を固定し、公開照合後mandatory STOP。**

[codec SOURCE](../../src/trottertracks/c1_real_orthogonal_codec.py)／[限定runner処理](../../src/trottertracks/c1_h6_codec_roundtrip.py)／[CLI](../../scripts/run_research_c_c1_h6_codec_roundtrip.py)／[人工codec tests](../../tests/management/test_research_c_c1_real_orthogonal_codec.py)／[人工runner tests](../../tests/management/test_research_c_c1_h6_codec_roundtrip.py)／[未承認20-call契約](research_c_c1_h6_codec_roundtrip_contract_unapproved_20261025.json)／[SOURCE・人工証拠](../../artifacts/research_c_c1_h6_codec_roundtrip/2026-10-25/technical/)／[前段の保存H6診断](research_c_c1_h6_initialization_execution_20261025.md)。

## 方式と再利用

既存ffsim0.0.83のpack/unpack、`c1_ffsim_compat.py`、前段の`c1_h6_initialization.py`と人工試験を確認した。リポジトリの他のSchur利用はcomplex spectrum／phase cluster用で、この実直交logの処理には適合しない。SciPyの既存[real Schur分解](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.linalg.schur.html)と[expm](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.linalg.expm.html)を呼び、その小さいblockの組立だけをtask-localに追加した。DF分解法・optimizer・Gaussian合成器や数値ライブラリを再実装していない。

旧経路は複素`logm(U)`の上三角実部だけをffsimの実反対称パラメータへ採用していた。新しい`real_schur_column_gauge_v1`は次の順序で処理する。

1. finite real float64・正方・直交U、finiteで厳密対称なJを要求する。誤った入力の実部cast・Uの再直交化は行わない。
2. det(U)<0なら最後の列だけ反転してV=UD、det(V)>0とする。Dは対角±1なので各column density projectorは変わらない。
3. V=Q T Qᵀをreal Schurで求める。非zero subdiagonalで識別される2×2回転blockはsigned atan2で角度を作り、個別の−1 real blockは対にしてπ平面を作る。−1までの距離に応じた恣意的cutoffやcomplex-log fallbackを使わない。
4. blockの実反対称BからA=Q B Qᵀを作る。`(A−Aᵀ)/2`で除く丸め成分を記録し、Schur再構成・expm・反対称性の残差を固定1e-10で確認する。
5. 既存ffsimの`antihermitians_to_parameters(real=True)`と`real_symmetrics_to_parameters`で従来の並びへpackし、既存`df_tensors_from_params(real=True)`で復元する。
6. Jは値として完全一致、rank/frame順は不変、復元UはVと一致、元Uの各density projectorとtensorが許容差内で一致することを必須とする。不一致・奇数の−1 block・非finiteは明示的FAILにする。

det=-1の元U自体を実反対称Aの指数で返すことはできない。返すのは記録した列符号を反映したVである。元UとVの直接差と、復元Vの誤差／projectorの誤差を分けて保存し、合法的な列符号の差を不一致として扱わない。

ffsim本体、旧per-matrix bridge、H2/H4/H6の既存SOURCEは変更していない。新codecは明示的に呼ぶ関数であり、旧fitterへ自動適用するglobal patchや新しいfit接続はない。

## 人工・回帰検証

| 検証 | 結果 |
|---|---:|
| 修正codec：2/6次元、両det、π/π±1e-7/π±1e-12/π±1e-14、複数frame・一般J、NumPy/JAX unpack互換、異常入力・破損検出、workerのpartial STOP | 50 PASS |
| eval runner：人工Hamiltonianのend-to-end、同値性／予算FAIL、fresh output、byte preflight | 4 PASS |
| 旧ffsim bridge／初期化診断 | 25 PASS |
| 旧H2/H6経路＋前段診断parent | 15 PASS |
| 旧Stage2 native mock・係数・phase | 27 PASS |
| 旧一般K adapter | 13 tests＋29 subtests PASS |

native mockと実回路を使う旧テストを同一processで混在させた最初の組合せは、既存mockのRZZ gate名期待が合わず1件FAILした。`evaluation_regression_combined.txt/xml`を保全し、旧テストを変更せず適合する独立processへ分離したところ上表の全検証がPASSした。これはH6科学実行のretryではなく、人工回帰試験のprocess分離である。

人工の代表例では、2次元反射のtensor round-trip誤差Fが旧0.2345→修正約1.65e-16、det=+1の6次元π置換例が旧0.1414→約2.46e-16、dense π近傍例が旧0.09067→約2.31e-15だった。これは人工行列の値で、保存H6のframe9等をdecodeした結果ではない。全case・seed・projector残差は`artificial_examples.json`に保存した。

## 保存H6の未承認評価条件

対象は直鎖H6・1.00Å・STO-3G、Nα=Nβ=3、sector400、通常number表現の保存初期J/U、rank14／18のみ。前段run01の`codec/R14_before.npz`／`R18_before.npz`を使い、元`XDF_full.npz`の対応prefixと成分・列符号を照合する。元gや初期分解を新しく生成しない。

旧H6入力SOURCE33c380d8…、保存初期値の同一性診断SOURCE14cf3558…、今回の新codec SOURCEを別々に保持する。κは元gから補正し、scalar、rank、元frame順、CA0.00159、DF予算CA/8=0.00019875 Ha、既存`sector_operation_scale_guard8_proposal_v1`は不変。PFのL_D split・時間窓は非適用である。

新旧の初期J/U、復元J/U、parameter vector・実反対称生成子・列符号representative、frame別残差、tensorF/max-entry、各stateの元物理Hamiltonianへのsector normを保存する。既存`SectorCAR`・`rank_assessment`・`component_check`を再利用し、元の予算内状態が復元後も予算内かを判定する。

| logical unit | 最大 |
|---|---:|
| `saved_codec_load` | 1 |
| `saved_initial_identity` | 2 |
| `corrected_codec`（各frame Schur/expm、pack/unpack、J/projector/tensor同値性を含む） | 2 |
| `tensor_assessment`（rank×before/after） | 4 |
| `snapshots_load` | 1 |
| `sector_context` | 1 |
| `physical_sector` | 1 |
| `sector_assessment` | 4 |
| `saved_sector_replay` | 2 |
| `fidelity_and_budget_gate`（sector operator同値性、差のnorm、両stateの予算） | 2 |
| 合計 | **20** |

codec側9＋sector側11であり、新fit／optimizer iteration／初期DF分解／rank探索／gradient／PF／native build・compile／shots／総費用は0。frame単位のSchur等を個別にcall計上する方式ではなく、契約のcomposite unitに含める。親STOPは子codecのcallを含めた合計を保存する。

component許容差1e-10、tensor残差は前段と同じ`1e-10*max(1,||g_before||F)`、保存sector scalar再現は1e-10 Ha。sector operatorのmax-entry同値性は既存1e-10を使い、spectral norm差も報告する。物理DF誤差は各stateで独立に測定し、既存guard込みで予算判定する。component toleranceをそのままoperator norm上界とみなさない。

将来の順序はbyte/env・資源確認→fresh output→保存初期値同一性→codecのframe/tensor gate→sector4状態→保存before値再現→sector同値性と予算保持gate→証拠公開・STOP。最初のcodec・来歴・数値・予算FAILで依存する新規callを停止し、開始済みworkerを回収してpartial evidenceを保持する。retry/resume0、結果を見た閾値変更・fallback・新fitは禁止する。

```bash
/path/to/eval-python scripts/run_research_c_c1_h6_codec_roundtrip.py \
  --contract docs/research/research_c_c1_h6_codec_roundtrip_contract_unapproved_20261025.json
```

既定はbyte preflightのみ。別の明示科学許可後にだけ`--execute --fit-python <read-only fit-python> --output <fresh>`を付ける。予定fresh outputは`artifacts/research_c_c1_h6_codec_roundtrip/2026-10-25/run01`。チャットの許可だけで足り、承認記録・証明書・新grantは不要。

## 環境・保全・残る制約

既存fit専用環境（ffsim0.0.83、NumPy1.26.4、SciPy1.14.1、JAX0.4.38）とeval環境（Qiskit1.3.0、OF1.6.1、PySCF2.7.0等）をread-onlyで再利用する。依存lockも既存のものを参照し、venv・共有設定を変更していない。byte preflightはSOURCE42、保存入力14、ffsim util固定hashと両環境版を確認しPASS。NPZはhash確認のみで、実H6の新規numeric decodeは0だった。

小さい2/6次元の人工線形代数はBLAS1で、適合する回帰群を独立processで実行した。試験中に別jobの資源利用が増えたが、今回の処理は小さい人工入力に限定し、他者job・共有環境へ変更を加えていない。将来のsector計算は実行時の空き資源から既存`resource_plan`で4独立stateの並列数を選び、BLASと協調する。CPU/RAM/worker/wallの人工capを加えず、dense4096／controlled8192は作らない。未測定のspeedupは主張しない。

生成子は−1の縮退部分で非一意で、branchをまたぐcodec写像の連続性や任意精度の保証は資格付けていない。入力／Schur／expmの残差が固定allowanceに収まらない場合はFAILにする。今回の人工成功から実H6での成功を先取りせず、codec成功を修正後fitの収束・DF予算達成・CDF資源優位と扱わない。実H6の未知の結果は未評価のまま保持する。

契約SHA256 `5e4ba1812e6da0419b43941e239cc06dfeb141507de82749c77cfc33b3fbdfbe`、SOURCE manifest SHA256 `2ee53559c0164077bd8910f9a075c8f750bf82acab00671e365b89ff850cffa7`。旧H6／H2／H4のSOURCE、結果、grant、ledger、marker、STOPに差分なし。人工証拠はlocal実行で、CIや外部独立再現ではない。

ユーザー／研究Cが固定SOURCE・入力・20-call契約を確認し、保存H6のround-trip科学評価を別途明示許可するか判断する。Codexはその許可後にだけ実行する。今回はcommit・push・remote SHA／SOURCE／成果物照合後、**mandatory STOP**。実H6評価や新規fitへ自動進行しない。
