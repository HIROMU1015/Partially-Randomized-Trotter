# 次の限定科学検証：保存36軸から理想branch・literal energyへの接続

2026-11-04 JST。**PREPARED / NOT EXECUTED。公開SOURCEへの別途明示実行許可が必要。**
今回の指示は技術準備と人工検証までであり、本計画の科学実行は許可していない。
ユーザーの実行指示自体を許可とする。承認証明書、grant発行、承認原文の保存・hash確認は要求しない。
旧研究のgrant、ledger、marker、one-shot、STOPはそのまま保全する。

## 1. 固定SOURCEと対象

専用branch `codex/rpe-branch-energy-preparation-20261104` の本packageを公開するcommitをSOURCEとする。
最終チャットでその40桁SHA・permalink・remote照合結果を提示する。
実行時は**そのSHAのcheckout**を用い、`frozen_source()`が以下の5ファイルをHEADと照合する：

`adapter.py`、`run_limited_validation.py`、`fixed_inputs.json`、`PROOF.md`、本計画。
入力と直接再利用するSOURCEのcommit/path/blob/SHA256/byte数は
[fixed_inputs.json](fixed_inputs.json)の10項目で固定する。全環境のbitwise同一性監査は要求しない。

科学条件はlinear H4／隣接1.00 Å／STO-3G／保存legacy DF12、Partial L_D=3、
δ=binary64 .02、8qubit・4電子sector70、保存physical stateのnormalized ray、m0..17。
q/r/K/N/α/β/ε、E_ref、SOURCEは既存の保存値。新しい条件・候補・配分は作らない。

| 入力 | 固定commit | path・役割 |
|---|---|---|
| RESULT | d2e7a7db683e116243eae061c0e4c87865054e9e | `artifacts/rpe_18round_ideal_enclosure/2026-11-04/run01/RESULT.json`：m0..16原包含。SHA256 `410221788791280c7f69ed71ad188c13a7bf0db7167b87aac7a01738fc9ba568` |
| AXES | 同上 | `docs/research/rpe_18round_ideal_enclosure_execution_20261104/AXIS_RESULTS.tsv`：36軸の保存literalと判定。SHA256 `0346da604f125e41c5afd073a91e8eb46afe4c2e1c950937d5de9eb33e588e0d` |
| FIXED_ROUNDS | f4905fff592351d33ba07b927892fb9532c99711 | `docs/research/rpe_18round_ideal_enclosure_preparation_20261104/fixed_inputs.json`：保存18roundとprovenance。SHA256 `72644832c1c8a159809efc30ff2909aaa576e49d69197d7ee007bf8a19d57673` |
| M17_RESULT | 34840ce9bf81d985b32764d9bc9d77645ed52e59 | `artifacts/rpe_m17_rigorous_mean_enclosure/2026-11-03/run01/RESULT.json`：m17原包含、再計算なし |
| GRID／WP01 | 4591b8d75a2a6a55c74a51225f0c61648023c8ee | manifest記載の元grid／WP01 JSON：E_ref・δ・保存全体予算・暫定目標のliteral照合のみ |
| DECODER／ACCOUNT | 同上 | 既存decoderと外向き和のSOURCE固定。新規decoder作成・cost呼出しなし |
| V2_LOADER | 78c49d0b5f1d71e63b5959e78813237923bbc648 | V2の純関数AST loader。確率解析runner本体を起動しない |
| BALL | 18fffce48e33136b91eb6ac5a6f9a5f77a08d36f | 既存Arb exact rational・endpoint helper。行列生成関数は呼ばない |

STATE/H/DF/basis/演算子NPZ等はprovenanceを上記から継承する。
新たにdecode、normalize、固有値solve、state-action、信号生成する入力ではない。
V1 `e520e6e396f7e0def9e522ea1cff58eb10ddfa8f`、V2 `78c49d0b...`、
全体ギャップ、旧包含計画の歴史的statusは変更しない。

## 2. 数値条件と評価順

Python3.11、NumPy1.26.4、python-flint0.9.0／FLINT3.6.0、IEEE binary64を使う。
新しい科学的scalar条件は**160 bit・1 pass**のArb/acb包含で評価する。
精度を上げて通るまで再試行する処理、adaptive subdivision、結果に合わせた閾値変更はない。
96-bit旧包含のprecision・endpoint・原結果を変更しない。

1. byte identityと保存literal・36軸判定を照合する。既存包含・Hoeffding判定は再計算しない。
2. μ₀・E_ref・δは元のbinary64が表すexact rationalへ変換する。
   ε・αは旧局所包含で使った保存decimal文字列のexact rational。
   φ_ref=−E_ref δをexactに作り、q_m倍もexactな整数演算で作る。
   exp/arg/asin/sqrtはArb/acbで外向きに評価し、最終endpointはexact dyadic rationalで保存する。
3. 全18roundでρ₀、D、s、b_diskを評価する。同じ矩形の四隅によるb_rectangleも事前固定して評価し、
   有効な上界の小さい方を選ぶ。円板も右半平面法も証明できなければそのroundはUNKNOWN。
   選択されたb_mと保存β_RPE=.4を**strict**に比較する。
4. qの完全な倍増列を確認し、全17隣接対でπ−2b_(m−1)−b_m>0を外向きに判定する。
   実Float64 decoderをH4保存平均へ新たに実行しない。
   一様a/c/dがないため実装decoderの判定はUNKNOWN固定とする。
5. 共通literal referenceのalias windowと最終代表元のstrict境界marginを評価する。
   理想decoderの条件が通ったときだけb_17/q_17から実数energyへ変換する。
   保存暫定目標`.00015936001019904` Haとstrict比較する。
   βと目標は元SOURCEのbinary64が表すexact rationalとして比較し、浮動小数点の許容差は加えない。
6. 保存decimal αをexactに36項加算し、外向きbinary64表示も保存する。
   alpha_total=.05とexactに比較する。元binary64 αのexact和・外向き表示は別の診断として保存する。
   decimal条件が通らなかったときにbinary64診断へ主判定を切り替えない。
7. η_refは未取得のためnull、physical-ground-energy判定はUNKNOWNとする。
   外部alias priorの正当性、物理ground/state preparation、実回路、固定PRNGのIIDも認定しない。

幾何・branch・alias/lift・保存phase/energy/alpha条件が全て通った場合の最高判定は
`CONDITIONAL_IDEAL_DECODER_SUFFICIENT`。これは保存state rayのideal categoricalモデルと
**数学的decoder・literal E_ref**に対する条件付き十分条件である。
どれかが未確認なら`UNKNOWN/STOP`。十分条件の不成立を科学的失敗の実証とは呼ばない。

## 3. 必要な科学call・禁止call

予定する新しい科学再解析は、18個の保存complex中心についてのscalar幾何セル各1回
（各セルの円板式・4頂点式）、17個の隣接branch条件、1個のalias/energy集約、
36項decimal exact和と同じ36項binary64診断和の各1回。
入力byte/literal照合は科学callではない。式内の通常の包含演算はこの固定セルの内部処理である。

新mean包含、power/expectation、CDF/law replay、二項PMF、sampling、RNG replay、
H/DF/state/basis/operator生成・decode・solve、回路build/compile、価格・費用評価は**全て0**。
U_catの再計算、旧結果への書込み、Full L_D12、Codex C／他worktree／SERVER2 H8への操作も0。

## 4. 資源・高速化

本経路は70×70行列を扱わず、q個のcandidate列挙も不要。qはscalarの整数係数にだけ使う。
既存V2のO(q)候補decoderや(Nc+1)(Ns+1)確率格子を新評価に持ち込まない。
18個の小さいArb幾何セルだけなので、現時点でprocess pool/GPUへの転送が必要な支配workloadはない。
実行前にCPU affinity・MemAvailable・runtimeを観測して記録し、必要資源を使える状態で実行する。
CPU/RAM/worker・phase/total wall-timeの人工上限は設けない。共有設定・他ユーザーjobは変更しない。
GPUの性能比較はこのscalar経路の必要条件ではなく、speedup倍率も主張しない。

## 5. 出力・STOP・実行入口

結果namespace案は`artifacts/rpe_branch_energy_connection/2026-11-04/run01/`。
**今回そのdirectoryや科学RESULTは作成していない。**
別承認後の初回実行で新directoryを一つ作る。既存outputは上書きしない。
SOURCE/input identity・runtime不一致、非有限、欠測、未保証入力、例外ではUNKNOWN/STOP。
例外はSTOP.jsonを残して停止する。判定不能条件はRESULT内のUNKNOWNとして保持して集約する。
同じrunの自動retry/resume、精度追加、予算/decoder/対象変更、別候補への進行はしない。
有効な結果の追記・公開に必要な技術作業はまとめて行い、公開照合後mandatory STOP。

準備の再確認（科学評価なし）：

```bash
<research-python> docs/research/rpe_branch_energy_preparation_20261104/run_limited_validation.py
```

以下は**公開SOURCEへの別途明示実行許可後のみ**：

```bash
<research-python> docs/research/rpe_branch_energy_preparation_20261104/run_limited_validation.py \
  --run --output artifacts/rpe_branch_energy_connection/2026-11-04/run01
```

この次段でも`RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`、
元仮想測定bridgeの`U_virtual_IID_upper=null`、`U_exec_upper=null`を保持する。
一様Float64丸め証明、真のground reference誤差やstate preparationの追加評価は、
このSOURCEの外に残す。結果に合わせて自動的に実行範囲を拡張しない。
