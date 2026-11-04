# RPE D：全18roundのbranch・energy成功保証への接続準備

2026-11-04 JST / LOCAL。技術準備と人工検証を完了。
**実H4の新しい数値評価・保存結果の科学的再解析は0件。次の科学SOURCEは未実行。**

対象はlinear H4／隣接1.00 Å／STO-3G／保存legacy DF12、Partial L_D3、δ=.02、
保存physical stateのnormalized ray・4電子sector70、m0..17の保存q/r/K/N/α/β/ε・E_ref。
Codex CのControlled RTE/Hadamard検証とは独立した専用branchで作業した。

## 準備できたこと

| 接続 | 成果 | 今回の判定水準 |
|---|---|---|
| 局所包含→共通E_ref位相 | 正の保存中心半径、asin円板式、同じ矩形の4頂点による最小代替式。U_cat/PF/RTEの二重加算を排除 | 条件付き十分条件とadapterを準備。実H4での成立は未評価 |
| 全round branch | 同じreference・q倍増の条件を明記し、全隣接対2b_(m−1)+b_m<π、runner-up gap下界を導出 | 理想decoderへの定理・strict判定を準備 |
| Float64 decoder | angle/wrap/root/distanceの一様誤差を含めたstrict条件。元argmin tieの限界を確認 | 一様誤差上界が未取得ならUNKNOWN。人工残差を上界へ格上げしない |
| alias・energy | 初期外部window、最終代表元のstrict margin、exp(−itH)とE_hat=−φ_hat/δ、η_refの一度だけの伝播 | literal E_refと真のphysical energyを分離 |
| 全36軸確率 | 各軸Hoeffding＋exact α和＋union bound。round/軸間独立不要の箇所を明示 | 次回のexact和評価経路を準備。今回保存α和は計算していない |

詳細な証明条件は[PROOF.md](PROOF.md)、次の固定範囲・precision・call・STOPは
[LIMITED_EXECUTION_PLAN.md](LIMITED_EXECUTION_PLAN.md)。
最小実装は[adapter.py](adapter.py)、準備用／将来実行用入口は
[run_limited_validation.py](run_limited_validation.py)、入力identityは[fixed_inputs.json](fixed_inputs.json)。

## 既存証拠・SOURCEの確認と再利用

GitHubの固定commitからRESULT、AXIS_RESULTS、U_CAT_TABLE、NEXT_STAGE_DECISION、実行報告、
SOURCEf490の全体ギャップ、V1e520・V278c4の報告を取得した。
公開byteとローカル固定objectを照合した。全36軸の局所成功をbranch保証へ格上げしていない。
RESULTはSHA256 `410221788791280c7f69ed71ad188c13a7bf0db7167b87aac7a01738fc9ba568`。
V2の保存中心点に対するfloat64二項確率は、新包含区間全体への保証へ移送しない。

関連する既存decoder・accounting・error評価・V1/V2・包含adapterと人工テストを静的に確認した。
V2のunchanged AST loaderで既存`reconstruct_rpe_phase`、`_wrap_phase`、`_phase_distance`、
accountingの`_conservative_nonnegative_sum`を人工テストに再利用した。
既存ball adapterのexact rational・Arb endpoint・Hoeffding分類も再利用した。
V2の文書隣接helperはimportしてloaderだけを利用した。
`src/trotterlib`の回路・信号・cost科学module全体のimportやpipeline起動は行っていない。

新規コードが必要だった理由は、既存accountingがPF有効固有stateとunit radiusを仮定し、
既存V2が保存中心点のfloat64確率を扱うためである。
今回必要な「保存中心の任意半径／座標矩形全体／strict数値margin／未知reference」の含意は、
既存研究runnerを実行せずに小さい純関数で補った。decoderの再実装や既存certified gateの変更はない。

## 人工検証と未実行範囲

新規24件PASS、既存V2人工10件PASS。
正の非単位半径、接線による位相上界、非等方矩形、origin/tie/cut/alias境界、
18人工roundの既存decoder、Float64のseam近傍、uniform envelope欠測・margin消費、
exact α予算超過、外向き和、reference誤差、符号、UNKNOWNとglobal status保全を確認した。
最初の人工testで同精度の二つの包含を大小比較できない箇所を見つけ、
独立referenceだけ320 bitにして上端包含を確認するよう修正した。科学入力・科学精度・判定条件の変更ではない。

新規実H4の半径、位相、branch、alias/energy、α和、二項確率の評価は0件。
保存mean/包含の新しい数値判定、sampling/RNG replay、H/DF/state/operator生成、
回路build/compile、費用評価も0件。旧36軸包含は再計算・編集していない。
runnerのdefault modeは固定byte・schema・literal照合のみで、new_H4_numeric_calls=0を確認した。
人工テストは技術的挙動の確認であり、immutable CI、外部再現、独立科学レビューや科学的GOではない。

再現する人工テスト：

```bash
cd docs/research/rpe_branch_energy_preparation_20261104
<research-python> -m unittest -v test_adapter test_runner
cd ../pr_rpe_v2_branch_success_20261103
<research-python> -m unittest -v test_saved_v2
```

Python3.11.0rc1、NumPy1.26.4、python-flint0.9.0／FLINT3.6.0の既存環境を使用し、環境変更・installなし。
新規人工24件は約1.22秒、既存V2 10件は約0.12秒。speedupの主張ではない。

## 最終energy成功保証までに残ること

1. 実H4の保存中心・εについて、本書の幾何・strict branch・alias/lift・保存暫定目標を判定すること。
2. 保存αのexact和を計算し、実際の成功下界と保存.95条件の関係を判定すること。
3. 実装Float64 decoderの全座標成功域・全候補上の誤差証明。
4. E_refの真のsector基底energyへの保証付き誤差、ground/state同定と外部alias prior。
5. 実state preparation・実PRNG/IID・全18round実回路・compile/noise等の実測定bridge。

1〜2だけを次の限定科学SOURCEへ含める。3〜5は既存結果だけでは保証できず、UNKNOWNを保持する。
保存physical stateとPF有効固有stateを同一視せず、経験的PF誤差をrigorous上界としない。
次のSOURCEが条件付き数学的十分条件を通っても、実装・真の物理energyまで成功保証したとは呼ばない。

## 保全・公開・次の担当

AGENTS.md、PROJECT_MAP、正式共通運用指示2026-10-25 §8.1・§8.4・§10に従った。
原checkoutのdirty編集・他branch/worktree・Codex C・SERVER2 H8、旧SOURCE/raw/grant/ledger/marker/STOPは変更しない。
独立cloneの専用branchで本packageと必要な索引だけをcommit/pushする。
研究結果statusの変更はないため、既存研究概要・科学結果manifest・過去研究ノートは更新しない。
公開SHAと公開本文のbyte一致は公開後に照合し、最終チャットで報告する。

次の担当は**ユーザー**：公開SOURCEに対し、計画1〜2の限定科学再解析を別途明示承認するか判断する。
**Codex（その許可後のみ）**：固定160-bit・1 passの限定検証を実行し、結果とUNKNOWN条件を保存・公開後STOP。
それまでは科学実行を開始しない。

`RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`、
元仮想測定bridgeの`U_virtual_IID_upper=null`、`U_exec_upper=null`を維持。
**本準備は専用branch公開・remote SHA/公開成果物照合後mandatory STOP。**
