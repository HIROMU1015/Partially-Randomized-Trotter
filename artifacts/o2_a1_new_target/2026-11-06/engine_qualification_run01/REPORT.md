# A1 N2補完：総engine誤差の一回限定数値認証

2026-11-06 JST、SERVER1 GPUA100-gleap-01。
**認証計算は成功。固定実装・理想モデルの条件下で、全positive-support path・両軸の総engine座標誤差を上から抑えた。公開照合後mandatory STOP。N3は開始していない。**

対象：保存linear H4、1.00 Å、STO-3G、legacy DF12、8 system qubits、全4電子number sector、L_D=3、m0/q1/δ=0.02/r1/K2、coefficient_atol=0.0、X/Y。
N1 actual finite-grid Qとselected basis policy、N2保証groundを再利用する。元未近似H、実backend/noise、最終費用やCV改善の保証ではない。

## 数値結果

以下の小数は正の有理数上界を外向きに丸めた表示。厳密な値はJSONを正本とする。

| 項目 | 取得値・上界 |
|---|---:|
| unique local gate keys | 4,315 |
| δ*（局所operator差上界の最大） | ≤1.2537168e-16 |
| L_max（最大gate列長の保守上界） | 955 |
| E≥(1+δ*)^L_max−1（外向き上側評価） | ≤1.1972995e-13 |
| 4E（gate生成＋最終正規化の座標上界） | ≤4.7891979e-13 |
| b_state | N2保存の厳密な有理数上界をそのまま再利用（表示≤7.8078960e-7） |
| 53-bit grid座標項 | 2/2^53=1/4503599627370496 |
| **総b_engine** | **≤7.8079008e-7** |

最終証明：[ENGINE_ERROR_CERTIFIED.json](ENGINE_ERROR_CERTIFIED.json)。厳密な総上界は

```text
45204656909390546910567498882300919608528077686769531894033658092337249
/57896044618658097711785492504343953926634992332820282019728792003956564819968
```

N2 `STATE_RESULT.coordinate_state_error_upper.exact` と、取得した同じE、grid項をexact Fractionで一度合成した。
固定24d6bdfの関数は旧計画の表示上界0.000000781を使うため、[raw UNIFORM_ERROR](UNIFORM_ERROR.json)にはその保守的な別総和も残る（表示≤7.8100048e-7）。
旧SOURCEとraw出力を変更せず、今回のユーザー指定に対応する厳密state上界での最終和を上記別ファイルに保存した。qualificationの再実行やstate保証の再計算はしていない。

## 固定SOURCE・入力・実行範囲

| 項目 | 固定・回数 |
|---|---|
| 認証kernel・計画 | `24d6bdfa11ca5964c83b2639f060f4655ab1d4be`（変更なし） |
| 最小runner SOURCE | [`fefa5a09d33789ebe8e1801c513c9776eedc7980`][RUNNER] |
| N1結果 | `744e1a09661ebba6841fe5faf456f88c41c47780` |
| N2結果 | `5e9bd08068738cdf1070bdc43bee8194b113ae33` |
| N2科学SOURCE | `a4141795bd3e23670612d0f0d4a43678457a1a37` |
| qualify_fixed_bundle | **1回** |
| saved bundle復元・一様qualification | 各1回 |
| D half-step局所列 | 4件、各1回 |
| aggregate scalar列 | 1件 |
| stored full/support basis catalog | 337件（inverseも対象） |
| candidate local gate entries | 14,644件、keyで重複除去後4,315件を各1回interval検証 |
| 厳密state上界での最終会計 | 1回 |

[SOURCE](SOURCE.json)、[EXECUTION_SOURCE](EXECUTION_SOURCE.json)、[INVOCATION](INVOCATION.json)、[STOP](STOP.json)を保存した。
Python3.12.3、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、mpmath1.3.0。入力前に固定538 SOURCE・全依存環境とN1/N2の25保存ファイルを照合した。
[実行仕様・command](../../../../docs/research/o2_a1_engine_qualification_20261106/EXECUTION.md)と追加runner hashを公開・照合後、cleanな専用worktreeから起動した。

実stateへのgate作用/Born評価、実path build/全列挙、NPZ load、N1 preparation/CDF、N2 solve/eigh、RNG sampling、Hadamard shots、train/cal/prod、compile、CV variance/N/費用はすべて0件。
禁止entry pointはfail-only guardで拒否した。実行の科学retryは0。guardはgate値・計算処理を置換していない。
実行の経過は約19.34秒だったが、性能・費用比較の結果とはしない。[資源観測](RESOURCE_OBSERVATION.json)を保存し、CPU/RAM/workerの人工上限は新設していない。

## 全supportへの保証とgate証明

[GATE_CERTIFICATES.json](GATE_CERTIFICATES.json)に全4,315 keyの名前・exact angle hex、raw matrix hash、exact rational matrixとinterval operator誤差上界を保存した。
[UNIFORM_ERROR](UNIFORM_ERROR.json)にはD half-stepのliteral列・scalar列・catalog/capの根拠を保持する。

固定計画の有限gate集合は、全保存full/support basisとinverse、D block局所列、n=0/2・sign±・support長1/2のRTE variant、event/product phaseとwrapperを含む。
L_maxはbasis cancellationを使わない構造的上界なので、少数pathの観測最大値ではない。実pathは列挙していない。
各理想local gateはunitaryで、δ*と列長からproduct vector差をEで抑える。E<1を確認したため、全非零入力の近似出力normの正値も保証する。
Born比による最終正規化を含めた座標差は4E以下。N2のstate誤差と条件付きgrid項を加えると、N1 Qで平均しても同じ上界が使える。

この結果は、固定SOURCEのexact Fraction局所contraction・Born平方和/除算・range検査と、独立uniform53-bit wordを仮定する同じ理想測定モデルに限る。
適用・Bornの算術丸め項0はexact有理数実装によるもので、未知のBLAS誤差を0と仮定したものではない。高速float kernelへ置換した場合は流用できない。
mpmath intervalは固定版の実装を信頼する数値認証で、形式検証済みlibrary/CI/外部再現という認定ではない。実機・noise・compiler誤差は対象外。

[保存値correctness](../../../../docs/research/o2_a1_engine_qualification_20261106/CORRECTNESS.json)では、全matrixの有理数表現・raw hash、δ*最大値、厳密state上界と総和、禁止call0を確認した。gate生成・intervalは再計算していない。
Fraction保存ではIEEEのsigned zero符号を区別しない3個の2×2 matrixについて、零符号bitだけの復元witnessを保存して元raw hashも照合した。科学値・SOURCEは変更していない。
最小runnerの人工3件PASSと既存kernel人工72件の証拠を再利用した。実科学結果はlocal取得であり、独立外部再現/CIではない。

## 未解決事項・N3の条件

未採択の1/32を正式受理条件として用いていない。今回の上界取得は、精度・失敗確率・測定予算・費用指標の採択や、N3の実行許可ではない。
N3には、同じinput/law/policy/state/ordering/phase・exact engineとwordモデルをbindし、engine内部情報をCV側へ漏らさずfresh bitだけを返す測定facadeと独立streamを接続する必要がある。
ε/α、条件付き取得法、train/pilot/cal/prod予算・range guard、必要な価格profile等を採択し、RNG/測定/full build/compile/価格評価を含むN3の有限scopeを別途明示許可する必要がある。
実stateのpath信号、実Born値、測定統計、学習rank/variance、shot数・費用・CV改善は今回未取得。
g2/g3の希少性、Control集合・pilot数は変更していない。N3へ自動進行しない。

[保全記録](../../../../docs/research/o2_a1_engine_qualification_20261106/PRESERVATION.json)：元mainのHEAD/statusと既存2,136ファイル、旧A1/N1/N2 artifact51ファイル、固定538 SOURCEとstate/spectral witness、既存STOPを保全した。共有環境・他ユーザーjobを変更せず、C1凍結を維持する。
成果物を専用branchでcommit/pushし、公開本文/blob/treeとremote SHAを照合後、**mandatory STOP**する。

[RUNNER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/fefa5a09d33789ebe8e1801c513c9776eedc7980
