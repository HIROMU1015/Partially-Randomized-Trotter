# Full-H κ2 P-1（m3、新controlled ZZ）：run01・run02ともSTOP

2026-11-14 JST。実行したのは主agent（Claude Code）で、検証は作者自身によるもの（独立した査読ではない）。

| run | 状態 | 内容 |
|---|---|---|
| run01 | `STOP_WIRING_CHECK`（科学計算前） | 実行treeのsparse checkoutに`references/`が無く、子processがimport時に`FileNotFoundError`。sampling・compose・compileは0（§5）。出力は無変更で保全 |
| run02 | `STOP_WIRING_CHECK`（科学計算後） | 利用者がチャットで再実行を承認。科学計算を1回実施し、**作用検査7件はすべてPASS、新ZZは有効、compile 2件**。ただし固定callerの`scalar_once`判定が`false`となり、`ok=false`でSTOPした（終了コード2）。retryなし |

**run02のSTOP原因（STOP後の構造診断）**

- `scalar_once`は「ancillaの単一qubit `p`がちょうど1個で、その角度が−scalar·T」を要求する。
- 実回路のancillaには`p`が4個あった。
  - 最後の1個が`p(0.2651822507334529)`で、−scalar·T（scalar −0.33147781341681615、T=0.8）にbit一致する。scalarは1回だけ適用されている。
  - 残り3個は`p(π)`である。これはpinned provider builder（c31ef424 `df_rte_qiskit.py`）が出すevent位相である。builderはrequestごとにevent位相（±1）を累積し、ancillaのPhaseGateを最大1個付ける。
  - 今回のtrajectoryではrequest 0・3・7がそれぞれorder-2 event（位相−1）を1個ずつ含み、それが3個の`p(π)`になった。
- 4個の角度の多重集合は、builderの規則から期待される値と一致した（X・Y両方）。根拠：[RUN02_SCALAR_ONCE_DIAGNOSIS.json](RUN02_SCALAR_ONCE_DIAGNOSIS.json)、[.py](RUN02_SCALAR_ONCE_DIAGNOSIS.py)。保存回路の構造を読むだけで、simulation・compile・samplingはしていない。
- event位相とscalarを含む独立occupation oracleとの比較（`independent_occupation_vector`）は1.2e−14で通過している。
- 結論：**判定式の仕様の欠陥による偽陰性**と判断する。order≥2 eventを含むrequestを想定していなかった。toy testはorder-2 eventを含まず、ancillaの`p`が1個だったため検出できなかった。
- ただし固定計画のgateは不成立であり、**判定を書き換えてPASS扱いにはしていない**。callerは価格record・sidecar・IMPLEMENTATION_EVIDENCEを作っておらず、**Cへの価格HANDOFFは発行していない**。

## 1. run02の実行

| 項目 | 内容 |
|---|---|
| 承認 | 利用者のチャット指示（2026-11-14「このチャットで対応して再実行してよい」）。run01のSTOPで提案した2点（実行tree・出力label）とimport preflightを本担当で対応し、1回再実行した |
| SOURCE | caller commit `72d4f4ff2`（branch `codex/d-full-h-kappa2-p1-results-20261114`、push済み）。固定SOURCE `04bead115`からの差分は2点だけ：出力label（`run02`）と、子processのimport-only preflight。科学条件は変えていない |
| 実行tree | 同worktree。sparse checkoutに`references/`を追加済み |
| 環境 | h6 venv（Python 3.11.0rc1、Qiskit 1.3.0、NumPy 1.26.4、SciPy 1.14.1、rustworkx 0.18.1） |
| 事前test | P-1 parent 4件（import preflight 1件を追加）、P-1 circuit 4件、S-1 caller 7件：すべてPASS |
| metadata preflight | PASS。configuration `54ac3eb3…`、`FULL_H_4de6130f0d3a66e5d695`、cohort `D_FULL_H_KAPPA2_NEW_ZZ_M3_WIRING_V1`、8 seedは固定値（run01と同じ） |
| 子processのimport preflight | PASS：`install()`→provider 9 moduleのbyte照合→実入力経路23 moduleのimport→契約SOURCE_PATHSの欠落0→保存入力2件・保存context binding・reference bytesのSHA-256一致 |
| 実行 | `python -B scripts/run_full_h_kappa2_p1.py --execute-approved-p1`を1回だけ実行。2026-11-14T08:49:56Z〜08:52:57Z |
| 出力 | [run02/SOURCE.json](run02/SOURCE.json)、[run02/P1_RESULT.json](run02/P1_RESULT.json)（callerの出力、無変更）。privateは`/home/AbeHiromu/projects/d-full-h-kappa-20261114-private/P1-KAPPA2-M3/run02/`（child.json、events、result、X/Yのwrapper・compiled qpy、compile record） |

**実行件数と資源**

| 項目 | 値 |
|---|---|
| Full-H context・preparation | 各1（保存入力の復元1、入力再生成0、新規reference 0） |
| normal trajectory | 1 path、index0、8 request、568 event（order0：565、order2：3） |
| 新ZZ compose | 1 |
| legacy replay compose | 1（parity比較用。compileしない） |
| measured X/Y wrapper | 2 |
| compile | **2**（cache hit 0） |
| retry | なし |
| 時間 | wall 3分0.7秒（callerの計測176.3 s）。うち`cost_trial`（sampling・compose・statevector検査・2 compile）は87.9 s。axisごとのcompile時間はcallerが記録しておらずUNKNOWN |
| CPU | user 195.3 s、sys 13.1 s |
| peak RSS | 598,508 KB（約613 MB、子process） |

## 2. 作用検証（index0）

| # | 確認 | 結果 |
|---|---|---|
| 1 | control=0側が恒等（`low_control`） | 1.19e−14（許容1e−9）PASS |
| 2 | high-control側が独立occupation oracleと一致（`independent_occupation_vector`） | 1.18e−14（1e−9）PASS |
| 3 | high-control leakage | 0.0（1e−9）PASS |
| 4 | Hadamard testの期待値と符号 | X 1.14e−15、Y 1.73e−16（1e−9）PASS |
| 5 | compiled回路の作用 | X 9.87e−12、Y 8.36e−12（1e−7）PASS |
| 6 | 新controlled ZZ builderが有効、旧`crzz`なし | builder `CONTROLLED_ZZ_CX_CRZ_CX_V1`、`crzz` 0、`crz` 568 |
| 7 | `instruction_parity` | PASS：置換したcontrolled ZZは172、命令数は旧56,555→新56,899（+344＝CX 2×172） |
| 8 | one-body成分のsample | 205 application |
| 9 | scalarと一回適用 | scalarの値は期待値と一致（−0.33147781341681615＝定数2.29310124732＋抽出identity −2.6245790607368162）。**callerの`scalar_once`判定は`false`**（上記のとおり判定式の欠陥。ancillaのscalar位相は1個で、最後に位置する） |

global phaseとancillaの相対phaseは落としていない。checkはすべて`phase_alignment=False`である。

**その他の整合**

- 実trajectoryの`log_B`は0.49796390531900514。S-1のm3（0.4979639053）と一致する。
- context：λ_R 21.0279、成分368（one-body 8、確率質量0.347）、sector 36次元、行列再構成誤差1.2e−15。

## 3. compiled価格（観測値。STOPのため未資格・HANDOFFなし）

| 項目 | X（cosine） | Y（sine） |
|---|---:|---:|
| 全compiled RZ数 | 92,448 | 92,448 |
| compiled CX数 | 27,719 | 27,719 |
| RZ depth | 26,079 | 26,079 |
| CX depth | 13,996 | 13,996 |
| total depth | 55,352 | 55,352 |
| size | 176,866 | 176,866 |
| qubit数 | 9（system 8＋ancilla 1） | 9 |
| circuit fingerprint | `8de2cd2bceea0ac0…` | `f5448c774286486e…` |
| compile前の命令数 | 56,902 | 56,903（`sdg`を含む） |

- 共通：compiler hash `1239304c14c65cde…`（basis `rz/sx/x/cx`、optimization level 1、seed 17、Qiskit 1.3.0）。native `af284d85`、provider `c31ef424`、lowering `01e068af`、cohort・seedは§1。
- compile前の内訳（X）：`p` 37,546、`xx_plus_yy` 18,432、`crz` 568、`cx` 344、`cz` 9、`h` 2、`measure` 1。
- これは**単一trajectoryのdirect compiled価格**である。Full-Hの期待価格\(\mathbb E[C]\)、平均event単価、価格modelの校正点、Gには昇格させない。run02は`ok=false`なので、callerは価格recordとsidecarを作っていない。

## 4. 報告項目の状況

| # | 項目 | 状況 |
|---|---|---|
| 1 | 実H4 P-1の作用検証 | 数値gate 7件はPASS。callerの総合判定はSTOP（`scalar_once`の偽陰性） |
| 2 | X/YのcompiledのRZ・CX・depth等 | 観測済み（§3）。STOPのため未資格 |
| 3 | SOURCE・seed・physical IDの照合 | PASS |
| 4 | scalar、one-body、新ZZ lowering | 新ZZ・one-bodyはPASS。scalarは値と配置が正しいが、固定判定はfalse |
| 5 | sidecarとCへの価格HANDOFF | **未作成**（STOPのため。builderは未実行） |
| 6 | 実行件数・時間・peak RSS | §1 |
| 7 | 残るUNKNOWN | 下記 |

**P-2までに不足する証拠**

- P-1の受理：`scalar_once`判定の扱いをユーザーが決める必要がある（§6）。
- 受理後のdirect価格record、source-bound sidecar、CへのHANDOFF。
- 単一path・単一seedの価格であり、law平均・分散は未評価である。order-2 eventの出方による価格のばらつきもUNKNOWN。
- axisごとのcompile時間（未記録）。

## 5. run01（参考、保全）

- 固定caller `04bead115`のまま1回実行した。sparse checkoutに`references/h46_m5_tol_pilot_inputs_v1.json`が無く、子processの`context('H4')`内のimportで停止した（`FileNotFoundError`）。
- 親processの照合と、子processの`install()`・provider照合・`setup()`は通過していた。sampling・compose・compileは0。
- 出力：[run01/SOURCE.json](run01/SOURCE.json)、[run01/P1_RESULT.json](run01/P1_RESULT.json)、privateの`…/P1-KAPPA2-M3/run01/job.json`。いずれも無変更。
- run02では、`references/`をcheckoutに含め、子processのimport preflightを追加した。

## 6. 次の判断（ユーザー）

run02のSTOPは、科学条件の不一致ではなく判定式の欠陥による可能性が高い。ただし固定gateの扱いを変えるので、ユーザーの判断が必要である。

1. **判定式の修正（推奨）**：`scalar_once`を次のように直す。
   - ancillaの`p`のうち、角度が−scalar·Tのものがちょうど1個で、最後に位置すること。
   - 残りの`p`が、sampleしたeventから計算したrequestごとの累積位相（0またはπ）と一致すること。
   - order-2 eventを含むtoy testも追加する。
2. **証拠の取り方**（どちらかを選ぶ）
   - **A（推奨、再計算なし）**：保存したrun02の回路（X/Yのwrapper qpy）に修正後の判定を適用する。PASSなら、保存したrun02 trialから`price_bindings`で価格record・sidecarを作り、Cへ引き渡す。保存結果の再解析と判定変更にあたるため、承認が必要。
   - **B**：修正したcallerでrun03を実行する（新規compile 2件、約3分）。新しい実行なので承認が必要。
3. P-2、S-2、N/MSE/Gには進まない（mandatory STOP）。

## 7. 保全

- run01・run02のcaller出力とprivate fileは、そのまま保存した。
- 固定SOURCE・input・raw・S-1結果・grant・ledger・marker・KEEP・STOP、D・B・Cのbranch・worktree・processは変更していない。
- 共有環境と既存venvは変更していない。
- STOP後に行ったのは、保存回路の構造診断（§冒頭）だけである。
