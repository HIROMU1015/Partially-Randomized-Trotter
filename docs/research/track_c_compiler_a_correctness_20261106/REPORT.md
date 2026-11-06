# Track C：参照compiler Aの実装と限定correctness

2026-11-06 JST、SERVER 2 `GPUA100-gleap02`。
基準commit：`05d9c5d2eb984e0f3ce89f44a3f2928685b8d4bf`。
専用branch：`codex/track-c-compiler-a-correctness-20261106`。

**微小CRZの消失を解消し、人工44ケースと指定H4 q=1の数値correctnessはPASSした。18round RPE全体はUNQUALIFIED。**
Full Y4の同じ旧q1/j0 SOURCEに対するX期待値誤差は、旧opt2の約`1.41e-5`からAの約`1.45e-12`へ改善した。
全roundのcompiler誤差上界、native成功率、shot数、総G、価格優劣の認定は行っていない。
限定検証を完了してmandatory STOP。旧run01の未完了価格セル、三pilot、C-S/C-P追加実験は再開していない。

## 1. 条件と再利用

Linear H4、隣接1.00 Å、STO-3G、保存DF rank12、生成順LD3とFull LD12、時間幅`h=.02`。
元の物理Hの厳密groundを供給する仮定を維持し、状態準備費は除外する。
H4入力照合PASSを引き継ぎ、SCF・DF fit・ground solve・H4照合は再実行していない。

旧pilot公開commitは`a78054dc48737b06b9396e649f53ecb9efd4bd3b`。
[静的原因調査](../track_c_compiler_static_investigation_20261106/REPORT.md)とその保存結果を再利用した。
入力DFのSHA256は`13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b`、
ground referenceは`6a350e7dd107f242bb00eae6a8d552ab4f4d1fc8df850c39ba8f838174994ae3`。
既存専用runtimeと`common.pkl`を読み取り専用で再利用した。共有パッケージは変更していない。
cacheのSHA256は`68269aa0415f048a0220beb4054daa1bba715c6eb29fbbe9d5ba779baf0346f2`。
既存SOURCE・raw・grant・ledger・marker・STOPを変更していない。

既存`trajectory_parts`、`d_part`、Gaussian/DF gate定義、scalar phase、実adjoint、wrapperを再利用した。
旧`forward_part`は価格compileも呼ぶため利用せず、既存`d_override`に純SOURCE生成`d_part`を渡した。
旧launcherやprice関数は起動していない。LD3では旧seed・stratum0・j0の同一固定trajectoryを再構築し、追加MC sample batchは0件。
これはRTE母集団平均の追加推定ではない。LD12は決定論的である。

4本のX/Y検証を別プロセスで並列実行した。各プロセスのBLASは1 threadとして多重poolを避けた。
worker数は独立タスク数と利用可能CPU affinityから決め、共有環境の設定や他ユーザーのプロセスは変更していない。
実行は開発中の専用worktreeで行ったlocal検証であり、immutable CIや第三者による外部再現ではない。

## 2. 最小実装

[compiler SOURCE](../../../src/trottertracks/track_c_compiler_a.py)の明示pass列は次のとおり。

1. opaqueな`unitary/isometry`を拒否する。
2. CX/X/H/Zの自己逆、S/Sdg・SX/SXdg・T/Tdgという固定gate対だけを`InverseCancellation`で相殺する。
3. `UnrollCustomDefinitions(SessionEquivalenceLibrary, [rz,sx,x,cx])`。
4. opaque gateを再確認する。
5. 同じ既存equivalence libraryによる`BasisTranslator`。
6. CX/Xの固定自己逆相殺。出力basisを検査する。

Qiskit 1.3.0を固定する。coupling mapはなく、確率的passもない。
`CommutativeCancellation`、微小角cutoff、`RemoveIdentityEquivalent`、Euler/KAK最適化、matrix consolidation、preset/HLS/UnitarySynthesisを呼ばない。
連続角の近似一致を使う相殺はしない。今回のgate定義経路では数値小角を保持できたため、symbolic parameterの追加は不要だった。
一般の未知gateや数値行列合成までを保証するcompilerではない。

変更は新compiler・限定validation・runnerと、既存`common()`の入力cache pathを環境変数で指定できる小差分に限る。
変数未指定時の旧読み込み先は維持する。旧worktree内のSOURCEは変更しない。

### phaseの適用条件

**制御を付けたfull joint回路を先に構築してからlowerする。出力へ後から量子controlを追加する用途は未資格。**
systemの`U`と`exp(i gamma) U`はcontrol後に異なる相対位相を生むので、systemだけをprojectiveに比較していない。
一方、既にcontrolを含むjoint回路全体の共通global phaseは物理的に観測されない。

人工composite-adjointの最初のstrict検査では`norm(U-V)=1.414213562373088`となり停止した。
[初期失敗記録][initial]と[pass別debug][debug]を保全した。
BasisTranslatorで全joint空間に共通の位相約`pi/2`が生じており、control0側だけでその位相を決めた全operator差は約`1.18e-15`、実測coherence差は約`3.14e-16`だった。
これは制御されたsystem scalarの欠落ではない。strict scalar equalityはPASSしていない。
入力が書き換わっていないことも確認した。

最終判定は、静的REPORTと同じ「既にcontrolledなjoint回路全体の共通global phaseを許容」という物理契約に合わせた。
control1側に独立のphase fitはしない。compilerに数値phase補正を加えたわけではなく、実際のX/Y期待値も補正せず比較する。
この制限を`PROFILE`とAPI docstringへ明記した。

## 3. 人工correctness

[SYNTHETIC.json][synthetic]に44ケースの全operator比較を保存した。
CRZ・controlled RZZの0、±`1e-15`、±`1e-6`、±`5e-6`、±`1e-5`の直下・一致・直上、±`1e-4`、±2pi、±4piを含む。
さらにcontrol前のsystem scalar、負角、sourceの実adjoint、nativeの実inverse、固定gate相殺と小角の共存を検査した。
opaque matrix拒否も確認した。

| 検査 | 結果 |
|---|---|
| 44ケース | 全PASS。最大control0校正後の全operator差`1.1845512648065984e-15` |
| `CRZ(1e-5)` | RZ2本・CX2本を保持。非ゼロ±半角を構造確認。全operator差`1.1102230246574683e-16` |
| nativeのactual inverse | SOURCE adjointとの物理operator差`8.086698725842868e-16` |
| scalar phase | system scalarをcontrol前に保持し、control0/1の相対位相を検査してPASS |

人工許容値`1e-12`は数値実装診断用であり、全RPE compiler予算ではない。
固定gate以外の一般的な角度融合・最適化を追加検証した結果ではない。

## 4. H4限定correctnessとbaseline

X/Y各軸の**測定を残したfull wrapperをcompile**し、operator評価時にのみ終端測定を外した。
全9 qubitの512×512 operatorを比較した。途中を70次元particle sectorへ射影していない。
併せて保存された元H groundに対する実測X/Yと、ancilla coherenceの位相・半径を評価した。
共通global phaseは既知prep/readoutを外したcontrol0 block全体のtraceだけで校正した。
raw strict差、校正位相、control0 identity、control mixing、unitarity defectも各JSONに保存した。

Partial S2の強baselineは、Rを挟む最大D runとprep/readoutを同じAで個別lowerして結合する既存の強加法契約を使った。
Full LD12の強baselineは既存定義どおりactual full wrapperであり、fullと同じ結果を参照する。独立した別検証数には数えない。

| 条件・variant | X全operator差 | Y全operator差 | X期待値差 | Y期待値差 |
|---|---:|---:|---:|---:|
| Partial S2 LD3 full | 1.6261e-13 | 1.5724e-13 | 1.4322e-13 | 6.6613e-16 |
| Partial S2 LD3 strong | 1.6246e-13 | 1.5934e-13 | 1.4411e-13 | 3.2752e-15 |
| Full Y4 LD12 full=strong | 1.5869e-12 | 1.5748e-12 | 1.4525e-12 | 1.8596e-14 |

全operator差はjoint共通phase校正後のspectral norm。
H4の診断許容値`1e-9`以下で、control0・mixing・unitarityを含む全チェックがPASSした。
これはcompilerへの新たな`1e-9`予算割当てや、rigorous上界の認定ではない。

実測XとYを組み合わせた複素信号について、full側の結果は次のとおり。
軸別JSONのcore coherenceと、この別軸測定ペアを区別して保存した。

| 条件 | SOURCE半径 | native半径 | 位相差 rad | 半径差 |
|---|---:|---:|---:|---:|
| Partial S2 LD3 | 0.9999352770812995 | 0.9999352770811565 | 6.9329e-15 | 1.4300e-13 |
| Full Y4 LD12 | 0.9999999999999911 | 0.9999999999985407 | 8.1490e-14 | 1.4504e-12 |

LD3 strongのペア位相差は`3.0327e-15`、半径差は`1.4411e-13`。
旧S2 X誤差`3.6637e-14`に比べAの丸め診断はやや大きいが、旧Y4で問題だった約`1.4077e-5`の差はAで見られない。
これにより旧Y4の差が交換判定だけで全量説明された、とは主張しない。

### 負時間段階の独立ターゲット

Y4の負時間`-0.03404828767838631`におけるD halfとscalarを、保存済みの独立Hamiltonian作用targetへ照合した。
新たなeigensolveや時間発展target生成はない。
[NEGATIVE_HALF.json][negative]：SOURCEとの差`5.2371e-15`、A nativeとの差`1.2754e-13`、全joint operator差`2.6806e-13`でPASSした。
targetファイルは固定基準commit内の旧`signed_control_debug_arrays.npz`を読み取った。

## 5. 18round RPEへの接続と未資格の理由

目標`epsilon_E=1.5936e-4 Ha`、全失敗確率`.05`、全位相幅`.4 rad`、`delta0=.02`、18roundを変更していない。
q1の小さい差や許容値PASSを、18roundの成功判定には用いない。

各実現x・round mの両軸に有効なjoint operator差の**上界**を`d_mx`とする。
同じsource重み・normalizationで、旧契約と同じ保守形

`delta_comp,m <= 2 sqrt(2) E[abs(W_mx) d_mx]`

を使う。corrected signalの半径rhoを使う場合、重みのrange Lもcorrected側にそろえ、
`delta_comp <= 2 sqrt(2) L max_x d_mx`とする。raw側へ移るときは信号とdiskを共に同じnormalizerで割る。
raw半径とcorrected diskを混ぜず、normalizationを二重適用しない。
`delta_comp < rho`のときだけ`beta_comp=asin(delta_comp/rho)`と半径下限の減少を会計できる。
既存PF/RTE/input/numerical diskとの重複を除き、全位相`.4`からcompiler分を差し引いた正の統計予算が必要である。

既存`shot_rows`へは資格を満たす新しい`radius_lower`と`beta_stat`を渡す。
range Lの各軸で`eps=rho_lower sin(beta_stat)/sqrt(2)`、
`N >= 2 L^2 log(2/alpha_axis)/eps^2`、36軸の`sum alpha_axis <= .05`を満たし、
branch/aliasとエネルギー精度条件も維持する必要がある。
新compilerでNや成功率を再計算しておらず、旧SOURCE meanのfeasible判定はnative資格を与えない。

同じ固定kernelを繰り返す場合に限り`||U^q-V^q|| <= q ||U-V||`が使える。
Full Y4の観測q1差に`2^17`を掛けると約`2.0800e-7`だが、これは仮定付き診断値であり上界ではない。
一体compile、round別r、全RTE supportへこの値を外挿しない。
全gate定義/角度・scalarの丸めと変換誤差、全supportの長さ/重み、実際の反復・隣接境界をカバーする上界がまだない。

q1の数値差から作ったdisk診断はS2約`4.60e-13`、Y4約`4.49e-12`で、正半径と整合する。
これもfloat64で評価した差であり、outward-roundedな保証上界ではない。
**all-round compiler upper boundはnull、native RPE資格はUNQUALIFIEDとした。**
C-Sの大きいrangeが丸めを増幅する問題も今回のPASSでは解決しない。C-S/C-Pの追加計算は行っていない。

## 6. 価格評価へ進む条件とSTOP

今回の新価格セル、T合成、18round native compile、追加sampling batchはすべて0件。
旧Q 63/306・T 24/108セルは旧profileの履歴のまま保全し、残りを0補完しない。
Aは別profileなので、旧価格と混ぜて同条件の総Gを作らない。価格増減・候補の優劣は未測定。
RZ/CX/T/depthの変換やT見積もりも追加していない。

次の担当と判断事項：

1. **ユーザー**：既にcontrolledなjoint回路としてAを採用する条件、残る全round誤差資格を閉じる範囲、次の限定検証/価格cohortの実行範囲を判断・明示許可する。
2. **Codex**：その許可後、定義・丸めの構造上界または明示された限定資格を整備し、同じ精度・成功条件を満たす候補と強baselineに同じprofile・最適化機会を与える。価格取得は許可された新領域で行い、旧run01は再開しない。

今回はこの限定correctnessと公開照合でmandatory STOP。価格・三pilot再開へ自動で進まない。

## 7. 成果物と再現性

- [SUMMARY.json][summary]：実測X/Yペア、旧値との比較、資格状態。
- [入力と継承provenance][inputs]、[最終SOURCE指紋][source]、[環境][environment]。
- [人工結果][synthetic]、[H4軸別JSON][h4]、[負時間結果][negative]。
- [lossless archive一覧][archives]：各512×512 SOURCE/native/strong operatorをそのままnpy化し、展開後のbyte hashを確認。元npzもローカル保全。
- [native QASM archive][qasm]：6回路。OpenQASM2は共通global phaseを保持しないので、strict operator履歴は保存matrixを参照する。出力へ新しいcontrolを付けない。
- [STOP][stop]、[実行log][log]、[初期失敗][initial]、[pass debug][debug]を含め負材料も保存した。

入口は[runner](../../../scripts/run_track_c_compiler_a_correctness.py)、人工testはvalidation module内の`synthetic_checks()`。
入力cacheは旧専用環境のPython objectを含むため公開archiveへ新たに含めていない。
同一cache/旧runtimeがある環境では、別途許可された再現実行を**新しい**outputで行う入口は次である（今回のSTOPを再開しない）。

```bash
python scripts/run_track_c_compiler_a_correctness.py --common-cache /path/to/common.pkl --out /path/to/new-output
```

H4 SOURCE自体を新規準備する外部再現やclean checkoutでの再生成は未検証。
開発中、人工検査のphase契約修正後に有限suiteを再実行した。H4科学4ケースはその最終suiteで各1回、負halfは別の限定呼出しで1回実施し、最後に保存JSONを集約した。
final runnerはこの同じ範囲を一括呼出しできるが、完成後のend-to-end再実行はしていない。
公開commitへSOURCE/結果を固定することを、実行時のimmutable CIと混同しない。

[summary]: ../../../artifacts/track_c_compiler_a_correctness/2026-11-06/run01/SUMMARY.json
[inputs]: ../../../artifacts/track_c_compiler_a_correctness/2026-11-06/run01/INPUTS.json
[source]: ../../../artifacts/track_c_compiler_a_correctness/2026-11-06/run01/SOURCE.json
[environment]: ../../../artifacts/track_c_compiler_a_correctness/2026-11-06/run01/ENVIRONMENT.json
[synthetic]: ../../../artifacts/track_c_compiler_a_correctness/2026-11-06/run01/SYNTHETIC.json
[initial]: ../../../artifacts/track_c_compiler_a_correctness/2026-11-06/run01/SYNTHETIC_initial_failure.json
[debug]: ../../../artifacts/track_c_compiler_a_correctness/2026-11-06/run01/ADJOINT_DEBUG.json
[negative]: ../../../artifacts/track_c_compiler_a_correctness/2026-11-06/run01/NEGATIVE_HALF.json
[h4]: ../../../artifacts/track_c_compiler_a_correctness/2026-11-06/run01/h4
[archives]: ../../../artifacts/track_c_compiler_a_correctness/2026-11-06/run01/ARCHIVES.json
[qasm]: ../../../artifacts/track_c_compiler_a_correctness/2026-11-06/run01/native_qasm.tar.xz
[stop]: ../../../artifacts/track_c_compiler_a_correctness/2026-11-06/run01/STOP.json
[log]: ../../../artifacts/track_c_compiler_a_correctness/2026-11-06/run01/CORRECTNESS_RUN.log
