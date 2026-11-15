# 決定論PF：未観測60軸のcompiled RZを保存証拠と実装でどこまで決められるか

**範囲**
- 対象：Full PF・B0I p7 × S2（16h/7）・標準Yoshida4（16h/3）・M5（16h）の6候補。
- 未観測round：m7・m8・m10・m11・m12のX/Y（60軸、物理点30）。
- 今回したこと：静的な技術調査と計画作成。
- 今回していないこと：新規の回路build・compile・transpile、保存回路のRZ再計数、価格予測・fit・N・G。
- 今回の新しい検査は、metadataだけを使うschedule同一性の検査1本（下記§2）。
- 既存計画 [NEXT_VALIDATION_PLAN.md](../d_pf_allround_price_coverage_plan_20261115/NEXT_VALIDATION_PLAN.md)（`bcb67b2b`）からの差分は§7にまとめた。

## 保存証拠で説明できた範囲

1. **未観測roundの入力回路は、観測済みroundと同じpiece列の繰り返しである**
   - 6候補それぞれで、m4〜m12の全signed pieceの時間を、binary64のbit単位で比べた。
   - roundの中でも、round間でも一致した。540 check、すべてPASS（[SCHEDULE_IDENTITY.json](SCHEDULE_IDENTITY.json)）。
   - roundによって変わるのは、step数qと、総時間Tで一回だけ掛けるscalar位相 `p(−scalar·T)` だけである。
2. **compilerは局所的な書換えしかしない**
   - 固定条件：Qiskit 1.3.0、opt1、coupling mapなし。
   - 使われるpassは次の4つだけである。
     - init段のInverseCancellation
     - BasisTranslator（gateごとの置換）
     - 1q runの再合成
     - 隣接する逆元対の相殺
   - layout・routing・可換性による相殺・block統合は使われない。
3. **書換えの影響は1つのpieceを越えない**
   - one-body blockの方向制御は、各signed pieceで全system wireに `CX(anc,k)·RZ·CX(anc,k)` を置く。
   - 制御を担うancillaにも、各方向制御blockにCXがある。
   - これらは局所passでは消えない障壁になる。そのため、書換えはpieceの中で閉じる。
   - よって、固定幅のcompiled RZ数は `A + B·q + δ(q)` の形になる。
   - δは最後のancilla 1q runからだけ生じる。
   - Qiskit 1.3.0のZSX合成の規則では、δは各軸で0または−1である。
   - δ=−1になるのは、T依存の角度がちょうど1e−12の幅に入るときだけである。
4. **保存済みの一致は、上の構造と矛盾しない**
   - m4/m5の較正（M5のq=1を含む）が厳密に一致した。
   - m6/m9の凍結予測が完全一致した（12/12、2回とも。**保存済みの観測事実として扱う**）。
   - 12モデルの切片が共通の307である。
   - X=Yである。m9では6指標すべてが一致した。
   - いずれも、境界の効果が先頭と末尾のpieceに閉じていて、観測点ではδ=0であることと整合する。
   - ただし、一致した事実からm7・m8を補間したり、m10〜m12に2%条件を適用したりはしていない。
5. **テンプレートを再利用できることと、compiled価格が加法的であることは別である**
   - p7のm9で使ったテンプレート再利用が保証するのは、compile前の入力が同一であることまでである。
   - Fullでは、凍結した傾きが次の関係になる。piece当たりのcompiled RZ数がsigned timeに依存することを示している。
     - Y4 = 3·S2 + 162
     - M5 = 11·S2 − 48
     - これに対しp7では、Y4は3·S2に、M5は11·S2に厳密に一致する。
   - したがって、Fullではpiece価格を公式間・幅間で移せない。
   - ただし固定幅のq依存は、各stepの角度がbit単位で同一なので（1.）、この影響を受けない。

## 残る不明点

| # | 不明点 | 判断を妨げているもの | 価格への影響 |
|---|---|---|---|
| N1 | m7・m8・m10〜m12での最後のancilla runの角度余裕 | 角度 `γ = 2·gp₀ − scalar·T（Y軸ではさらに−π/2）` の実際の値を見ていない。gp₀はone-body blockの方向制御で生じる位相 | 各軸で0または−1 RZ（保守的に見ても3以下） |
| N2 | 障壁の前提 | one-body方向制御の回転角 `|eps_k·τ/2|` が全wire・全piece種別で1e−12より十分大きいかを数値で確かめていない。compiled回路が実際に周期構造をもつことも、保存回路では未確認（総数の一致しか見ていない） | 前提が崩れると、閉包の式自体が成り立たない |
| N3 | Fullの角度依存の発生箇所 | 保存回路を見ていない。仮説は、Fullだけにあるfragment 7〜9で、角度が1e−12の判定幅をまたぐこと。向きは観測と合う（|τ|が大きいY4では+、小さいpieceが多いM5では−） | 固定幅のq閉包には影響しない。幅や公式をまたぐ移送はできない |
| N4 | 価格の採否と重みの情報 | Cが閉包から導いた価格をGに使えるか（その採否基準）、Bの各roundのN | 次の作業の選び方が変わる（§6.6） |

- 60軸の具体的なRZ値は今回計算していない。
- m10〜m12で2%が成り立つとは主張しない。
- 凍結モデルの資格（`qualified_for_candidate=false`、`long_q_extrapolation_qualified=false`）は変更していない。

## 推奨する次の1作業

**保存compiled QPYの閉包監査（新規compileは0）**

| 項目 | 内容 |
|---|---|
| 対象 | Full・p7 × S2・Y4・M5 × X/Y |
| round | m5・m6・m9（追加でS2・Y4のm4） |
| q | S2 14/28/224、Y4 6/12/96、M5 2/4/32 |
| 実幅 | 0.2285714285714286、0.5333333333333333、1.6 |
| 読み込むQPY | 最大56 |
| compile | 0 |

**目的**
- 直接価格の取得ではない。凍結モデルの値を検証するものでもない。
- 閉包の前提（N1・N2）を、保存済みの実回路で確かめる。
- 観測前に判定規則を固定する。照合に使う予測値は、凍結係数のa・bそのものである。

**結果に応じた扱い**
- 全group×軸がPASS：60軸すべてで閉包の前提が観測点で確認できる。m12のcompileは不要になる。
- FAILまたはUNKNOWN：そのgroup×軸、またはそのroundだけを直接取得の候補にする。

詳細は§6。

---

## 1. 固定資料と再利用

| 資料 | commit | 使ったもの |
|---|---|---|
| 既存計画 | `bcb67b2b2bb78776db1dd8d0625a3facb37cd298` | 優先順位、m12の条件付き案、未検証の範囲 |
| alignment | `0b4b2d941cd669aaa45231e1236bc2d3158a9d12` | 60軸・30物理点のID、T、q、実幅 |
| 凍結12モデル | `12372fa6b53c878dc5dd124e036572ba0cd82061` | 係数a/b。X/Yが同一であること |
| m6 holdout | `a55813373b84b9128cbd10c6812c0d26143a4280` | 12/12の完全一致、実行SOURCE、資源 |
| m9 holdout | `0a35101914c6833244b6e0bad7bc2767a49b5a47` | 12/12の完全一致、cellごとのwall・RSS、テンプレート再利用の回数 |
| m1/m2直接価格 | `6e110c083ec2db2530cdda3f7b5c84d2a7d3f31c` | coverage 96/156、作業branchのHEAD |
| 既存の人工検査 | `51d324ea9c83bc12f1e866da0fa409cef926f74a`（`pr_resource_repo_wide_audit_20261114`） | X/Yのcompiled回路の違いは最後のancilla runだけ（人工回路2000個、[script](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/51d324ea9c83bc12f1e866da0fa409cef926f74a/docs/research/pr_resource_repo_wide_audit_20261114/checks/check_xy_wrapper_locality_artificial.py)） |

**静的に読んだ実装**
- `scripts/h46_long_time_native.py:38-67`（compose）
- `src/trotterlib/h46_m5_plan.py:47-55`（schedule）
- `src/trottertracks/resource_applicability/ax2a_control_plan.py:33-66`
- `src/trottertracks/resource_applicability/ax2b_native_df_v5.py:94-166`、`:259-283`
- `scripts/h46_new2_native.py:14-56`（Givens基底：XXPlusYY＋Phase）
- `src/trotterlib/df_trotter/ops.py:226-321`
- `scripts/pf_native_price_fast.py`

**Qiskit**
- 実行に使った venv（Python 3.12）と h6 venv（Python 3.11）で、Qiskit 1.3.0のtranspiler・synthesis・standard gatesのPython sourceは同一だった（`diff -r`、`__pycache__`を除く）。
- Rust部分は、upstreamのtag 1.3.0から次の3ファイルを読んだ。
  - `crates/accelerate/src/euler_one_qubit_decomposer.rs`（SHA-256 `15d9f65e…`）
  - `inverse_cancellation.rs`（`b1867ec8…`）
  - `basis/basis_translator/mod.rs`（`d2b287ce…`）
- installされたbinaryとbit単位で同一であることは確認していない。

## 2. 繰り返されるpiece列と隣接する命令

**schedule同一性の検査**（[check_schedule_identity.py](check_schedule_identity.py)、540 check、すべてPASS）
- `signed_time = T·w/q`（`h46_m5_plan.py:54`）を、alignmentの各roundのT・qで評価した。
- 確認した内容：
  - stepごとのpiece列が周期的である。
  - round間でpiece列がbit単位で一致する。
  - Tが正確に2のべき乗倍、qが正確に倍になっている。
  - 実幅のhexが一致する。
  - control planの半分の時間が `s/2` と一致する。
- 2のべき乗によるscalingと、正しく丸められる除算により、`T·w/q` はroundによらず同じ浮動小数点数になる。
- 保存実績とも合う：p7のm9のテンプレート再利用で、実際にbuildされたpieceは S2 224/1、Y4 288/2、M5 352/6 だった（呼出し数/実build数）。

| 公式 | step当たりのpiece数 | 異なるsigned timeの数 | signed time（binary64） | 負のpiece | step境界の文脈 |
|---|---:|---:|---|---:|---|
| S2 | 1 | 1 | 0.2285714285714286 | 0 | τ→τ |
| Y4 | 3 | 2 | 0.7206438357118174、−0.9079543380903016 | 1/step | 0.7206→0.7206 |
| M5 | 11 | 6 | 0.1796…、0.1939…、0.2008…、0.2205…、0.2269…（左右対称）、中央 −0.4434 | 1/step | 0.2008→0.2008 |

**1つのsigned pieceの構造**（`deterministic_control_plan`、`2nd`、q=1）
- 前半：`U_0 … U_p` 。各blockは `basis⁻¹_k · Diag_k(τ/2) · basis_k` で、制御なし。
- 後半：`D_p … D_0`。各blockは同じ形で、方向制御付き。
- p+1はblock数で、Fullは11（one-body＋fragment 0..9）、p7は8（one-body＋fragment 0..6）。

**命令の隣接関係**

| 場所 | 隣接する命令 | 時間に依存するか |
|---|---|---|
| block間（前半・後半） | `basis_k → basis⁻¹_{k±1}`（XXPlusYY・Phase） | 依存しない |
| piece中央 | `Diag_u,p(τ) · basis_p · basis⁻¹_p · Diag_dir,p(τ)`（互いに逆のGivens列） | 角度だけがτに依存 |
| piece境界＝step境界 | `Diag_dir,0(τ_j) · basis_0 · basis⁻¹_0 · Diag_u,0(τ_{j+1})` | 角度だけ。固定幅では全境界で同一 |
| ancillaの内部 | 各方向制御blockの後の `rz(2·gp_k)`（対角のみ） | piece時間だけ |
| 先頭 | `h(anc)`。system wireは最初のpieceの `basis⁻¹_0` から始まる | 依存しない |
| 末尾（system） | 最後のpieceの `basis_0` で終わる。後ろに測定はない | 依存しない |
| 末尾（ancilla） | `rz(2·gp₀(τ_last)) · p(−scalar·T) · [sdg] · h · measure` | **T依存は、roundによって変わる唯一の角度** |

**補足**
- build時に枝刈りされるのは、角度がちょうど0のときだけである（`ops.py:236/285`、`ax2b:97`）。乱数はない。
- 前半の制御なしblockはancillaに作用しない。
- scalarは、pieceの中では0として扱い、composeの最後に一回だけ掛ける（`h46_long_time_native.py:64-65`）。
- p7のscalarは `c_H＋tail identity` である。この経路はFullと同じで、値だけが違う。

## 3. compilerの経路と局所性（Qiskit 1.3.0、opt1、basis rz/sx/x/cx、seed 17、coupling mapなし）

| 段 | 内容（`level1.py`、`builtin_plugins.py`） | 局所性 |
|---|---|---|
| init | InverseCancellation（CX、H、X、S/Sdg、SX/SXdgなど）。3q gateの展開は、coupling mapがないので行われない | 同じgateが隣接する組だけ |
| layout・routing | なし（`coupling_map or initial_layout` が偽）。seedは結果に関与しない | — |
| translation | UnitarySynthesis・HLS（対象gateなし）、BasisTranslator | `basis_search` はgateの種類の集合だけで決まり、置換はgateごと |
| pre-opt | 空（`GateDirection` は coupling mapがあるときだけ） | — |
| optimization | DoWhile { Optimize1qGatesDecomposition、InverseCancellation、GatesInBasis/unroll、Depth・Size の FixedPoint } | 下記 |

- **Optimize1qGatesDecomposition**
  - `collect_1q_runs` で集めたrunを、1つずつ独立に、行列からZSXへ再合成する。
  - 置換するのは次のいずれかの場合だけである：basis外のgateを含む、gate数が厳密に減る、新しい誤差が0になる（Rust `euler_one_qubit_decomposer.rs:1076-1244`）。
  - `ANGLE_ZERO_EPSILON = 1e−12` である。
  - θ≈π/2のrunは `rz(λ−π/2)·sx·rz(φ+π/2)` になり、各RZは `|mod 2π| > 1e−12` のときだけ出力される。
- **InverseCancellation**
  - `collect_runs` で集めた連続する同一gateのうち、qargsが同じ部分列を偶奇に応じて除去する（`inverse_cancellation.rs:42-106`）。
- **可換性による相殺、ConsolidateBlocks、RemoveIdentityEquivalent、RemoveDiagonalGatesBeforeMeasure は opt2/3 だけで、今回の経路には含まれない。**

## 4. 閉包の論証と、その条件

**条件**

| 条件 | 内容 | 状態 |
|---|---|---|
| P1 | piece列とその引数が、stepごと・roundごとにbit単位で同一 | §2で確認済み（metadata、540 check） |
| P2 | 経路上のpassがすべて局所的（§3） | sourceで確認済み（Pythonはinstalled、Rustはtag 1.3.0） |
| P3 | 障壁：各pieceで、全system wireに `CX(anc,k)·RZ(±eps_k·τ/2)·CX(anc,k)` があり、ancillaにも各方向制御blockのCXがある | 構造は確認済み。数値（RZ角が1e−12の幅の外）は未確認（N2） |
| P4 | 末尾の効果は最後のancilla runだけ | 構造は確認済み。X/Yについては51d324eaの人工検査と整合。角度の余裕は未確認（N1） |

**論証**
- P3が成り立てば、RZは2つのCXに挟まれて恒等にならず、CXの組も隣接しない。
- したがって、1q runも相殺の連鎖も、pieceを越えて伝わらない。
- P1が成り立てば、内部のstepと境界の文脈はすべて同一である。
- 先頭の効果は最初のpieceに、末尾の効果は最後のpieceに閉じる。
- step数×piece数が2以上なら、`C(q) = A + B·q + δ(q)` となる。
  - 全roundでこの条件を満たす（M5のq=1でも11 pieceある）。
  - δは最後のancilla runのRZ数の差である。
  - runの形は対角×Hで、θ=π/2の枝に入るので、通常はRZ 2個になる。
  - `γ+π/2 ≡ 0（mod 2π、1e−12以内）` のときだけ1個になる。

**保存一致との対応**

| 観測 | 実装からの説明 |
|---|---|
| m4/m5を通る直線が厳密に引ける（M5はq=1,2）。m6・m9の予測が12/12で完全一致 | `A+B·q` の構造、かつ観測点でδ=0 |
| 12モデルの切片が共通の307 | 先頭・末尾に関わるのはone-body blockとancillaだけで、routeや公式によらない構造 |
| X=Y（凍結係数、m9の6指標） | X/Yの違いは最後のancilla runの `sdg` だけ。対角×Hの形ではRZ数が変わらない |

これは実装レベルの論証であり、証明でも観測でもない。

**残る前提**
- 実際のcompiled回路が周期的であること（N2）。
- 未観測roundでのδ（N1）。
- installされたRust binaryがtagのsourceと同じ挙動をすること。

## 5. 結論をどこまで移せるか（Full↔p7、公式、X/Y）

| 移す先 | 共通する経路 | 結論に影響し得る差 | 判断 |
|---|---|---|---|
| Full ↔ p7 | compose、方向制御、wrapper、compiler、scalarを一回掛けること、one-body blockが先頭・末尾にあること | block数（11/8）、Full専用のfragment 7〜9、scalarの値 | **q閉包の論証は両方に適用できる。** Fullだけ、piece価格が角度に依存する（N3） |
| S2・Y4・M5 | `2nd` のpieceの内部構造とstep境界の型が同じ | signed timeの種類（1/2/6）、負のpiece、公式内部の境界 | 論証は共通。piece価格は移せない（Fullでは +162/−48） |
| X ↔ Y | 最後のancilla runより前は同一 | `sdg` による角度のずれ（−π/2） | 総数は保存点ですべて一致。未観測roundでは軸ごとにN1が残る |
| 幅の違い（m0〜m3） | 同じ経路 | 角度が違う。Fullは角度に依存する | 移さない。m0〜m3はすでに直接価格がある |

## 6. 推奨する有限案：保存compiled QPYの閉包監査（compile 0）

### 6.1 対象

**compiled QPY**
- 置き場所（private、読取専用）：
  - `/home/AbeHiromu/projects/d-full-pf-shortlist-20261114-private/{calibration,holdout,m9_holdout}/run01/compiled_cache/`
  - p7は同じ構成の `d-truncated-p7-t3-20261114-private`
- 公開済みRESULTのwrapper fingerprintで、各QPYを1対1に束縛する。

| group（X/Y各1） | m5のq | m6のq | m9のq | 追加のm4 | 実幅 |
|---|---:|---:|---:|---:|---|
| Full・p7 S2 | 14 | 28 | 224 | 7 | 0.2285714285714286 |
| Full・p7 Y4 | 6 | 12 | 96 | 3 | 0.5333333333333333 |
| Full・p7 M5 | 2 | 4 | 32 | —（較正では保存値を再利用。QPYなし） | 1.6 |

**読み込む件数**
- compiled QPY：主36件＋追加8件。
- m9のwrapper QPY：12件。最後のancilla runの入力角を確かめるためだけに使う。
- 合計56件以下。
- 新規のcompile・transpile・build・価格予測・fitは0件。

### 6.2 観測前に固定する判定（すべて厳密一致。閾値は事前に固定）

- **A1 束縛**
  - QPY・cache JSONと公開済みreceiptの照合：fingerprint、6指標。
  - 一致しないものは、その軸をUNKNOWNにする。再compileはしない。
- **A2 周期閉包**
  - 対象：各group×軸のwireごとの命令列。tokenは（gate名、qubit番号、paramのhex）。最後のancilla 1q runは除く。
  - 条件：全監査qで `H_w + B_w×(q−c) + E_w` と書ける。整数cと、先頭H_w・本体B_w・末尾E_wは共通。
  - 判定はPASSかFAIL。
  - 51d324eaの `wires_without_final_ancilla_run` の方式を流用する。
- **A3 凍結係数との整合**
  - 確認する式：
    - `Σ_w RZ(B_w)` が凍結した傾きbに一致する。
    - 「本体以外のRZ＋最後のrunのRZ」が `a + c·b` に一致する（a=307）。
  - 予測値は凍結JSONの係数だけを使い、新しく算出しない。
- **A4 最後のrun**
  - 形を記録する：`rz·sx·rz` 形で、SXが1個。
  - 監査点で、T依存の角度とTとの線形関係（傾きは −scalar）を確かめる。誤差は1e−9以内。
  - 各未観測round m∈{7,8,10,11,12}・軸について、`|mod 2π(λ_m − π/2)|` を求める。
    - 値が10⁻⁶以上なら、RZ数は不変とする。
    - 値が10⁻⁶未満なら、その（group,軸,m）をUNKNOWNとし、直接compileの候補にする。
- **A5 障壁**
  - 各wireの `B_w` にCXが1個以上あることを記録する（A2から分かる）。

### 6.3 判断できること・残ること

**全group×軸でA1〜A5がPASSした場合**
- 判断できること：
  - 60軸すべてで、`C(q)` が凍結した直線と同じ構造であることが、保存回路で直接確認できる（q≤224）。
  - 未観測roundのδが0であることが、数値で確認できる。
- 残ること：
  - q>224への延長は、§3のpassの局所性（実装の設計）に依る。
  - 価格の値を算出するのも採否を決めるのも、Cの別作業である。

**FAILまたはUNKNOWNがある場合**
- そのgroup×軸の未観測roundは未確定のまま残す。
- 最小の直接取得は、該当するgroup・軸・roundだけにする。
- groupの構造全体がFAILなら、そのgroupのm12 X/Yを終端の直接点とする代替案（§6.6）。

### 6.4 §10.5による重複照合（今回の確認範囲）

| 照合先 | 結果 |
|---|---|
| D branch HEADの価格dir（`d_full_pf_shortlist_prices_20261114`・`d_truncated_p7_t3_prices_20261114` の全PRICES/RESULT/AXIS_PRICES） | 30物理IDのいずれも価格0件 |
| Track Aのexploration_costs表（`155529e2`）、COVERAGE_V2 catalog（`e7dca5f7`） | 一致0件 |
| HEADで一致したfile | `full_pf_latest_price_guard` COVERAGE（`OUTSIDE_CHECKED_CATALOG`、価格なし）、T2のm12信号check（価格ではない）、計画のmetadata |
| privateのstage root | m7・m8・m10〜m12のstageは存在しない |
| 実行中のjob | なし |
| 保存回路の周期性を調べた既存の作業 | 確認範囲では0件。関連はするが代替にならないもの：51d324eaの人工X/Y検査、m9 ACCELERATIONの人工4-mode検査、P0（`442cafa1`、別の回路） |

- 再利用するもの：公開fingerprint、凍結係数、今回のschedule検査、51d324eaのwire比較の方式。
- 一覧にないarchiveや他のserverまでは網羅していない。

### 6.5 負荷の見通しとSTOP

**負荷**
- QPYの読込は未計測。
- 最大のfileはFull M5 m9のcompiled回路で、5,752,332命令。
- m9のcompiled 12件の合計は48,256,656命令。m5/m6はm9の1/8以下なので、全体で約5.6×10⁷命令である。
- RSSの参考上限は、m9のtrial（compileを含む）のcellごとのpeak 6.29 GiB。
- 進め方：最小のfile（p7 S2 m5）で時間・RSSを測り、命令数に比例するとして見積もり直してから、RAMの空きを観測して並列化する。
- fileごとに開始・終了・RSS・終了codeを段階logに残す。faulthandlerの定期dumpは使わない。

**STOPの条件**

| 事象 | 対応 |
|---|---|
| Qiskit・QPYの版が合わない | 全体STOP |
| privateへの書込みが必要になる | 禁止（読取専用で開く） |
| 束縛できない、または欠損 | その軸をUNKNOWNにする |
| tokenのRZ総数が保存rz_countと一致しない | 監査toolの不具合としてSTOP |
| OOMなどの異常 | 部分logを保全してSTOP |

- FAILでも、再fit・閾値変更・compileの追加は自動では行わない。
- 公開後にmandatory STOP。

### 6.6 比べた他の案と、欠けている情報による変化

**m12終端の直接取得（既存計画の条件付き案。12 compile）**
- 見積りの根拠：m9のcellごとの実績を8倍した線形外挿。同じ経路のm6→m9では、piece数8倍でstageのwallが7.98倍だった。

| cell | Full | p7（高速経路） |
|---|---|---|
| S2 | 約6.6 h | 約2.1 h |
| Y4 | 約8.5 h | 約2.7 h |
| M5 | 約11.8 h | 約3.4 h |

  - Fullにはm9でテンプレート再利用を使っていない。使えば短縮する見込みだが、実対象では未検証。
  - CPUは約35 CPU-h。cellごとのRSSはm9の5.6〜8倍（Full M5で約35〜50 GiB）。
  - 分かるのはm12の12軸だけで、m7・m8・m10・m11は閉包に依存したまま残る。N1もm12でしか分からない。
- **p7 S2だけをm12で取得**：約2.1 hで最小。ただし1 groupしか分からない。
- **m7・m8で取得**：観測済みのq範囲の内側なので、得られる情報が少ない。

**欠けている情報と、それによる変化**

| 欠けている情報 | 選択がどう変わるか |
|---|---|
| Cの採否基準 | 閉包から導いた価格をGに使わず、終端は直接観測を必須とする場合は、監査の結果によらず、BのNで重みが大きいgroupのm12直接取得が次になる。監査はその補強になる |
| BのN | m12の重みが小さいことが分かれば、監査だけで十分である可能性が高まる |
| QPYの束縛ができないgroup | そのgroupだけm12の直接取得に切り替える |

## 7. 既存計画（`bcb67b2b`）からの差分

| 項目 | 既存計画 | 今回 |
|---|---|---|
| 優先度0〜3（m3束縛、m9、m0、m1/m2） | 提案 | 完了済み（96/156）。変更なし |
| 反復block・境界のcompiler閉包 | 「重いcompileの前に技術調査」 | 静的には成立の条件を特定した（P1〜P4、δ∈{0,−1}）。P1・P2は確認済み、P3・P4の数値は未確認 |
| 次の作業 | m12終端（条件付き）。最大12 compile | 0-compileの保存QPY監査を先に行う。m12は、監査のFAIL/UNKNOWNか、Cの採否基準で必要になったときだけ。対象は該当するgroupに限る |
| 未観測roundの誤差 | UNKNOWN。2%を外挿しない | 2%の外挿はしないまま。誤差源を最後のancilla runに特定（N1、各軸≤1 RZ）。ただし前提N2付き |
| Fullとp7の扱い | 同じ規則 | 同じ閉包の論証。Fullはpiece価格の角度依存があるので、幅・公式間の移送を禁止 |
| 資源の根拠 | m6だけ | m9のcellごとの実績、m6→m9の倍率7.98 |

## 8. 実施しなかったことと保全

**実施しなかったこと**
- 新規の回路build・compile・transpile、保存QPYの読込み・再計数、未観測60軸の価格予測、fit、N、G、winner。
- 新規の科学計算は0。

**保全**
- 元のSOURCE、raw、凍結JSON、旧STOP、privateのcache・QPYは変更していない。
- privateについては、ファイル名と容量の一覧だけを取得した。
- 今回の検査は `check_schedule_identity.py` だけである。metadataの算術で、qiskitはimportされるが回路は作らない。

**作業branchと引渡し**
- 作業branchは `codex/d-price-records-c-bridge-20261114`（継続）。
- [HANDOFF](HANDOFF.md)は、Cと統合担当向けである。ロードマップ本体は編集していない。
- 推奨案の実行は、別途の指示を待つ。**mandatory STOP。**
