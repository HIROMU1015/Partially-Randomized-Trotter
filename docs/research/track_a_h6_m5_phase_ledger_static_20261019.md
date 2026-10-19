# H6 full m5_best：保存gate列・位相帳簿の静的調査

2026-10-19。**静的調査は完了。compile時の位相処理経路を特定したが、今回のraw差の原因は未確定。** 新しい回路作用・compile・実回路の位相総和は0。旧raw基準・STOP・58受理軸を保全し、local commit後mandatory STOPする。

対象結果commit：`71481e485772c0e8d0af657bacffd846bc570483`。科学SOURCE：`99e8723d402245a9ad923e7dbc0800f47bc575b6`。構造検査tool SOURCE：`a3089e7d7cb36962bd90223418ee14e341008baa`。branch：`track-a-h6-m5-phase-ledger-static-20261019`。

[前回結果](track_a_h6_m5_saved_phase_run01_results_20261019.md)、[所見bundle](../../artifacts/resource_applicability/track_a_h6_m5_phase_ledger_static/2026-10-19/summary.json)、[行根拠](../../artifacts/resource_applicability/track_a_h6_m5_phase_ledger_static/2026-10-19/source_trace.json)。

## 1. 今回確認した範囲

対象はH6直鎖1 Å/STO-3G/tol-only1e-8 DF19、full m5_best/p19/q2、T=.8/P=0、symmetric-directional/full Hadamard cosineの保存2 QPYのみ。保存科学入力・NPZを復元せず、4再生も繰り返していない。

新しい[構造検査器](../../scripts/inspect_h6_m5_phase_structure.py)はQPYのgate名・配置・parameterのbinary64 bit表現・順序hashを読むだけで、gate matrix・definition・statevectorは評価しない。2 fileを2 processで読み、両processを正常回収した。実角度の和やdeterminant、位相差の新しい実数計算もしていない。

[保存構造metadata](../../artifacts/resource_applicability/track_a_h6_m5_phase_ledger_static/2026-10-19/saved_circuit_metadata.json)は既存schemaと一致した。

|項目|wrapper|compiled|
|---|---:|---:|
|保存operation数（測定を含む）|536,388|1,740,557|
|P|233,465|0|
|XXPlusYY|116,160|0|
|RZ|38,424|831,662|
|CX|120,748|367,008|
|SX / X|0 / 0|533,094 / 8,792|
|保存top-level global phase|1.9714733084376022|1.7007329427166216|

いずれも13 qubit、ancilla12、layout None、最後の測定だけ。wrapperのHはancilla12に2回、ancilla12のPは1回で、最後のHの直前にある。SOURCEでも各signed pieceのscalarは0、外側scalarを一回だけ追加している。**単純なscalar二重挿入を示す構造は見つからなかった。** これは全てのscalar/controlが正しいという新しい数値保証ではない。

異なるqubit上で隣接するPのparameterが、binary64の符号bitだけ反転している箇所を116,160か所確認した。GaussianのXY/P/P構造と整合する。ただし、異なるqubitのPは作用として相殺するgateではない。これを理由にgateを除去したり、位相総和を計算したりしていない。また、隣接一致の件数を、そのまま不重複な取消pair数とは扱わない。

## 2. 位相が伝播する具体的な経路

固定Qiskit 1.3.0の公式release commitは`896d1c8b72888a52a46081a6d1f4de8c56db8aea`。11 sourceを確認し、うち5 Python fileはインストール済み固定版とbyte一致した。Rust6 fileは同releaseの公式ソースとして確認したが、installed binaryをこのcommitから再buildしたというattestationはない。[取得・SHA](../../artifacts/resource_applicability/track_a_h6_m5_phase_ledger_static/2026-10-19/official_source_manifest.json)、[固定版との結合](../../artifacts/resource_applicability/track_a_h6_m5_phase_ledger_static/2026-10-19/release_source_binding.json)。共有runtimeは変更していない。

|処理|静的に確認した内容|今回確定していないこと|
|---|---|---|
|P/U1→RZ|P(θ)=exp(iθ/2)RZ(θ)に対応する位相付き等価則が登録されている|今回どの等価則が何回選ばれたか|
|basis translator|置換回路の位相や、束縛済みparameter由来の位相をDAGへ伝播する|各置換の位相値・丸め寄与|
|opt1|defaultでは1qubit再合成とinverse cancellationを使う|実runのpass履歴・反復数|
|1qubit再合成|runの小行列と角度/位相をbinary64で扱い、置換後の位相をDAGへ加える|ここが今回の差を発生させたか|
|DAG位相加算|浮動小数の加算後、binary64の2πで折り返す。該当関数に補償加算はない|累積誤差の実量・発生pass|

等価則は[公式equivalence library](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/circuit/library/standard_gates/equivalence_library.py#L1216)、basis位相伝播は[公式translator](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/basis/basis_translator/mod.rs#L665)を参照。

1qubit runの行列・角度・位相の生成は[公式Euler実装](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/euler_one_qubit_decomposer.rs#L1174)、位相加算と折り返しは[加算関数](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/circuit/src/dag_circuit.rs#L7012)と[setter](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/circuit/src/dag_circuit.rs#L821)へ追跡した。Rustの`Complex64`は実部・虚部がf64の型であり、NumPy complex128に相当する。NumPy complex64への降格を見つけたという意味ではない。

この経路に数値誤差が入り得ることは分かるが、それだけでcompiler bugや今回の原因を認定しない。実runにはpassごとの回路・個別位相更新値が保存されていない。最終2回路だけから、位相の生成・加算・折り返し・角度簡約の寄与をpass単位に分離することはできない。

## 3. 前回の拡張精度診断との関係

前回の保存値では通常raw差1.504043707445665e-9、拡張raw差1.5039128868467355e-9、拡張common phase1.5026891953807317e-9 rad、aligned残差6.065602469072673e-11だった。今回はこれらを再計算していない。

拡張再生は、**既に保存された**角度とglobal phaseを精度を広げて作用させた。compile時のbinary64再合成やphase加算を再実行したものではない。そのため、拡張再生でraw差が残ったことと、compile時の丸めが原因候補であることは両立する。

候補は、compile時の位相帳簿への誤差、1qubit再合成/角度簡約による非一様な作用差など。既存初期状態で共通位相に見える成分が、全状態で一様なglobal phaseであるとは限らない。top-level global phase2値を直接引くだけでも判定できない。P等のgate本体へ位相が分散しているためである。

## 4. 次の最小案：回路再生をしないdeterminant位相台帳

これは**理論的な設計案で、実回路での実数計算は未実行。runnerも未実装**。新compileを再び行う前に、保存角度のscalar台帳だけで切り分けられる可能性がある。[未承認plan](../../artifacts/resource_applicability/track_a_h6_m5_phase_ledger_static/2026-10-19/determinant_ledger_plan_DRAFT.json)、[無効grant](../../artifacts/resource_applicability/track_a_h6_m5_phase_ledger_static/2026-10-19/grant_INVALID.json)。

n qubit、D=2^nの回路Uについて、k qubit gate gのdeterminantは全系でdet(g)^(D/2^k)となる。局所gateのdeterminantは次のとおり。

|gate|局所determinant|正規化した位相への寄与|
|---|---|---|
|P(θ)|exp(iθ)|θ/2|
|RZ、RZZ、XXPlusYY|1|0|
|H、X|−1|π/2|
|SX|i|π/4|
|CX|−1|π/4|
|回路のglobal phase γ|exp(iDγ)（全系）|γ|

したがって形式上、F=γ+ΣPθ/2+π(N_H+N_X)/2+π(N_SX+N_CX)/4として、det U=exp(iDF)。Fは2π/Dを法としてしか決まらない。

**今回の13 qubitの両gate集合では、H/X/SX/CXの全系determinantはいずれも1。** その定数項はこのmoduloでは消える。ゆえに、必要な実数計算は保存2 phaseとwrapperのP角度だけに縮小できる。

\[
 B=\gamma_{\rm compiled}-\gamma_{\rm wrapper}
      -\frac12\sum_{P\;\rm in\;wrapper}\theta_P,
 \qquad
 \delta_{\det}=\operatorname{wrap}_{2\pi/8192}(B).
\]

branchは事前に[-π/8192,π/8192)へ固定し、保存common phaseへ合うbranchを事後選択しない。各保存binary64値はbitから正確な有理数へ移し、角度総和をbinary64で足し直さない。moduloのπは数学的πを用い、binary64 `math.pi`を高精度値として扱わない。候補の内部精度80/120桁による収束診断は実装前の案であり、正式採択していない。

δ_detと保存common phaseは連続値として並べる。これが近い場合でも、平均的なdeterminant位相との整合を示すだけで、発生pass・全状態での一様位相・作用等価性を認定しない。determinant一致でも異なるunitaryは存在し、2π/Dごとのaliasもある。逆に、異なる場合も自動でbranch変更・位相補正・再fitをしない。

この案には新しい状態・full matrix・statevector再生・sampling・compile・費用会計は不要。実装・人工検証・新SOURCE/入力/runtime sealと新grantを準備し、**別の数値実行承認後に一回だけ**行う。旧4再生の消費済みgrantは流用しない。旧raw基準と受入は一切変更しない。

## 5. 検証・保全・次の担当

人工検証8件PASS（0.388秒）。構造検査5件と、小さい人工gateの厳密記号式3件で、局所determinant・tensor lift・determinant aliasの限界を確認した。実H6パラメータの総和や新しい数値診断には進めていない。作者自身の確認であり、独立レビューではない。subagent0。

[保全](../../artifacts/resource_applicability/track_a_h6_m5_phase_ledger_static/2026-10-19/preservation.json)：2,223 fileが不変、元科学seal1,831 fileも一致。原SOURCE・QPY/NPZ・結果・grant・ledger・marker・STOPを保持し、他worktree・共有runtimeを変更していない。QPY metadata復元2、NPZ復元0、新再生/compile/sampling/実角度総和/有効grant0。

**ユーザー**：再compile不要のscalar位相台帳照合を、次に準備して数値実行する価値があるか判断する。

**Codex（次の指示がある場合）**：正確なscalar蓄積・高精度wrap・改変拒否の実装と固定契約を準備する。実数値計算を始めるのは別承認後のみ。今回の静的調査を不一致の修正完了とは呼ばず、AD03/Coreの保留を維持する。

資料をlocal commit後mandatory STOP。公開push・新科学数値解析・追加compileは行わない。
