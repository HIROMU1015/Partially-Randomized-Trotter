# 研究C：局所blockの相殺損失・phase会計と保持Pauli-frame方式

2026-11-05 JST。Branch `codex/research-c-block-witness-cancellation-readiness-20261105`、基点 `acdb1c404b21c004c16d55a702446b02fe67f2a6`。正式共通指示原本 `research_project_instructions_unified_2026-10-25.md` の§8.1・8.4・10を確認し適用した。

**保存witnessの集計・phase会計に重複は見つからなかった。一方、非centralなPauli補正を境界ごとに大きい誤差として捨てる構造を確認し、その補正を保持する新しいopt-in経路を人工で実証した。** 実H4の新しいArb上界はnull、科学call0。新規・独立6-call契約は未承認であり、実行していない。

参照対象は保存直鎖H4・1.00 Å/STO-3G、number CDF8/元順LD2、partial ordered S2/s1/r32/K0、h=δ=.1、controlled full512、Arb192、既存P0/Qiskit1.3.0/opt1/seed17/rz,sx,x,cx、Gaussian/relative fusion、MPFR160 declared phase、53-bit IID-grid。Endpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、F6:0/F7:6,7の交互32-event word。Own-DF ground oracle・状態準備費用除外、元のCA/RPE政策・数学的targetを変更していない。

## 1. 保存値・構造の照合

旧科学SOURCEは `7f70ca0af877621d6e4abb5db94945819ee0826e`、結果commitは基点acdb1c4。[旧6-call報告](research_c_cx_structural_block_pilot_execution_20261105.md)のraw・契約・STOPは読み取り専用とした。

新規 `c_block_witness_audit` と [監査script](../../scripts/audit_research_c_block_witness.py) は、保存plan・witness・descriptorの有理数集計、旧partitionの構造的replay、phase抽出式の照合だけを行う。新しいgate行列・trig・Arb operator評価・実H4 compileは行わない。
正本は [saved_witness_audit.json](../../artifacts/research_c_block_witness_cancellation/2026-11-05/technical/saved_witness_audit.json)。

| 保存された構成 | 件数 | 保存endpointの重み付き和 |
|---|---:|---:|
| 1q factor | 5,761 | 12.637442504334746 |
| 内部block | 15 | 11.076444046951735 |
| 数値factor合計 | 5,776 | **23.713886551286482** |
| 共通CX（誤差0の厳密factor） | 2,876 | 独立数値項なし |

245 unique bodyのmultiplicityと全factorの対応、175 primitive定義のmultiplicity、primitive項約2.04624e-13、保存global phaseの組立てが全て一致した。共有CXを数値誤差として重ねて数えていない。独立translation legとnative候補同士のtelescopeも区別している。

表示用に保存上界1e-8以上をまとめると、18 unique body（1qが8種類・11 occurrences、内部blockが10種類・10 occurrences）がほぼ全ての大きい項を占める。これは保存endpointの内訳であり、新しいoperator上界・lower boundの取得ではない。個々の保存上界が大きいことから、真の局所operator距離や回路全体の不一致を断定しない。

8 windowの内部だけを調べても不十分だった。大きい1q項はwindow外にもあり、例としてfactor8643/8650はcompiled空列に対してraw `Sdg,RZ(t)`、tの保存値はbinary64の−π/2近傍。factor21/83はraw `Sdg,RZ(t),RZ(t),S` である。tを厳密な−π/2と同一視せず、δ=t+π/2と書けば、前者は厳密に `exp(iπ/4) Z RZ(δ)` となる。

旧SU(2)の中心符号±1だけでは非centralなZを保持できない。各境界でこれを大きい局所差としてtriangleへ入れると、後のZやCX共役で相殺する構造を失う。`CX Z_control CX=Z_control`、`CX Z_target CX=Z_control Z_target` なので、相殺に必要な補正はwire間へも移る。これが新方式で明示的に保持する対象である。実H4のδの新しい数値包囲や、別partitionの数値最適化はしていない。

## 2. Global phase 1.41421の会計

保存式を再構成すると次の通りで、各補償は一度ずつ使われていた。

| 由来 | π係数 |
|---|---:|
| 1q determinant phaseの差 | 546 |
| 内部block determinant phaseの差 | 1 |
| SU(2)中心符号の補償 | +282 |
| 内部blockの有限central root補償 | −10 |
| 合計 | **819** |

Compiled保存phaseは `0x1.965de502a4900p+0`、exact radは27925312186953/17592186044416。Raw保存phaseとprimitive definition追加phaseの合計はともに0。
1q P-gateからのexact rad差は−25358845202010948763/288230376151711744。従って最終会計は保存値と同じ

\[
 \chi=819\pi-24901316887139910811/288230376151711744
\]

である。ここで新しいsin/expやphase上界は計算せず、保存された約1.4142135623726069を参照した。

このscalarは、個々のSU表現と符号／root選択に依存した補償付き表現の一部である。Local差とglobal scalarは、独立に測定した物理誤差ではない。Gaugeを変えると両方が変わるが、補償を残せば全operatorは変わらない。二重計上の実装ミスと、三角不等式によって相関・相殺を失うことは区別する。

人工の `HH CX HH` 対逆向きCXでは、同じoperatorにもかかわらず旧guard0分割がglobal項>1・合計上界2を返す例を再現した。従ってH4の1.414を即座に実回路の物理的位相誤差とは扱えない。Controlled相対位相は削除せず、full operatorに残す。既存の同じpairの鋭い約1.586e-11上界も維持する。

## 3. 方法比較と採用経路

| 方法 | 相殺を保持する範囲 | 費用・適用条件 | 今回の判断 |
|---|---|---|---|
| Window拡大・近接factor融合 | 同じblock内 | k-wireのd=2^k行列、O(d²)メモリ。伝播O(g d³)、finite rootのGram比較O(d⁴)。Supportが広がる | 有効な人工例あり。H4の新partition評価は行わない |
| 厳密rewrite | 登録inverse/C4/既知CX可換・Pauli共役 | 記号計算で低費用。任意RY/RZをClifford扱いできない | 既存規則を再利用。近い角を厳密零にしない |
| 相対Pauli列＋Clifford scalar補題 | 相対列の可換融合・小回転のcharged除去 | Signed Clifford生成子が一致する場合のみ1列からscalarを囲める | 試作・人工で実証したが限定的。H4には適用不可 |
| **保持Pauli frame付きtelescope** | **境界を跨ぐPauli補正とCX共役** | **同じ局所support≤4、matrix≤16。新しい全体Clifford一致条件なし** | **次の限定pilot候補として採用** |

相対列の代替prototypeは `c_relative_pauli_enclosure`。RZのquarter-turn分解はinteger kに対して厳密で、binary64のπ近傍をCliffordへ置換する場合も、δ=theta−kπ/2を全て有理数＋π係数で残す。小回転の除去はArbで課金する近似であり、誤差0とは扱わない。直接relative作用のtriangleで相殺を回復する人工例を得た。

一方、[exact Clifford signature](../../artifacts/research_c_block_witness_cancellation/2026-11-05/technical/clifford_signature.json)では保存H4から得るClifford部分のsigned tableauは一致しなかった。これはBoolean tableau・有理数/πのdescriptor変換だけで、rotation plan、trig、gate行列、Arb上界を生成していない。この補題の前提を欠くので1列を全operator保証へ昇格しない。H4全operatorの不一致を意味するものでもない。一般の人工P0でも同じ制限を確認した。この代替prototypeを科学runnerのfallbackには含めない。

### 保持Pauli frameの導出

`c_pauli_frame_enclosure` は既存の完全factor分割を変更せず、各段でHermitian Pauli frame Pを保持する。局所supportに制限した入力Pinに対して、有限Pauli Qと補償phase βを選び、

\[
 \|C_jP_{in}-e^{i\beta_j}Q R_j\|_2\le\epsilon_j
\]

をArb192で囲む。Compiled prefix≈exp(iα) P raw prefixという関係を保ち、共通CXでは P→CX P CX の正確なsigned Pauli共役を行う。最終的に

\[
 \|C-R\|_2\le t_C+t_R+\sum_j\epsilon_j+
 \|e^{i\alpha}P_{final}-I\|_2
\]

を得る。最後の項はPfinal=Iなら|exp(iα)−1|、非自明Hermitian Pauliならmax(|exp(iα)−1|,|−exp(iα)−1|)。残ったframeを捨てない。

入力／出力Pauliのdeterminant phaseを含めたSU(d)表示と有限central rootを使い、βをexact π係数＋dyadic radとして保持する。Qはlocal midpointのPauli trace係数から**有限候補を選ぶだけ**であり、その浮動小数点値を誤差上界にしない。選んだQの残差は必ずArbで包囲する。Rootの補償phase、CXによる負符号、入力のdeclared scalarを全てαへ一度ずつ入れ、連続phase fitはない。

局所伝播と外向きrecenter半径は既存 `_matrix`、SU2/Frobenius・Gram上界、primitive translationを再利用した。Pauli選択の誤り・非最適性は上界の粗さになり得るが、選択後のArb検証を省略しないので小さい未資格上界にはならない。1qはbody×全4入力frameを並列cacheし、内部blockはframe依存順で評価する。使用したwitnessだけをmultiplicity付きで合成する。

Pauliのtensor identityへの埋込み、unitary telescopeとexact CX共役による全registerの2-norm保証であり、**Full512の有限probe／sector射影による代用ではない。列評価0である。** 非unitary literal targetは拒否し、clip2は厳密native unitary pairに限る。All-law、R→D、Head、期待bias、actual CAの資格は付与しない。

新規実装が必要だったのは、旧実装が中心符号とscalarしか次のfactorへ渡さず、非centralなPauli operatorを保持する状態を持っていなかったため。旧評価器・旧SOURCEは編集せずopt-inで追加した。

局所d=2^kに対し、現在の実装はPauli選択とroot/Gram比較にO(d⁴)、gate伝播にO(g d³)、行列にO(d²)メモリを使う。k≤4なのでd≤16であり、全registerの2^9行列は作らない。Cacheはunique 1q body×4入力frameと、実際に必要になった内部block/input frameだけを保存する。一般的な連続回転やnon-Pauli補正までframeだけで正規化できるわけではない。

## 4. 人工・回帰テストとbenchmark

新規127 PASS、関連回帰158 PASS。[新規test記録](../../artifacts/research_c_block_witness_cancellation/2026-11-05/technical/new_tests.txt)、[回帰記録](../../artifacts/research_c_block_witness_cancellation/2026-11-05/technical/regression_tests.txt)。対象は人工record／人工Qiskit回路／mock実行だけである。

多数のPauli相殺、連続回転で相殺を失う反例、境界変更、異CX skeleton、共通CX/1q、Clifford共役、global/controlled phase、SU2中心符号、相対operator、exact normal-form再構成、独立dense/既存全列との比較、全16局所Pauli×CX向きの符号、support超過、nonunitary/nonfinite、SOURCE/record/plan、資格の誤昇格、permission default拒否、partial FAIL/STOP/retry禁止を確認した。条件付きcacheの並列・直列も同じexact endpointとなった。

[人工benchmark正本](../../artifacts/research_c_block_witness_cancellation/2026-11-05/technical/artificial_benchmark.json)。3反復、method順序交替、fresh process。Wall/CPUはevaluator（plan形成を含む）、入力作成・人工compileは計時外。Affinity32と実RAMの既存resource plan、当該jobのBLAS/OMP1で列／独立taskを並列化した。RSSはprocessごとのlifetime high-waterで、合計peakではない。

| 人工例 | 全列wall／上界 | 旧block上界 | 新frame wall／上界 | 全列/frame wall比 |
|---|---|---:|---|---:|
| Pauli相殺・Full512＋2^-40 scalar | 1.12025s／2.05795e-11 | 2 | .00394581s／9.09495e-13 | 283.91 |
| P0・raw primitive、3 qubit | .0309251s／8.64604e-15 | 9.32635e-15 | .0467292s／9.32635e-15 | .66179 |
| 連続RZ相殺、Full512 | .517781s／1.81518e-56 | .249837 | .00444145s／.249837 | 116.58 |

第1例のCPU中央値は全列30.05505／新frame .003951秒、親RSS最大101632／102612 KiB（全列child最大100352 KiB/process）。第2例では新方式が遅く上界も僅かに粗い。第3例では離れた連続回転の相殺をPauli frameだけでは回復できず、rigorousでも粗い。この反例を成功扱いしない。
相対prototypeは第3例で上界0を得たが、第2例ではClifford前提未成立で上界nullとなった。異なる適用範囲の優劣を同一視しない。人工の短縮率・鋭さを実H4へ移送しない。

## 5. 新規未承認6-callの最小保存pair検証

[新契約](research_c_pauli_frame_contract_unapproved_20261105.json)は **C_RETAINED_PAULI_FRAME_SAVED_PAIR_PILOT_V1**。採用するのは保持Pauli-frame方式だけで、相対prototypeや別partitionへ切り替えない。科学SOURCEは本公開commitと [206file SOURCE](../../artifacts/research_c_block_witness_cancellation/2026-11-05/technical/SOURCE.json) で固定し、旧194fileを保持した。保存入力21件をpinする。

| logical区分 | 上限 | 固定処理 |
|---|---:|---|
| records_load | 1 | 同じC/R/word、旧sharp/loose、保存primitive/structural plan |
| identity_gate | 1 | SOURCE/input/target/word/phase/definition対応 |
| structural_plan | 1 | **旧8window分割の完全一致**を確認、変更なし |
| local_certificate | 1 | Primitive、条件付きPauli残差、CX伝播、最終phase/frame |
| comparison | 1 | 旧sharp lower/upper・旧loose2、時間と鋭さを別判定 |
| summary | 1 | 結果・未資格・STOP |

Arb192/full512、max local support4/行列16、fresh `artifacts/research_c_block_witness_cancellation/2026-11-05/frame-pilot01`（未作成）、retry/resume0。旧6-callや残枠を再利用しない。結果が粗ければVALID_BUT_LOOSERと保存して終了し、phase fit・rank/word/compiler変更・再試行を行わない。入力/SOURCE/表現/数値correctness FAILは依存callを止めpartial evidence/journal/STOPを保存する。

Passive入口は `scripts/prepare_research_c_pauli_frame.py`。[passive preflight PASS](../../artifacts/research_c_block_witness_cancellation/2026-11-05/technical/passive_preflight.json)。将来、別途明示許可後だけ `--execute-frame-pilot` を付ける。実行承認の証明書・新grant・承認記録は要求しない。
CPU/RAM/worker/wall capはnull。Unique primitiveと1q条件付きcacheを実資源に応じ並列化し、frame依存部分は逐次処理する。実行直前の実際の空き容量・quota・非干渉を確認する。新しい科学上界はまだnullである。

## 6. 未解決・保全・STOP

H4では保存値会計とexact descriptor/tableau調査だけを行った。新frame方式でH4の最終Pauliが消えるか、局所残差が小さいか、global endpointが小さいか、cache/逐次部分の費用が有利かは未検証。Genericな連続回転・non-Pauli補正まで回復できる保証もない。新しい科学的GOや全law保証を認定しない。

独立sparse checkoutを使用し、元repoのrefs/index/worktree、旧SOURCE/raw/契約/grant/ledger/marker/STOPとDirect-round側のprocess/cache/lockを変更していない。共有objectは読取り参照し、checkoutの重複を抑えた。共有環境・venvは変更していない。

[保全・技術STOP](../../artifacts/research_c_block_witness_cancellation/2026-11-05/technical/STOP.json)、[検証](../../artifacts/research_c_block_witness_cancellation/2026-11-05/technical/verification.json)。研究総括・global manifestの科学statusは変更しない。次担当はユーザー／研究C：新規6-call保存pair検証の可否を判断する。許可後にCodexがこの契約だけ実行・公開・STOPする。

公開commit・remote SHA/tree・本文・主要blob照合後 **MANDATORY STOP**。新しい実H4評価、期待bias、Stage C、actual CA、重要科学レビューへ自動進行しない。
