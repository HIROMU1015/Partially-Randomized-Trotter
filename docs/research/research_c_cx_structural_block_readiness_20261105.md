# 研究C：CX構造差と内部block包囲の技術準備（2026-11-05）

固定基点は `f6b1516116d40c3d8d1467e476abb032a0f562b6`。
Branchは `codex/research-c-cx-structural-block-readiness-20261105`。
今回の結論は **内部block方式の人工実証と保存H4の構造適用性PASS**。実H4の新しい上界・速度は未評価であり、どちらもnullのまま維持する。

対象条件は直鎖H4・1.00 Å/STO-3G、number CDF8/元順LD2、partial S2/s1/r32/K0、h=δ=.1、controlled full512、Arb192。固定F6:0/F7:6,7交互32-event word、候補SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。既存P0/Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/MPFR160 declared phase、Gaussian/relative fusion、53-bit IID-grid、own-DF oracle・状態準備費用除外、既存CA/RPE政策を変更していない。

正式共通指示原本 `research_project_instructions_unified_2026-10-25.md` の§8.1・8.4・10を適用した。実H4の新しいsampling/build/compile/Arb評価、旧pilot再実行、重要科学レビューは0。別Codex-DのDirect-round実装・branchには触れていない。

## 1. 保存構造から確認した事実

[構造監査JSON](../../artifacts/research_c_cx_structural_blocks/2026-11-05/technical/structural_audit.json)は保存gate descriptorと旧planだけを読む。配列decode・operator評価・Qiskit再compileは行わない。

旧4-call pilotのcompiled/raw候補の展開gate数は11,386/32,073、正規化CXは2,894/2,912。旧inverse消去0/9,752組は全gate種の集計である。向き付きCX tokenの決定論的alignmentでは共通2,892、未対応compiled2/raw20、非一致区間8件だった。

| 区間 | compiled CX index | raw CX index | 未対応の向き付きCX |
|---|---|---|---|
| 1 | [2,2) | [2,4) | raw (1,0)×2 |
| 2 | [20,20) | [22,26) | raw (1,0)×2、(6,7)×2 |
| 3 | [38,38) | [44,46) | raw (6,7)×2 |
| 4 | [988,988) | [996,998) | raw (1,0)×2 |
| 5 | [2846,2846) | [2856,2858) | raw (1,0)×2 |
| 6 | [2872,2872) | [2884,2890) | raw (1,0)×2、(8,7)×2、(6,7)×2 |
| 7 | [2882,2884) | [2900,2900) | compiled (8,7)×2 |
| 8 | [2892,2892) | [2908,2910) | raw (6,7)×2 |

これは一つの構造alignmentであり、歴史的なcompiler passの一意な復元ではない。差引き18を「18個のCX削除」とは解釈できない。間の1量子ビットgateがあるため、この表の同向CX対を直接inverseとして削除することもできない。

旧登録のcontrol-diagonal/target-X軸可換規則を構造的に適用してもCXは2,894/2,912のまま。今回追加した厳密C4積もこの8区間を消さなかった。したがって既存の単純正規化だけでは旧shared-skeleton gateを通せない。

## 2. Compiler経路と原因特定の限界

確認した固定SOURCEは次の経路である。

1. `pr_dictionary_validation/gates.py::compiled_wrapper`：公式definitionをnative basis＋Uへpre-unrollし、その後固定P0 `transpile` を呼ぶ。素のtranspileだけへの置換は行わない。
2. `c1_h6_compiler_precision.py::compiled_wrapper`：MPFR160のdeclared phaseを保ち、補償offsetを一度除きbinary64へ一度castする。出力operatorからのphase fitはない。
3. Qiskit1.3.0の `preset_passmanagers/builtin_plugins.py` のopt1：`Optimize1qGatesDecomposition` と、CXを含む `InverseCancellation`。前者は1q runを行列化・再合成し、そのglobal phaseをDAGへ渡す。後者は隣接する登録inverseを消す。
4. `BasisTranslator`／`UnitarySynthesis` はtranslation経路の構成要素。対して旧局所planはraw primitiveの公式definitionを保存候補へ展開し、disjoint/inverse正規化だけを施す。**二つは同じ正規形を生成する処理ではない。**

固定Qiskit実装は既存venvから読み取り、変更していない。変換passごとのsnapshot/callback履歴は保存されておらず、どのpassが各区間を変えたかは未確定。1q再合成がCX間のgateを変えてinverse消去を露出する機構はあり得るが、この保存H4の個々の差の確定原因とは扱わない。CXとrotationの共役・diagonal可換、definitionの選択、順序の違いも最終recordだけから寄与を分離できない。

Head/Gaussian/relative fusionは既存raw生成の上流条件として保全し、新しい融合を導入していない。最終descriptorから個々のCXをhead/frame起源へ一意に帰属させていない。Global phase・controlled相対位相は誤差対象に含め、CX数だけからoperator一致／不一致を判定しない。

## 3. 採用方式と保証

新規opt-in `c_cx_structural_blocks.py` を追加した。旧 `c_compiled_raw_local.py`、`c_arb_column_batch.py`、fixed-certificate、全旧SOURCEは変更しない。

**厳密部分**：disjoint supportを跨ぐinverse消去、宣言済みSX/SのC4積のみ。SX⁴=S⁴=Iをそのgate自身のphase込みで使う。RZ(binary64 π)をZと同一視せず、微小rotationも落とさない。Clifford共役・controlled rotationを新しい未検証rewriteへ置き換えず、必要なものは内部blockで包囲する。

**内部block部分**：向き付きCXと末尾wire tokenをalignmentし、非一致区間と前後一つの共通anchorをwindowへまとめる。末尾へ達したwindowは全terminal wireを含める。各windowを両armのsupport結合グラフで分解し、互いにdisjointな成分だけを別factorにする。各成分内のgate順序は維持する。全input itemを一度ずつ覆うことを再構成して確認する。Alignmentは境界を決めるheuristicであり、blockの一致や近さを仮定しない。

固定政策は `guard=1, exact_clifford=true, max_support=4`。これは局所行列を最大16次元とするアルゴリズム適用範囲であり、CPU/RAM/worker制限ではない。support超過は `NOT_APPLICABLE_BLOCK_SUPPORT`、上界null。Full512列伝播への自動fallbackはない。

**数値部分**：既存primitive bridgeの独立Arb評価を再利用し、raw primitive→保存候補definitionの丸め誤差をmultiplicity付きで残す。外側の1q対は既存SU(2)評価、内部blockは局所全行列をArb192で伝播する。各gate後のrecenter半径を加算し、真のunitary左作用が既存誤差を増幅しないことを用いる。

局所差行列Mのoperator上界は、Frobenius上界と、M* MのGershgorin上界の平方根の小さい方にrecenter誤差を加える。後者は

\[
 \|M\|_2^2\le\max_i\{(M^*M)_{ii}+\sum_{j\ne i}|(M^*M)_{ij}|\}
\]

を外向き丸めする。連続phase fitは行わない。局所determinant phaseをexactなπ係数＋dyadic角として抽出し、SU(d)の有限central root `exp(2πij/d)` の中から局所boundを選ぶ場合も、そのrootの**正確な補償phase**をglobal項へ残す。

各factorを A_j=e^(iα_j)U_j、B_j=e^(iβ_j)V_j とし、補償付きroot γ_jを用いた ε_j≥||e^(iγ_j)U_j−V_j||₂を得ると、全体は

\[
 \|C-R\|_2\le t_C+t_R+\sum_j\epsilon_j+|e^{i\chi}-1|,
 \qquad\chi=\Phi_C-\Phi_R+\sum_j(\alpha_j-\beta_j-\gamma_j).
\]

t_C,t_Rはprimitive bridge。未知項は0補完しない。1q版の±符号選択も同じ補償規約である。Gateのlittle-endian qubit対応を維持し、局所support以外へのtensor identityとregister permutationはoperator normを変えないため、これは全512入力・全出力に対する2-norm上界を構成できる。有限probeやsector射影で代用せず、全512列を評価したとも記録しない。

最終clip2は**厳密native unitary pairだけ**に適用する。Nonunitary literal targetは入力で拒否し、この議論をR→Dへ流用しない。旧column-only certificate admissionも新proofを拒否する。Full512証拠取得後であっても、全word/compiler成功・全law synthesis・期待bias・actual CAへ自動昇格しない。

## 4. 保存H4の構造適用性

固定政策では8 window、15内部成分（support2が5、3が9、4が1）、最大29gate/arm。共通CX factor2,876、1q factor5,761、内部factor15、数値block bodyのunique数245。Raw primitiveのunique定義175と合わせ420のunique局所算術jobを予定する。Multiplicityは保持する。全H4 operatorに対するArb評価はまだ0。

Guard0では10内部成分、最大support3となるが、同じ科学pilotで結果を見て切り替えない。Guard1は近傍anchorを含めて内部相殺を捉える技術上の事前選択であり、実H4での鋭さを保証するものではない。

## 5. 人工テストと費用

新規テストは人工record・人工Qiskit回路・mock runnerだけ。CX同数／異数、可換／非可換、Clifford共役、異なるCX順序、support成長、phase、controlled P、内部相殺、粗いtelescope、蓄積誤差、非finite／nonunitary、SOURCE/record/plan、false full512昇格、permission default拒否、partial FAIL/STOP、retry/resume拒否を検証した。独立列伝播、人工dense operator、全factor再構成とも照合する。

結果の正本は [新規テスト](../../artifacts/research_c_cx_structural_blocks/2026-11-05/technical/new_tests.txt) と [既存回帰](../../artifacts/research_c_cx_structural_blocks/2026-11-05/technical/regression_tests.txt)。新規79 PASS。既存回帰は476 PASS・1 FAIL・3 deselected。FAILは `test_real_contract_passive_cold_without_numerical_imports` が旧Stage Rの消費済み出力を `FRESH_NO_RETRY_RESUME` として拒否したもの。旧テスト・gate・入力・出力は基点と不変であり、新方式の数値的FAILではない。既知の消費済みH/P/local-pilot cold preflight 3件は事前除外した。旧出力の削除・契約書換え・科学再実行でテストを通すことはしていない。

[人工benchmark全記録](../../artifacts/research_c_cx_structural_blocks/2026-11-05/technical/artificial_benchmark.json)：各method3回、順序を交替しfresh processで測定。Wall/CPUはevaluator内（block側はplan生成を含む）。RSSはprocess lifetime high-waterで、全worker合計peakではない。Affinity32、実際のquota/RAMを参照する既存resource planを使い、列伝播は独立列を並列化した。BLAS/OMPは当該jobだけ1thread/processとしてoversubscriptionを避けた。

| 人工例 | 列伝播 median wall | 新block median wall | 列/block | 列伝播上界 | 新block上界 |
|---|---:|---:|---:|---:|---:|
| CX skeleton不一致・Full512、phase 2⁻⁴⁰ | 1.15519 s | .00304152 s | 379.81 | 2.05795e-11 | 9.09495e-13 |
| Raw primitive＋固定P0、3 qubit | .0304678 s | .0363145 s | .83900 | 8.64604e-15 | 9.32635e-15 |
| 同skeleton・block間で相殺する反例、Full512 | .531294 s | .00185440 s | 286.50 | 1.81518e-56 | .249837 |

第1例のCPU中央値は列30.3250/block .003037秒、親RSS最大101,856/101,796 KiB（列child最大100,352 KiB/process）。第2例では新方式が遅く、上界も少し粗い。第3例はrigorousでも後続校正に役立つ鋭さを失う明確な反例で、成功扱いしない。旧strict方式は第1例でNOT_APPLICABLE、上界null。数字は人工問題の実測であり、実H4のspeedupではない。

## 6. 新規・未承認6-call契約

[固定契約](research_c_cx_structural_block_contract_unapproved_20261105.json)は**新たな独立6-call枠**であり、旧6-callの未使用2 unitsも、旧4/7-call枠も再利用しない。

| logical区分 | 上限 | 内容 |
|---|---:|---|
| records_load | 1 | 固定C/R、旧primitive plan、旧bound、wordを読む |
| identity_gate | 1 | hash/SOURCE/phase/wordと保存definitionの完全descriptor replay |
| structural_plan | 1 | 固定rewrite/window/supportで新planを保存 |
| local_certificate | 1 | 全primitive/1q/internal block＋phaseの包囲 |
| comparison | 1 | 保存7-call lower/upper・歴史的wallとの照合 |
| summary | 1 | 結論・未資格事項 |

科学SOURCEは [194file manifest](../../artifacts/research_c_cx_structural_blocks/2026-11-05/technical/SOURCE.json) と本公開commitで固定する。旧187fileのhashはそのまま、新規7実装/test/scriptを追加した。契約はSOURCE manifest・保存inputのSHA256を固定。対象は保存済みの同じ1 word/pairのみ、Arb192/full512、retry/resume0、fresh `artifacts/research_c_cx_structural_blocks/2026-11-05/block-pilot01`（未作成）。非対応ならload/identity/plan/summaryの4callsでSTOP、Arb0/null。正しさFAILは依存callを停止しpartial evidence/journal/STOPを保存する。

Passive入口は `scripts/prepare_research_c_cx_structural_blocks.py`、将来の明示許可後だけ `--execute-block-pilot` を付ける。現時点は [passive preflight PASS](../../artifacts/research_c_cx_structural_blocks/2026-11-05/technical/passive_preflight.json)、科学実行未承認・非実行。CPU/RAM/worker/wall capはnull、局所算術jobを実資源に応じ並列化し、共有環境は変更しない。

比較対照の旧上界は約1.585733058571e-11、wall1173.086005588秒。正本は保存exact endpointであり新計算しない。新方式のplan/identity/certificate各時間と全journalを記録し、独立fresh baselineとの速度比較とは区別する。新上界が粗ければVALID_BUT_LOOSERと記録し、phase fit・条件調整・別partitionの再試行を行わない。

## 7. 残る事項・次担当・STOP

技術的には保存pairを局所supportで完全分割でき、人工入力で厳密包囲を実証した。実H4で上界が十分鋭いか、175定義＋245blockのArbと並列起動の費用が有利かは未確認。全law、compiler失敗scope、Direct-round、期待bias/Stage C/CAは別課題のまま。

次担当はユーザー／研究C：この**新規6-call固定保存pair pilot**の実行可否を判断する。別途明示許可があればCodexがこの1契約だけ実行・公開・STOPする。今回は科学計算を実行しない。

旧SOURCE/raw/契約/grant/ledger/marker/STOPと部分失敗結果は保全。[保全・技術STOP](../../artifacts/research_c_cx_structural_blocks/2026-11-05/technical/STOP.json)。新科学成果や科学的GO判定ではないため研究総括・validation manifestの科学statusは変更しない。公開SHA/tree・主要blob照合後 **MANDATORY STOP**。
