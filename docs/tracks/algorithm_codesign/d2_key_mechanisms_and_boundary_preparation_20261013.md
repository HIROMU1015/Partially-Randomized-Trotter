# D1 key/event費用機構とD2共通境界相殺の準備

2026-10-13 JST。**READY_FOR_SEPARATE_D2_BOUNDARY_ONE_SHOT_AUTHORIZATION**。
これは保存値の技術分析・新source準備の判定である。登録pairの費用取得は未実施、authorizationはpending、mandatory STOP。

## 固定入力と作業範囲

- GPT判断原本：[D1科学レビュー](../../research/track_b_d1_results_and_resource_mechanisms_scientific_review_20261013.md)。39,355 bytes、SHA256 `f3d1f84eece8d476bf425aefaa1847e3ef17b328c5d9afceac6be8e0eed4c344`。原本は初めにcommit `b15d2a2bc01030914d4f7ee0479e99296c919cfa`へ公開し、固定URLからbyte一致を確認した。
- D1 science S `874ca12d347f8f35e5d8d64f65f24abc4a2df523`、authorization A `f803291638d453873e4ab6f1ecf7bcc607f48e4b`、result C `fc96571881f301435a655f057709df4d6c1678e0`。
- 前段の保存分析 `2dcd56737be35526f6436a52835f59d817adf30c`を今回のbranch基点とした。90 factor分解・全affine比較や1890 event/36 sequenceの旧監査は再実行していない。
- 3-system-qubit synthetic involution、p=(1/5,3/10,1/2)、τ=5/7、state=000、σ=+、P5、r=1,2。Q0=Z0、Q1=V1†Z1V1、Q2=V2†Z2V2、V1=R_XX01(π/4)、V2=R_XX12(π/4)R_ZZ01(π/4)。R_P(θ)=exp(−iθP/2)。分子geometry/basis/DF rank/split L_Dは適用外。
- 既知developmentの同I1 provider。独立validation、分子PR総資源、最適compiler/最小T数、新algorithm採択ではない。旧closure/STOPは変更しない。

## 保存keyとT単価の対応

各rowの保存proposal qとnonzero確率Zを用い、positive eventの条件付き確率π=q/Zで集計した。

\[
d_{i,T}=\ell_i+\sum_k\mu_{ik}t_k,\qquad
\ell_i=\sum_e\pi_e C_{T,e}^{literal},\quad
\mu_{ik}=\sum_e\pi_e m_{e,k}.
\]

literal T/T†はproviderに含まれる正確な回転、t_kは保存されたR placeholderのsequence価格である。Rの逆符号は同じ実sequenceのadjointで、T/T†価格は同じ。R angleはsigned atan(ratio)、native primitiveはexp(−i sign atan(ratio) Z/2)、controlled event全体は二つのsigned primitiveを含む。新しい角度・合成は取得していない。

12 row・1890 eventへの索引を作り、event index、reduced/raw word、child/Pauli、controlled scalar phase、係数/proposal/weight、R位置と符号、inventory SHA key、sequence SHA、価格・誤差記録へ対応付けた。全rowで上式のT単価および旧2NrZ^r倍のT切片にexact rational一致した。

| r | 方式 | literal条件付きT | 合成条件付きT | 合計d_T |
|---:|---|---:|---:|---:|
|1|ordinary|3.014642|136.931723|139.946365|
|1|partial|3.075166|133.293214|136.368380|
|1|closed P3|3.169200|140.108119|143.277319|
|1|local full|3.141354|136.348640|139.489994|
|1|closed P5|3.141354|136.348640|139.489994|
|1|CTS|0|114.116971|114.116971|
|2|ordinary|2.728524|136.230413|138.958937|
|2|partial|2.734266|143.849804|146.584070|
|2|closed P3|2.762458|140|142.762458|
|2|local full|2.760119|136.014108|138.774227|
|2|closed P5|2.760119|136.014108|138.774227|
|2|CTS|0|135.587907|135.587907|

条件付き単価だけで全資源順位を決めない。local fullとP5の同単価も、同じN・acceptance・全trial量子費用を意味しない。

## 先頭root三件と全eventの単価差

ここでroot三件は保存index 0–2、child 0/1/2の対応群。partialにはそれ以外にraw wordがreturnしてreduced word=[]になる45件がある。これらを先頭三件へ混ぜない。異なる方式の他eventを無理に一対一対応させていない。

符号はすべてP5−partial。rootの価格/確率差は対称恒等式
Δ(πc)=((π₀+π₁)/2)Δc+((c₀+c₁)/2)Δπで配分した。

| r | 全単価差 | 先頭三件root差 | root価格部分 | root確率部分 | その他event差 |
|---:|---:|---:|---:|---:|---:|
|1|+3.121614|+4.938494|+3.491446|+1.447048|−1.816880|
|2|−7.809843|−7.571561|−7.703222|+0.131660|−0.238282|

root差/net単価差はr=1で158.203%、r=2で96.949%。符号付きの記述的配分であり、一意な因果割合ではない。r=1では他eventがrootの増価を部分的に相殺する。literal単価差はそれぞれ+0.066188、+0.025853で、残りは合成key価格×使用回数の差に対応する。

root外のreduced-word長2/4/追加空wordによる差は、r=1で−1.454830/−0.053180/−0.308870、r=2で−0.218353/−0.000466/−0.019462。全category差の和もexact一致を確認した。

## key価格の記号的感度

旧law・角度・bias・N・acceptanceと他key価格を固定する。全T切片差は

\[
\Delta G_T=\Delta L+\sum_k a_k t_k,
\quad a_k=2N_{P5}rZ_{P5}^r\mu_{P5,k}-2N_{partial}rZ_{partial}^r\mu_{partial,k}.
\]

各keyを単独変更したときのexact tie、t_k≥0での優劣半直線、複数keyのtie hyperplaneを保存した。以下は全T切片h_T=0の境界であり、単価だけの境界と区別する。

| r | key tangent | 所属 | 現価格 | 全T切片のtie価格 | P5が低い領域 |
|---:|---|---|---:|---:|---|
|1|2189372/2921709|P5 root|68|68.933703|t<tie|
|1|2845/3717|partial root|66|65.086863|t>tie|
|1|5/21|partial tail|70|63.612900|t>tie|
|2|18205486/50387421|P5 root|68|72.463861|t<tie|
|2|2333/6426|partial root|72|67.555922|t>tie|

例えば記号的にはr=1のP5 rootが69 Tなら反転し、r=2は73 Tなら反転する。partial rootをr=1で65 T、r=2で67 Tに置く場合も反転する。これはその精度・角度に対応する合法sequenceが取得可能という主張ではない。合成器のparity制約やerror guardを満たす実価格への移送は未検証。価格差を未知keyへ無料注入せず、観測価格と反実仮想を分けた。

`strict_sign_flip_integer`は非負整数の反転例であり、下方向には0を示すことがある。最小の価格変更ではない。必要な境界はexact `tie_price`と係数符号を参照する。

全価格を同じ倍率で動かす感度も単独key感度とは異なる。r=1のuniform-scale tieは負のため倍率≥0の登録モデル内ではP5の低費用関係が残る。r=2は約0.0012894でtie。これを新しい合成条件への頑健性の証明としない。

## D2：結果前固定した共通ステップ境界相殺

[機械可読contract](../../../artifacts/track_b_d2_boundary_source/2026-10-13/v1/contract_v1.json)が正本。旧D1のcross_step_fusion=falseは書き換えない。

1. r=2の全六方式だけを対象に、固定D1のpositive保存step IRを時間順に二つ結合する。
2. D1と同じexact adjacent inverse cancellationを境界にも適用する。非可換因子の並べ替え・角度加算・synthesis sequence内部の追加簡約・汎用compilerは使用しない。
3. positive supportは273/264/258/63/63/24、ordered pairは74,529/69,696/66,564/3,969/3,969/576、計219,303。一つも欠ければcompleteとしない。P5のみの簡約や成功pairの抽出はしない。
4. q、weight、N、Taylor/native誤差上界、state、confidence、旧sequence価格をそのまま使う。zero-fillは全stepの古典draw先行、どちらかがzeroなら初期化前にwhole zero。
5. 新期待費用は2N Σq_iq_j C_after(i,j)、prep/readout slopeは旧2NZ²。workspace・未知classical費用は旧扱いを保持する。旧accepted-tail/hard費用上界を保守的に残し、新worst pairはdiagnosticとして保存する。
6. 旧error boundはそのまま保守的に保持する。signed Rは同じ実sequenceとadjointなので、逆対の除去は理想回路だけでなく、その実装sequenceの積も厳密に保つ。literal Clifford/Tおよびcontrolled scalar phaseも同じinverse規則で扱う。相殺を理由にbias/Nを小さくしない。
7. 元D1のaxis allowanceをr=2のsubsetへ据え置く。12 axesと6 rowsのfailure allowance和は1/40≤1/20。過去runとの新しいjoint測定実験のcoverage claimを追加しない。実量子shotsは0。

成功は **D2_BOUNDARY_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW**、technical failure/cap/incompleteは **D2_TECHNICAL_INCONCLUSIVE**。性能差の符号・大小に関係なくmandatory STOP、prefixから研究勝敗を作らない。P5の全資源優越・一律5%改善を成功条件にしていない。

### 実装と資源

- [境界kernel](../../../src/trottertracks/algorithm_codesign/rte_development/boundary.py)：各step内部が既にreducedであることを確認し、左suffix/right prefixだけを比較。step内suffix価格を共有する。219,303本の結合IRや巨大pair JSONを保持しない。
- qを共通denominatorの整数massへliftし、内側のpair loopではFraction reductionを繰り返さない。各rowの結果だけexact Fractionへ戻す。全positive pairのmass conservationを検査する。
- [runner](../../../scripts/tracks/algorithm_codesign/run_d2_saved_boundary_one_shot.py)／[外側observer](../../../scripts/tracks/algorithm_codesign/observe_d2_saved_boundary_one_shot.py)／[launch gate](../../../src/trottertracks/algorithm_codesign/rte_development/boundary_launch.py)。pending時は登録inputより前に拒否。
- 固定旧isolated runtime、1 process、numeric threads1。CPU900秒、RSS512 MiB、AS1536 MiB、出力16 MiB、log64 KiB、review file512 KiB、pair219,303、denominator16,384 bits。phase/total wall capなし。新native compile/synthesis/matrix/sampling/LP/DF/NPZ/GPU呼出し0。
- POSIX周期guard、AS制限、外側wait4のexit/signal/CPU/RSS/log記録、独立terminal・STOP。source S→authorization-only直接子A、remote A一致・clean・hash検査後、exclusive新markerを消費。旧markerは使用しない。失敗でもretry0。
- 今回はsaved-key分析0.238秒wall/0.161秒CPU/peak23,964 KiB。登録pair kernelの性能は未測定。1 processの設計は共有IRと整数集計を優先したもので、速度倍率を実測したとはしない。実行前にはCPU affinity/メモリと固定runtimeを再確認する。GPUは不要でquery0。

### focused validation

[45 tests](../../../artifacts/track_b_d2_boundary_source/2026-10-13/v1/focused_tests_final_v1.json) PASS。初回43 PASSも別記録で保持。既存D1の57件や旧監査を再実行していない。

登録外の2-qubit、atan(1/9)、atan(2/13)、人工価格・確率・native列・mock sourceで、旧inverse pass一致、strict matrix phase/order、phaseful sequenceと実adjoint、no commutation/no angle fusion、zero-fill、整数mass集計、missing key、cap、direct-child/dirty/contract/marker拒否、UTF-8復元・partial保存を検証した。

追加二件のrunner統合testは一時ディレクトリの架空六行のみ。正常終了と二行目の人工停止を検査し、technical prefix不使用、marker・terminal・STOPを確認した。テストstdoutのCOMPLETE/INCONCLUSIVEはmock結果であり、登録D2の実行結果ではない。

保存分析の初回には先頭三件と全空word群を同一視したassertが停止した。分類を分離して修正し、最終全row/category/rootの恒等分解を通過した。[失敗記録](../../../artifacts/track_b_d2_boundary_source/2026-10-13/v1/saved_analysis_preliminary_failure_v1.json)を保持。これは保存分析の実装過程で、D1 failure/科学run retryではない。原科学値・分類は不変。

## 次の停止点

技術準備に科学契約の変更が必要な問題は見つかっていない。新しいGPT詳細sourceレビューを自動要求しない。次は**今回公開した新source Sに対するユーザーの明示D2 one-shot実行指示**である。

承認後にSの直接子Aを専用実行worktreeから作成・pushし、observerを一回だけ呼ぶ。今回A/登録marker/登録pair費用は0。結果取得後に重要な科学解釈が必要になれば、ユーザーの開始承認を得てGPT側へ戻す。

別precision/seedの合成条件は今回固定も採択もしていない。現在は境界相殺対照一つを優先し、新しいr、provider、分子、SP、online sampler benchmark、G10/RA-D0再開へ進まない。

### 証拠入口

- [key分析manifest](../../../artifacts/track_b_d2_key_mechanisms_preparation/2026-10-13/v1/analysis_manifest_v1.json)、[row/event一覧](../../../artifacts/track_b_d2_key_mechanisms_preparation/2026-10-13/v1/row_event_index_v1.json)。全新JSONは512 KiB以下。
- [r=1 exact key感度](../../../artifacts/track_b_d2_key_mechanisms_preparation/2026-10-13/v1/P5_vs_partial_r1.json)、[r=2](../../../artifacts/track_b_d2_key_mechanisms_preparation/2026-10-13/v1/P5_vs_partial_r2.json)。
- [input identity](../../../artifacts/track_b_d2_key_mechanisms_preparation/2026-10-13/v1/input_identity_v1.json)、[技術準備manifest](../../../artifacts/track_b_d2_boundary_source/2026-10-13/v1/source_manifest_v1.json)、[保護・scope記録](../../../artifacts/track_b_d2_boundary_source/2026-10-13/v1/protection_and_scope_receipt_v1.json)。
- 原D1 result C・前段分析のURL/bytes/SHAはそれぞれの元handoffと今回のidentity表を参照する。原result・source・authorization・marker・STOPは不変。

親worktreeについて、HEAD・dirty path一覧・index差分は開始時と一致した。未stage差分の全体hashは作業中に変化しており、開始時にpath別snapshotを取得していなかったため、変化の発生元は特定していない。本作業の書込みは新worktree内のみで、親のreset/clean/stash/stage/commitは行っていない。親全ファイルのbyte不変を確認したとは表現しない。固定D1および旧保護対象344ファイルは別途全SHA256一致を確認した。
