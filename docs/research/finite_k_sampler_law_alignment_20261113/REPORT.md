# ideal finite-K lawとPCG64/CDF samplerの接続調査

2026-11-13。技術調査・数学的評価方法・人工テストを完了。**実H4のlaw距離と価格への影響はUNKNOWN。既存26 E[S]は不変で、適格なE[C]はまだ供給しない。**

同じ数学的な重みを使う生成構造は一致している。binary64正規化、CDF、53-bit格子の差は既存helperを用いて有限個の確率から評価できる。一方、固定seedのPCG64列をIID列と同一視する保証は、その評価から得られない。この二つを分離する。

## 対象・SOURCE

対象は固定Linear H4/1.00 Å/STO-3G・保存tol-only DF10、L_D=1/S2/ΔPF=.1/finite-K2/h=.1/M=12/P=0、元S0 exactκ、元H/state/snapshot・fragment/basis/qubit順・scalar一回・symmetric-directional controlled full Hadamard X/Y。canonicalは`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`、preparationは`b99720d6189a9975a46f064ba895d2f83514671e668ce42c34a4d71e834568d7`。324非identity component、9 basis、13 lawのR=2,4,8,16,32,64,128,255,510,1020,2040,4080,8160を変更しない。

固定参照と所在・blob・SHA256・symbol行は[SOURCE_INDEX.json](SOURCE_INDEX.json)、NumPy一次SOURCEは[NUMPY_SOURCE.json](NUMPY_SOURCE.json)。確率配列を使う新しい数値解析は行わず、ソース本文、公開の説明・接続metadata、固定成果のbytesだけを確認した。

|役割|固定SOURCEと実際に確認した箇所|確認結果|
|---|---|---|
|Cの全13round結果|`9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f`、`h4_structural_schedule_results_20261113/` REPORT/HANDOFF、coverage/CSVの固定bytes|ideal source lawのS。既存26値・区間・K0/K1を保全|
|C科学SOURCE|`de2427d4a31d6b4b39d2c8eb8b63e3c972ffb89e`、`h4_structural_kernel.py`のinteger_weights/discrete/mixture/Model/Wire|中心値は元nominal浮動小数点、区間は保存binary64 raw weightの厳密比を含む。CDF実効確率の区間ではない|
|finite-Kと実sampler|`71ba49bb28b295aa28e51101bc3b1bf07d500767:src/trotterlib/rte.py`|C科学SOURCE、C1/D参照provider `c31ef424175c02379eafebe8fc6c7548c1d9aabe`と**全blob一致**。現在のtoy回帰対象6関数もAST一致|
|元request/event接続|providerの`df_partial_s2.py:make_df_partial_s2_step_request`→`df_rte_circuit.py:sample_occurrence_request/sample_events`→`rte.py:sample_rte_events`|同じcomponent順・distribution・seedを渡す。単なる同名lawの照合で終えていない|
|seed/CDF既存helper|`155529e20a8fac39e00786c40b57830b71203c4f:scripts/h46_long_time_native.py`、trial_seed/sample_requests/cdf_law|既存53-bit写像helperを変更せず再利用。既存rarityは今回実行しない|
|C1 paired S/C|`1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7:r_price_phase1_science.py` Backend|同じrequest/event digestでSとCを作り、同じpathを両軸で利用|
|D残差|`af6f27d7a6467bb355686421acd067544edcbbd2`、compiled_residual.py/HANDOFF_C|paired full-wrapper C−S。有限cohortと母平均を区別する設計|
|D P0|`442cafa14f75aeee9e04696df54aaaff1f53260e`、controlled_zz_p0.py/REPORT|local controlled-ZZ loweringの診断。sampler/IID/全path残差平均の証明ではない|
|B|`bf515a976f1987ad6a19c61d14689b27b0894542`、compiled_residual.combine/b_price_recordとS3 metadata|law_context不一致をreasonに残し、和は常に未資格。law_binding同値だけで距離0を認定しない|

## 数学的lawと実装の対応

有限K2はorder k∈{0,2}。数学的重みは `|τ|^k/k! × sqrt(1+(τ/(k+1))²)`、その和で正規化する。componentはtailの絶対係数に比例する。`_paired_order_weight`はlog/lgamma/exp/hypot、normalizationはmath.fsum、確率はbinary64除算で計算する。τ自体のλ・時間・除算、componentの元正規化、libm重みの数学的値との差も、CDF写像の差と別に扱う。

`iter_sample_rte_events`はPCG64(seed)を生成し、orderをchoiceで選択してから、同じGeneratorでcomponentをorder+1個、復元抽出する。index列の先頭をrotation、残りをproductsとする。実際の作用順はproducts→rotation、phaseは(-1)^(k/2)、角度はatan(τ/(k+1))。componentのsign・basis・relative phaseはevent metadataへ保持される。

従って、**同じ重み・τ・metadata、厳密算術・独立連続一様乱数という数学的モデルでは生成lawが一致**する。固定物理H4に対する浮動小数点の一致/不一致を、新解析なしに断定していない。

`_make_event.event_probability`は元weightの積と除算で作るnominal metadata。samplerはこれを選択確率として読まず、order/componentのCDFを使用する。このmetadata値やK0のγ16 factorization検査を、CDF実効確率の証拠へ転用しない。

## CDF・有限乱数・PCG64

[NumPy choiceの一次SOURCE](https://github.com/numpy/numpy/blob/9815c16f449e12915ef35a8255329ba26dacd5c0/numpy/random/_generator.pyx#L843)では、pの検査後、binary64累積和を最後の値で割り、right境界比較で選択する。[PCG64のdouble入口](https://github.com/numpy/numpy/blob/9815c16f449e12915ef35a8255329ba26dacd5c0/numpy/random/_pcg64.pyx#L41)と[uint64変換](https://github.com/numpy/numpy/blob/9815c16f449e12915ef35a8255329ba26dacd5c0/numpy/random/_common.pxd#L67)から、上位53bitを2^53で割る格子を確認した。公式v1.26.4 tagのcommitを照合し、検証環境もNumPy1.26.4。ただしインストールbinaryの全SOURCE同一性を証明する検査ではない。

連続一様モデルFと、一様53-bit格子モデルGを区別する。CDFの閾値をc_i、D=2^53とすれば、Gのカテゴリmassは `(ceil(D c_i)−ceil(D c_(i−1)))/D`。right比較では境界上の乱数は次の非空区間へ進む。正の微小確率が格子上で0になる例も、先頭微小区間がu=0を受け取る例もある。実H4でそうなったと主張するものではない。

trial_seedはcandidate ID/cohort/path index/outer/pieceを含むJSONのSHA256先頭16桁から得る整数。axisは含まれず、X/Yは同じpathの対応応答。sample_requestsはouter/pieceごとにGeneratorを再生成する。order0/2でcomponent抽出数が変わるため、後続のPRNG消費位置も変わる。異なるround/cohortのseed namespaceやseedの非重複は、streamの独立性の証明ではない。

固定seed・固定入力の列は決定論的。seedをどの確率空間から選ぶか未指定なら、実列のlawとideal IID trajectoryとのTVを小さいCDF誤差だけで保証できない。有限の周辺分布検査やMonte Carloも一般のIID証明を与えない。本調査のPCG64 joint-law距離はUNKNOWN。

## 誤差評価と長trajectory

[MATH.md](MATH.md)に、次を分離したexact-rational診断と条件付き上界を記す。

1. 保存binary64 raw weightの厳密正規化Pと、choice入力の厳密正規化Qの差。
2. Qと、丸め済みCDFを連続一様に作用させたFの差。
3. Fと一様53-bit格子Gの差。
4. order/componentからeventへのcoupling、eventからtrajectoryへのcoupling。
5. SまたはC−Sのfull-support oscillationを使う期待値差。

位置数nのtrajectoryでは、**一致した履歴に条件付けた各event距離がδ以下**という条件があればTV≤min(1,nδ)。独立一様格子モデルは十分条件の一つ。周辺のδだけでは不十分で、実PCG64の固定seedへ自動適用しない。m12のn=33,423,360でもevent列は生成不要だが、union boundが大きい場合は不有用な上界として残す。

今回、実H4の確率配列からδや価格誤差を計算していない。libm入力誤差、実PCG64 joint-law、full-supportの価格範囲、rare条件付き残差は未取得。**実価格差が小さい/無視可能であるとの結論は出ていない。**

## 平均価格との接続

IをCのideal source law、QをDが標本化すると仮定したlaw、D_res=C−Sとする。`E_I S + mean_cohort(D_res)`を使う場合、Iをtargetとする期待biasは`E_Q D_res−E_I D_res`。Qをtargetとするbiasは`E_I S−E_Q S`。どちらを目標にするかで必要な範囲上界が異なる。有限cohort標本誤差、law差、長q/Rモデル系統差を混ぜずに残す。

同じtrajectory・axis・full wrapperのS/C pairing、event digest/seed/cohort/index、候補/τ/R/controlled profile、Sの元Accountingを照合する。Dのcompiled loweringを変えても、本law調査は元Sを変更しない。forced rareは別stratumであり、重みなしpoolは禁止。未観測order2寄与は0ではない。既存K1区間は構造/数値誤差用で、law距離やcohort CIではない。

Bへの具体的情報は[HANDOFF.md](HANDOFF.md)と[LAW_CONTEXT.json](LAW_CONTEXT.json)。SOURCE一致、数学的生成構造一致、CDF距離の実数値、PCG64 joint-law、科学採用資格を独立に保持する。平均compiled価格recordを新規生成せず、既存B/D codeも変更していない。

## 実装・検証・保全

新module `src/trottertracks/finite_k_sampler_law.py`は既存cdf_lawをAST抽出して呼び、P/Q/F/GのTVをFractionで分解する。K2 event bound、履歴条件を明示したtrajectory bound、既知oscillationからの期待値差だけを追加した。physics reader・sampler・compile・fit・controllerは追加していない。categorical処理はsupport長に線形、trajectory union boundは位置数に依存する列生成を持たない。実H4上の速度は測定していない。

[TESTS.txt](TESTS.txt)：新12人工test PASS、既存sampler関連toy回帰5 PASS（11 deselected）。zero/微小確率/CDF境界/格子/丸め、K2条件付き分布の人工全列挙、等しい周辺lawでもjoint lawが異なる反例、未知量のnull保持を確認。作者自身の技術確認であり、独立科学レビューではない。既存toy回帰で2×2 Pauli等の人工event生成はあるが、実H4 event生成は0。

既存C branch `codex/r-variable-compiled-price-design-20261111`を継続。旧C1/C2/connected-cluster、S0a失敗/成功、K0/K1・全26値・原本input/raw/cohort/grant/ledger/marker/KEEP/STOPを変更していない。新科学再解析・実sampling・simulation・build/compile・fit・E[S]/価格/N/G評価は0。他担当の作業領域へ変更なし。

## 次の担当とSTOP

- **ユーザー**：必要なら[SCIENCE_PLAN.json](SCIENCE_PLAN.json)のF0（保存確率だけの有限law診断）の実行を承認する。固定seedを実験上のIIDモデルとして扱うか、厳密joint-law保証を要するかも科学採用時に明示する。
- **Codex C**：F0が承認された場合のみ、1 preparationの324 componentと13 order分布について、CDF/格子の距離と条件付きtrajectory上界を算出する。現在は未実行。
- **Codex D**：自身の許可範囲で同path/axisのC−S、有限cohort/rare不確かさとsampling provenanceを保持する。
- **Codex B**：本packetを診断metadataとして参照し、law差/未知systematicをnullのまま保持する。E[S]と標本残差の和を適格E[C]へ昇格しない。

commit/push・remote SHA/tree/主要成果物照合後、**mandatory STOP**。F0、sampler変更、実平均価格評価へ自動進行しない。
