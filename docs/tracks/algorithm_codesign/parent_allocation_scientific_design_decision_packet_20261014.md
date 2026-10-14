# Parent allocation：科学設計のDecision Packet

2026-10-14 JST。技術状態：**`TECHNICALLY_READY_FOR_SCIENTIFIC_DESIGN_REVIEW`**。比較方法・候補・precision・confidence・budget・primary・科学解釈は未採択である。**本資料は技術準備の報告と選択肢であり、科学レビュー・科学契約・実行承認ではない。mandatory STOP。**

今回閉じたのはnative/cache/guardの互換性、内部配分の成立条件、proposalの情報要件である。登録D1〜D3の配分最適化、係数・資源再評価、保存合成列のoperator再検証、新合成/compile/行列評価/shot forecastは行っていない。登録外の人工2×2／controlled 4×4演算、mock、形式的な有理数controlを使用した。

## 1. 固定入力と証拠への入口

| 資料 | 固定identity |
| --- | --- |
| 前回source／公開確認 | source `4d35c2bc4b11a61dbb38f876e826a8d56c19f0ab`、公開HEAD `a9ee6a900a8e41fdb42126bfd01d20f933964b66` |
| 前回実装・62 tests・未解決事項 | [引き継ぎ](parent_allocation_source_preparation_and_gpt_handoff_20261014.md)、[技術options v2](parent_allocation_comparison_technical_options_v2_20261014.md) |
| 先行する数学監査 | `c12558d682f6aef3cfc68e8d75082269f2282ac8`：[mean/KKT/source監査](parent_allocation_technical_audit_20261014.md) |
| GPT方針原本 | `4d80b45693a1ea7c14d7cbf9035e7358788953bd`：[原本](../../research/track_b_algorithm_direction_review_20261014.md) |
| 今回の明示指示原本 | [byte一致コピー](inputs/parent_allocation_final_technical_preparation_user_instruction_20261014.md)、11,692 bytes、SHA256 `f20c9e73fc8b09d77b4b84e39b35c1eeffc1c4086df556e6e9179c2abf41ec0b` |
| 今回の取得・静的照合 | [固定GitHub資料13 files](../../../artifacts/track_b_parent_allocation_final_technical_preparation/2026-10-14/v1/fixed_input_identity.json)、[native/cache schema監査](../../../artifacts/track_b_parent_allocation_final_technical_preparation/2026-10-14/v1/native_compatibility_audit.json) |
| 今回のsource・検証証拠 | [source manifest](../../../artifacts/track_b_parent_allocation_final_technical_preparation/2026-10-14/v1/source_manifest.json)、[tests履歴](../../../artifacts/track_b_parent_allocation_final_technical_preparation/2026-10-14/v1/focused_test_execution.json)、[保護/証拠manifest](../../../artifacts/track_b_parent_allocation_final_technical_preparation/2026-10-14/v1/evidence_manifest.json) |

指定資料と前回sourceをGitHub rawから取得し、commit blob/localのbytes・SHA256へ照合した。過去の62 tests、36合成やD1〜D3数値監査を一式再実行していない。今回の新sourceは別branchで管理し、以前のcommit・source manifest・結果・authority・消費済みmarker・closure・STOPを保持した。

## 2. Native接続で解決したこと

新しい[parent_allocation_native.py](../../../src/trottertracks/algorithm_codesign/parent_allocation_native.py)を追加した。前回の[parent_allocation.py](../../../src/trottertracks/algorithm_codesign/parent_allocation.py)はcache checkerの空sequence拒否だけを限定修正した。空文字はidentity productであり、小さい角度の正当な近似になり得る。SHA・strict error・precision等の条件は維持し、空列を自動的に成功扱いしない。旧固定commitは不変である。

| 互換性項目 | 対応と限界 |
| --- | --- |
| 旧cache schema | D1/D3はratioを外側keyとし、rowに`atan:ratio:scale:1:epsilon:...`、sequence SHA、counts、strict errorを保存する。新keyのtool等はrowだけでは得られない |
| bridge | `bound_json`でraw文書のbytes/SHAを照合し、`bridge_legacy_row`でinventory spec/hash、tool/runtime、source/result/cache/contract/監査hash、strict guard policyと保存rowをbind。変換後は**metadata-only、guard未再実行、使用可能false** |
| tool/key | tool文書からversion/tree SHA/seedに加えてoptions、package lock SHA、runtime executable SHA、guard dpsを新keyへ含める。seedだけが同じでも他設定の差を同一keyにしない |
| strict guard | 既存`rte_reallocation.numeric.strict_guard`を呼ぶshim。`RZ(theta)=exp(-i theta Z/2)`、theta=`atan(exact ratio)`、Wを保持し、phase最小化をしない。precision contextを復元する |
| angle区間 | 既存mpmath intervalとexact binary endpoint→Fraction経路を利用。登録angleは取得していない。大きいratioで端点を代用しない |
| signed sequence | pygridsynth保存文字列はwritten-product順。正ABCは回路時間C,B,A、adjointはA†,B†,C†。W/wを落とさず、価格は正/負で共通、operator errorもadjoint不変 |
| controlled端点 | 既存の新`native_plan`のS/S†/ZとCQ/parent順序を使用。pure-child/pure-parentにはR token/key/合成を要求しない。providerのexactゲート費用は0ではない |
| reference target | 旧G9 referenceは正回転・有限ratioを前提とし、そのままでは負sign/`ratio=None`端点を扱えない。`operator_target`を追加し、provider-neutralなsign/endpoint/順序をoff-domainで接続確認した |
| native価格・誤差 | `native_accounting`はguard再検証済みentryのみを消費し、各R occurrenceのdeltaを足す。old G9の一律`2e-6`を新端点・任意precisionへコピーしない。T/CX/native1Qは別座標、whole-circuit最適化ではない |

bridgeのhash照合は来歴の整合性であり、合成列の演算子正当性を再証明するものではない。hashの期待値と各receiptの真正な出所は将来の新source/contract/workerが固定Git証拠へ結び付ける必要がある。正規化rowを任意に作れば信頼できる、という設計ではない。

`recheck_entry`は別のstrict guard処理で、成功後のkey/sequence/error/dps bindingを保存する。今回、登録cacheの36+36 rowsはschema・保存metadataの読取りに限定し、bridgeへ一括変換・operator再評価していない。既存の完了監査は参照する。新workerで再検証するか、固定監査receiptをどの条件で信頼してreuseするかを結果前に明示する。現`native_accounting`の入口は再検証済みを要求するため、過去cacheの無条件使用はできない。

### Runtimeは存在し、metadataを確認できた

既存isolated Pythonは`/home/abe/Project/prt-worktrees/track-b-sp05-economics-preparation-20261006/.venv/bin/python`。Python 3.10.12、pygridsynth 2.0.0、mpmath 1.3.0。tool文書SHA `803d902a18d925a9565749fbc64422dca4979bfa6c0bd8b438b28de18d3aa984`、resolved executable SHA `b94e9b56bc1f96b18d36a0f1d14308575bb7b5960eda94ae0520f0376e95d12d`。

既存`verify_runtime`で全package versionsとpygridsynth/mpmathの`.py` tree SHAを実際に照合しPASS。合成backendをimport/callせず、登録operatorを評価していない。[runtime receipt](../../../artifacts/track_b_parent_allocation_final_technical_preparation/2026-10-14/v1/runtime_metadata_identity.json)。このruntimeを新科学契約へ採択したわけではなく、実行時にも再照合が必要である。wheel等を再インストールする必要は現状ない。将来別hostではtool文書のversion/tree/wheel SHAから同じisolated環境を用意する。

### 検証の範囲

**新規26 tests PASS、影響範囲の既存CacheControls 6 tests PASS**。初回24件の後、provider-neutral target接続2件を追加し新suiteを26件で再実行した履歴を保存した。旧62件全体は再実行していない。

mock cache/spec改ざん、projective receipt拒否、missing price、error coverage、空列identity、sign/adjointとW、context復元、controlled端点/非可換順序、固定単価KKT、root-local/globalの区別を確認した。mpmath guardを使った人工2×2 fixtureは保存科学列ではない。pygridsynth/Qiskit/NumPy/SciPy importを拒否するharnessで新suiteを実行し、合成・compile・登録matrix・N取得0。1 worker / numeric threads 1。局所focused証拠であり、immutable CI・外部再現・一般数学証明ではない。

## 3. 頂点だけでは内部配分を判別できない

理想rootは`z_i>=0, sum z_i=a, c_i=sqrt(z_i²+b_i²)`、phi=`atan2(b_i,z_i)`。平均は配分によらず同じで、質量は`L(z)>=sqrt(a²+s²)`。common-angleは質量最小であり、新配分にnormalization減少を期待する比較ではない。

固定正単価`C_i>0`、`a>0,b_i>0`のcost×moment proxyに対し

\[
S(z)=\sum_i\sqrt{C_i}\sqrt{z_i^2+b_i^2},\qquad
z_i(\lambda_K)=\frac{b_i\lambda_K}{\sqrt{C_i-\lambda_K^2}},
\quad F(\lambda_K)=\sum_i z_i(\lambda_K)=a.
\]

`0<lambda_K<sqrt(min C)`で

\[
F'(\lambda_K)=\sum_i\frac{b_i C_i}{(C_i-\lambda_K^2)^{3/2}}>0.
\]

Fは0から無限大へ増加する。SのHessianはpositive diagonal `sqrt(C_i)b_i²/(z_i²+b_i²)^(3/2)`なので、一意のstrict interior最適点を持つ。等単価ではcommon-angleになる。これは全ensembleへ同じglobal解析ISを許す固定価格proxyの結果であり、canonical有限confidenceの最適配分の式ではない。

登録外のexact controlではa=7、b=(4,3)、C=(1,9/16)、lambda_K=3/5、z=(3,4)、c=(5,5)がKKTを満たす。commonと両頂点はこの固定価格proxyでより高い。これは「頂点の失敗から内部の失敗を結論できない」ことの人工的な形式検証であり、PR/native改善の科学結果ではない。登録costを使ったlambda探索は行っていない。

**内部候補を実装へ渡す際には、数学的な最適点とexact rational候補を区別する。** lambdaやzは一般に無理数である。将来の有限候補化では、固定解像度・決定的丸め・残差配分によって非負性とexact総和を保ち、定義済みの候補を先にfreezeする。可解性・端点への丸め・失敗処理を事前に決める。これは近傍の候補であり、irrational optimumをexact実装したとは呼ばない。この変換source/探索を今回採択・実行していない。

実価格は`C_i(phi_i,epsilon,tool,seed,provider,control,lowering)`である。phiの変更が価格を変え、T数は不連続である。固定CのKKT式にcommon角のT数を代入しても、得られるのはsurrogate候補であってnative最適点ではない。candidate作成に使うproxy情報と、候補の新角度を同じ精度で実測する情報を分離する。価格probeを反復して選ぶ場合は探索・取得budgetを別途固定し、取得済み候補だけへ比較を偏らせない。

零bは純parentの線形項、零Cは境界・退化を含むので、このstrict interior式の仮定に含めない。T=0へのhidden floorや失敗後のsolver/tolerance変更は行わない。

### 合成key数と情報量

両表現のcommon点がidentity付きでreuseでき、L頂点に加えてJ個の内部候補を各表現に与えるなら、rootの新positive keyは重複除去前に**最大`2L(1+J)`**。任意M非common候補なら最大`2ML`。これはG9と同じscale1・正primitive/adjoint共有の場合のroot上限であり、実key数・非root新取得・別providerの追加basis keyはまだ数えていない。端点はexactでkey不要。旧「6件」はL=3・頂点のみの条件付き上限である。

zの取得後の係数・angle metadataはrootごとO(L)個。KKT候補の選定には少なくとも全childのproxy単価が必要で、実T価格に基づく選定には新角度の合成情報まで必要になる。global ISなら全eventの係数と価格が必要で、P5のO(L²)群生成と同じ情報量ではない。

## 4. Proposalの数学と実装条件

| Proposal | 使用情報・law | normalizer/support・費用の意味 |
| --- | --- | --- |
| canonical | digital coefficientだけ。全eventではq比例tilde_c | ideal/digital Bと実dyadic lawを区別。既存group構造を使える。sourceの固定bitsが全positive supportを保持できなければ失敗 |
| root-local cost-aware | root各atomの係数と単価。root内q比例tilde_c/sqrt(C) | top-level group qは固定。rootの条件付きcost×moment proxyの解析式であり、rest込みのglobal optimumではない |
| global cost-aware | 全positive eventの係数・正価格。q比例tilde_c/sqrt(C) | 全表のnormalizer取得、走査、sampling用tableの準備/online accessが必要。I0や少数群samplerへ無料で移送しない |
| regularized | 上記のCをC+kappaへ置換、kappa>0を明示 | zero costへの発散を避けるsampling用parameter。**charged actual costへkappaを加えない**。lambda_K（配分KKT乗数）とは別物 |

一般のpositive costで`nu*Cbar >= (sum tilde_c sqrt(C))²`、等号lawがglobal解析ISである。zero-cost eventとpositive-cost eventが混在すると、この極限はfinite full-support lawで達成されない場合がある。全cost0ならproxy0でも有限confidence taskのlaw/rangeを別に決める必要がある。現在の解析IS関数はzero costを拒否する。

root-localの注意は本質的である。root group mass gを固定した場合、global productは

\[
\left(C_{\rm rest}+g\sum_i p_i C_i\right)
\left(\nu_{\rm rest}+\sum_i\widetilde c_i^2/(g p_i)\right).
\]

rest項があるので、条件付きp比例tilde_c/sqrt(C)がこの積を最小にするとは限らない。人工有理数controlでnonzero勾配を確認した。現sourceの機能名は最適なglobal有限confidence lawの認定ではない。新たなroot-constrained optimizerは今回設計・採択していない。

finite-bit後は`q*w=tilde_c`をexactに保持し、`nu=sum tilde_c²/q`、`W=max tilde_c/q`を実lawへ戻す。probability rounding自体をmean誤差と二重計上しない。係数・angle/native誤差は別に戻す。group変更でnonroot qも変わるため、旧global moment/rangeを無条件コピーできない。proposalだけを変えてもnormalization Bは変わらず、momentとrange・費用が変わる。

有限confidenceではrange項、ceil、biasからの統計残余幅、準備費用h、複数stepでのmoment積も関わる。global解析式の最適性を、これらを含む十分Nの最適性と呼ばない。setup/online計測は未取得。root-local tableはO(L)規模だが、全group law準備や精度bitを含む実費用をO(L)だけとしない。

## 5. GPTが判断する7領域

以下は**未採択の選択肢**である。どの案も科学的に推奨・優先順位付けしていない。旧草案のinput、r、lambda、accuracy/confidence、capsを自動採択しない。

### 1. 比較方式

- **選択肢**：ordinary/closed P5に同じ自由度を与える二表現の機構比較。partial-return等を新ledgerへ追加する比較。CTS/SPS等の強対照まで含む比較。
- **確認済み**：共通root adapterはordinary/closed P5でmeanを保持する。既存D1〜D3の別ledgerは参照可能だが、新予算の競争結果ではない。既知法との関係は前回監査・文献照合を継承し、priorityを新認定しない。
- **成立条件・追加情報**：各対象に同じ配分/proposal/precision/情報access機会を与える。他表現でどのparentを対応させるかは別に定義する。CTS/SPSでは辞書・情報取得・controlled first momentの対応も必要。
- **計算量/failure**：二表現ならrootだけの変更で済む。追加方式は新adapter/coverageを増やし、比較class不整合や費用欠測がfailureになり得る。
- **公平性/科学判断**：新方式だけに配分・ISを与えない。二表現限定の差を全既知法への優位とするかどうかをCodexが決められない。

### 2. 配分候補

- **選択肢**：commonのみのcontrol。common+全頂点。common+全頂点+事前proxyに基づく有限内部候補。角度依存価格probeを使う有限探索。
- **確認済み**：commonはnorm最小。固定正価格proxyには一意の内部解があり、頂点だけではその機構を棄却できない。native利益は未取得。
- **成立条件・追加情報**：内部案にはproxy、KKT仮定、rational化規則、候補数、freeze時点が必要。probe型には取得budgetと選択/evaluationの分離が必要。
- **計算量/failure**：root key上限は§3。無理数丸め、zero-cost/zero-b退化、key上限超過、結果依存候補追加が問題となる。
- **公平性/科学判断**：両表現に同数・同情報・同規則の候補生成機会を与える。有限候補の価値と探索範囲をどこまで問いに含めるかを科学設計で決める。

### 3. Proposal

- **選択肢**：canonicalのみ。canonical+root-local cost-aware。全表を公開してcanonical+global IS。有限confidenceを目的とするproposal探索は別設計。
- **確認済み**：4つの既存proposalのsupport/weight条件と情報要件を§4で整理した。root-local解析式はrest込みの最適性を持たない。
- **成立条件・追加情報**：cost-awareには該当scopeの価格coverageが必要。globalは全event表/normalizer/生成費用を両表現へ同じように提供する。
- **計算量/failure**：group生成の小ささと全表取得量は異なる。表サイズ、missing price、dyadic support不足、range過大、setup/online費用欠測があり得る。
- **公平性/科学判断**：配分とproposalの効果をfactorとして分離する。I1の全表比較を一般I0の安価な生成と混同しない。比較に与える情報accessと最適化機会の採択が必要。

### 4. 正則化・zero-cost

- **選択肢**：zero costがあるlawでは解析ISを不適用としてcanonicalを残す。明示正kappaを共通使用する。有限候補の複数kappaを同じ機会で比較する。別のsupport/range制約付き最適化を設計する。
- **確認済み**：hidden floorはなく、正則化はproposalだけを変える。空列identityも正当なguardを満たせばT0になり得る。endpointにもCX/phase等の費用がある。
- **成立条件・追加情報**：kappaの単位、値/集合、対象scope、freeze時点を固定する。T0をCXやnative1Qの0と扱わない。
- **計算量/failure**：proposal候補はN会計・family confidence配分・table準備を増やすが、同配分内では新角度keyを増やさない。小kappaのrange発散、bits/support不足、shot cap不適格があり得る。
- **公平性/科学判断**：各方式に同じ処理機会を与え、kappaを実費用へ加算しない。正則化比較を何の研究claimに対応させるかは未決定。

### 5. Accuracy・confidence・budget

- **選択肢**：旧草案にある一step条件を新ledgerとして採択する。accuracy/confidence/反復を別に固定する。取得budget内の機構診断へ限定する。
- **確認済み**：mean target不変でも実装bias、moment/rangeは変わる。旧Taylor保証の一般導出は参照できるが、旧十分Nを新lawへ流用できない。
- **成立条件・追加情報**：総時間/fixture/provider/符号/反復、finite P5 target、primitive precision、K/H、許容bias、shot cap、失敗alpha配分を結果前に固定。K構成なら統計用alphaをX/Yの2K axesへ配り、resource failure分と合計整合を取る。複素誤差にはaxis予算の二乗和条件を満たす。
- **計算量/failure**：候補数でconfidence費用・出力量が変わる。bias残余幅<=0、証明不十分、primitive coverage不足、資源超過はtechnical/ineligible処理が必要。新契約のwall capは共通指示どおりデフォルトなし、CPU/RSS/output/log/callの具体値は未採択。
- **公平性/科学判断**：同じtaskと最大誤差budgetを各方式へ与え、unused budgetを統計へ戻す規則も固定する。point signalをshot削減に使わない。どの精度域/取得budgetに情報価値があるかは科学判断。

### 6. Primary endpoint

- **選択肢**：総Tをprimaryにする。量子資源vector/Paretoをprimaryにする。準備/読出しh>=0の領域を主対象にする。proxyだけの機構診断へ限定する。
- **確認済み**：normalization増と費用減の交換がある。canonicalのBだけ、per-event Tだけ、IS proxyだけでは有限confidence総量を決められない。
- **成立条件・追加情報**：十分Nとactual qの期待単価、native bias、workspace、準備/読出し、classical setup/online、angle取得費用を区別する。期待費用とhard/tail費用のいずれを保証するかも固定する。
- **計算量/failure**：複数座標には共通IR/cacheをreuseできるが、未取得online費用やwhole-circuit最適化の影響は保存値だけでは分からない。
- **公平性/科学判断**：T/CX/native1Qを重複加算しない。h=0を準備費用無料の実測としない。未知費用を0にしない。scalar重み・materiality・資源上の主張範囲の採択が必要。

### 7. 科学的解釈

- **選択肢**：有限候補での機構存在/不存在。固定providerでの条件付きnative利益。新method差・一般化の研究候補としての評価。
- **確認済み**：同じ理想mean、common包含、限定固定価格KKT、controlled semanticsと技術互換性。独立新規性・native優位は未確立。
- **成立条件・追加情報**：利益が配分、RA、IS、合成価格、error ledgerのどこから生じたか分離する。一般化やpriorityには強対照・別implementation/情報accessの関係が必要。
- **計算量/failure**：本runを増やさず判断できる範囲と追加取得が必要な主張を分ける。prefix成功、missing価格、頂点negative、proxy改善をmethod採択に変換しない。
- **公平性/科学判断**：頂点のみのnegativeは内部配分を排除しない。positiveも有限候補・provider・precisionの条件付きであり、分子PR/QPE全体の優位ではない。何を継続/修正/終了と読むかはGPTが明示的レビュー開始承認後に判断する。

## 6. 科学実行までの依存関係と残存課題

| 区分 | 課題と状態 |
| --- | --- |
| **A：今回解決** | 旧/new schema対応、expanded tool/options key、raw bytes/provenance照合API、guard未検証rowの拒否、strict guard/atan interval shim、W保持とsigned sequence展開、primitive occurrence誤差、endpoint/signed reference、正当な空列、runtime metadata一致、内部配分/conditional ISの形式control |
| **B：条件採択後にCodexで実装** | 採択providerへのreference/controlled実装の接続、rational内部候補化、inventory/job/worker、正規化cacheの真正な原資料binding、error ledger/production bounds、十分N、hard guards/partial失敗保存/分割、zero-computeから独立した実行launch gate |
| **C：GPTの科学判断が先** | 7領域。固定proxy/価格probeの使い方、reuseと新合成の公平規則、他方式への最適化機会、confidence/resource保証の適用範囲、primary/解釈。今回選択値なし |
| **D：別途科学実行承認** | 登録fixtureの係数/angle/key取得、保存列のoperator guard、新sequence、native評価・資源表・十分N・比較実行。新source/contractをfreezeし、authorization-only直接子、新namespace、exclusive marker、outer process/resource receiptを必要とする |

依存順序は **Cの採択 → Bの新実行source準備・技術検証 → 別途明示承認 → Dの限定取得**。Aの完了はDを認可しない。旧marker/authorization/resultの再利用・既存classification更新はない。旧D1の全native pipelineやG9 referenceへ新端点を無加工で渡して実行する構成は使用しない。

性能面では独立keyのparallel化とimmutable cache共有を検討できるが、必要数・aggregate RSS・CPU/call/出力上限を採択してから構成する。今回の小さなfocused testsは1 processで十分であり、speedupの主張はしない。failureではtolerance/precision/候補を自動変更せず、missingをfreeにしない。新failure evidenceを保存し、旧STOPを上書きしない。

## 7. 今回の完了点

科学条件なしで解消可能なnative/cache/guardの互換性を実装し、登録外検証とruntime/read-only証拠を揃えた。さらに、頂点限定の解釈限界、内部候補のrational化・情報取得、root-local/global/finite-confidence最適性の境界を明確にした。7領域の判断に必要な選択肢・取得量・failure・公平性を本資料へまとめた。

登録比較の価格・誤差・資源優位は取得しておらず、科学的な採否・新規性・研究GO/STOPをCodexが決定したものではない。重要科学レビューはユーザーの開始承認待ち、科学実行も別途明示承認待ち。**mandatory STOP。**

> 後続の[近接先行研究との厳密対応調査](parent_allocation_prior_art_correspondence_20261014.md)：Euler/ISの既知部分、SPSのprojected/signed embedding、限定class差を整理。上記7領域の選択値・STOPを変更していない。
