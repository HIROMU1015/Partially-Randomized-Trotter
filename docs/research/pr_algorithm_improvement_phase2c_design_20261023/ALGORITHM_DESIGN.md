# 主設計：残差fragment別のPauli／frame辞書選択

2026-10-23 JST / Phase 2C / 設計仕様。実装・計算・新証明は未実施。

## 1. 目的、位置付け、変更を一つに限定する

PRで使う同じ残差fragmentについて、frame-conjugated Z/ZZのまま実装するか、固定物理register上のPauliへ展開して実装するかを、normalizationとrun共有後の構造価格を含めて選ぶ。主変更はfragmentごとの二値の**残差辞書選択**だけ。新軌道最適化、再DF、rank変更、決定論／残差の再分割、次数・microstep最適化、新sampling分布、native角度配分、routingは追加しない。

状態は `KNOWN_METHOD_APPLICATION_CANDIDATE`。確認した原PRには両辞書の着想があり、既存toyにも異種primitiveを扱える接口がある。本書が定義する選択手順を、そのまま独立新原理として採択しない。`INDEPENDENT_METHOD_DELTA_UNESTABLISHED` であり、現時点で独立新法として擁護できる段階ではない（`NO_DEFENSIBLE_NEW_METHOD_YET`）。一方、全手順が既存文献の同一算法と一致したという証明も行っていない。構成差の比較は[別紙](METHOD_DELTA_AND_PRIOR_ART.md)に明記する。

## 2. 固定する物理問題

保存入力が定める演算子を

\[
H_{\rm ref}=cI+H_D+R,\qquad R=\sum_{\ell=1}^{M}F_\ell
\]

とする。元の量子化学modelと保存DF入力の差があれば、`ΔH_input`として既存のまま残す。本案が理想的な係数代数で追加する表現差は0。有限精度の追加差 `ΔH_dictionary` は別に記録し、既存DF誤差やPF誤差へ吸収しない。

各残差fragmentは既存入力で

\[
F_\ell=\sum_j a_{\ell j}Q_{\ell j},\quad
Q_{\ell j}=B_\ell Z_{S_{\ell j}}B_\ell^\dagger,
\quad |S_{\ell j}|\in\{0,1,2\}
\]

と与えられる。`|S|=0`は既存identity policyが残差に残したIだけを意味する。`c`へ抽出済みのIを再投入せず、残差内Iを新たに外へ出さない。これは有限Taylor targetの不要な変更を防ぐ条件である。`B`の向きは本書の定義であり、SOURCEのforward/inverse名と機械的に同一視しない。

固定項目は、物理geometry・AO basis・active space・電子sector・qubit mappingと順序・scalar・状態の物理表現、保存DF rankと切捨て方針、決定論block列と`L_D`、S2の順序、時間`T`、反復`q`と`δ=T/q`、残差occurrence内microstep`r`と偶数cutoff`K`、測定taskと誤差／confidence規約である。固定Hの辞書を替えるので、状態を無料の別frameへ回す操作はない。AO basis変更は別の物理model比較である。

同じ物理sectorをtargetにしていても、展開後の各Pauliが粒子数を保存するとは限らない。trajectory中にsector外へ出て戻ることを許すfull registerでのHadamard meanが本案の基本taskであり、途中のsector投影・postselectionを追加しない。sectorだけに符号化した実行やsector内matrix-free simulationを使う場合、全atomがその符号空間を保つことを別途確認できなければP候補は不適格（PO-09）。固定sectorの行列だけでprimitiveのunitarityを検証しない。

## 3. 必要入力と取得費

| 入力 | 取得方法・不要な情報 |
|---|---|
| 固定H identity、上のfragment係数・frame・scalar方針 | 既存保存inputとSOURCEから取得。新分子・軌道・DF生成は不要 |
| 物理registerでのfragmentの二次／四次fermion係数、またはそれを得る保存factorと軌道行列 | `Z=I-2n`と保存軌道行列の係数代数で得る。Fock空間の全Pauli traceやexact stateを要求しない |
| PR設定 `T,q,r,K`、基準実装の順序とseed規約 | 既存契約をそのまま固定。新しいr/K探索はしない |
| full frameおよび既存singleton support completionの回路recipe | 公開SOURCE／保存recipe。数値compilerを呼んだ価格ではなく、明示recipeのliteral countに限る |
| 主資源座標 `μ`、prep/readout recipe | 一つの評価で一座標を先に固定。既定候補はRZ count。CX・depth・Tを根拠なく重み加算しない |
| 共通の複素信号統計allowance `η_stat>0`、失敗率 `α`、採用済みbias条件 | 真のE0、gap、ground state、最適誤差oracleを使わない。共通biasの資格情報がなければsignal同精度適格性は未判定とする |
| 係数のprecision／zero判定規約 | exact symbolic入力または検証可能な精度表現。丸めた係数をexactと呼ばない |

入力不足を新しい科学計算で自動補完しない。主価格が作れない場合、既存構成を返し`UNPRICED_FALLBACK`とする。bias情報だけが不足する場合は構造案まで返せるが、同精度選択済みとは表示しない。

## 4. 出力とデータ構造

| record | 内容 |
|---|---|
| `TaskIdentity` | H／sector／register／state／scalar policy／元誤差／PR設定／recipe版のidentity |
| `FragmentAlternatives[ℓ]` | S辞書、利用可能ならP辞書、係数代数の導出metadata、精度・zero判定、不能理由 |
| `Atom` | ID、実係数、符号、種類 `FRAME_Z` / `PHYSICAL_PAULI` / `IDENTITY`、basis keyとhash、supportまたはPauli word、由来fragment集合、制御recipe |
| `MixedTailView` | `RTETailLike`相当のtail_id/hash、λ、component列、別表のtyped dispatch情報。Pauliを偽のDF fragmentとしない |
| `SelectionRecord` | 二値列ξ、候補毎のλ/B/価格・適格性、訪問順、採用・棄却・未確定理由、評価回数、cache依存 |
| `ExecutionPlan` | 固定D列、mixed residual routing、既存run共有・phase集約・境界融合の適用範囲、ancilla・wrapper、precision義務 |

出力は、同じ残差Rを表す辞書と既存samplerに渡すmetadataであり、新しい実行許可ではない。

## 5. 二つの辞書をどう作るか

Sは元の`a_lj, B_l, Z_support`を使う。Pは**同じF_l全体**の固定registerでのPauli係数map `b_l[P]`である。保存factorから二次／四次fermion多項式を展開し、固定JW変換の積・符号・同一wordの加算を行う。元入力がこの低次数構造を持たない場合、汎用dense展開へ逃げずP候補を利用不可とする。既存toyの`padd/pmul/annihilator_paulis/orbital_paulis`は実装参考であり、現在の浮動小数点helperを大規模なexact証明器として使えるとはしていない。

二値列ξに対し、`ξ_l=0`ならS、`ξ_l=1`ならPを選ぶ。Pへ替えたfragment群の同一Pauli wordは**全体で加算してから**abs・符号・確率を決める。Sのcomponentは元の区別を維持し、異なるframeとの非自明な同値性探索やS/P間の追加collectはしない。P側のIは元のidentity policyの置き場所を維持する。全ての非零係数を残し、新しいthresholdは導入しない。

exact zeroだけを削除する。intervalが0をまたぐ等、符号・zero・正確なsamplerが決まらないfragmentのPは不適格とする。物理の近似入力なら有限精度実装の誤差仕様が必要で、勝手な小係数切捨てはしない。`λ=0`を認証できる場合のみ既存のdeterministic-only経路へ接続し、0で割らない。

## 6. 既存finite-RTEへの接続

collect後を `R=Σ_i a_i Q_i`、`λ=Σ_i |a_i|`、`p_i=|a_i|/λ`、`P_i=sgn(a_i)Q_i`とする。各occurrence時間はδ、microstep時間は`h=δ/r`、`τ=λh`。既存 `finite_rte_distribution` / `make_rte_config` のpaired even lawを使う。

ここでpを定義するのはλ>0の場合だけ。λ=0を認証できた候補はRTEを呼ばず、残差の価格を0、`B_tot=1`とするdeterministic-only構成へ送る。未認証の相殺をこの特例に入れない。

\[
w_n={|\tau|^n\over n!}\sqrt{1+(\tau/(n+1))^2},\quad
B=\sum_{n=0,2,\dots,K}w_n,\quad \Pr(n)=w_n/B,
\quad\theta_n=\arctan(\tau/(n+1)).
\]

次数nごとにcomponentを既存の独立law `p`で`n+1`個引く。SOURCEではsampled indexの先頭がrotation、残りがproductsであり、**circuit時間順はproductsを記録順に適用し、最後がchild rotation**。位相`(-1)^(n/2)`、係数符号、`event_coefficient`と`event_probability`をそのまま維持する。変更後のλで次数law・角度・Bを必ず再構成し、旧分布を流用しない。

条件付きで期待する接続は、理想係数・正しい制御回路に対して一microstepの補正meanが `T_{K+1}(-ihR)` になること。Rとh/Kが固定なので目標の有限多項式は辞書によらない。これは既存RTEの性質の適用条件であり、本案のmixed adapterについて今回証明・検証した成果ではない（PO-03）。粗いtail boundはλ依存で変わるので、元のcertificateまで同じと決め付けない。独立event・occurrenceの通常構成なら全体補正因子は`B_tot=B^(rq)`。これは係数であり、channel一致や真の `exp(-iTH)` との一致を意味しない。

## 7. 選択に使う価格を具体的に定義する

### 7.1 統計と1-shotを分ける

一時刻の複素Hadamard信号に対し、各axisの補正outcomeの範囲は `[-B_tot,B_tot]`。fresh IID quantum shotと既存Hoeffding policyを条件として、axis allowanceを`η_stat/√2`、axis失敗率を`α/2`に固定する。選択用shot式は既存 `_hoeffding_shots` と同じ関係

\[
N_X=N_Y=\left\lceil {4B_{\rm tot}^2\over\eta_{\rm stat}^2}
\log{4\over\alpha}\right\rceil
\]

を使う。新しいvariance最適samplerではない。この式はfinite corrected meanの複素信号統計用であり、energy/RPE精度保証ではない。別の採用済みshot policyを使う設計改訂なら全armで同時に変更する。未知のstate依存varianceを無料で入力しない。

### 7.2 構造価格Φ：既存run共有を与えたliteral recipeの期待count

compilerやnative合成を呼ばず、同一recipe版で数えられるRZ等のcountを選択に使う。各nonidentity atomにはsource basis key、full basisのentry+exit価格`d_full(b)`、既存support completionによるsingleton価格`d_single(i)`、中央のproduct／rotation価格`z(i,role,n)`を持たせる。Pauli側は既知の局所Clifford＋parity compute/uncompute＋中央制御操作、frame側は既存Gaussian recipe＋Z/ZZ中央操作。Pauli局所basisの同一keyにも同じ共有規則を許す。

literal planでは、同一source basisの連続runがsingletonなら既存`support_run_le_1`、長さ2以上ならfull frameを一回だけ出入りする。productは制御Pauli、rotationは制御指数である。identityはframeを必要とせずphase項へ送る。価格model上はidentityをrunの区切りとして扱い、その保守的な区切りを全armに共通にする。実builderがさらに相殺できることは妨げない。

全trajectoryを列挙せず、次の有限状態で順序付き期待価格を計算する設計とする。

| 状態 | 新しいnonidentity atom iへの遷移と加算価格 |
|---|---|
| `EMPTY` | `SINGLE(i,role,n)`へ。価格は保留 |
| `SINGLE(j,role',n')`、同じbasis | `MULTI(b)`へ。`d_full(b)+z(j,role',n')+z(i,role,n)` |
| `SINGLE(j,...)`、異basis | 前singletonの`d_single(j)+z(j,...)`を払い、新`SINGLE(i,...)`へ |
| `MULTI(b)`、同じbasis | `z(i,role,n)`を払い、そのまま |
| `MULTI(b)`、異basis | 新`SINGLE(i,...)`へ。前runの出入りは`d_full`に計上済み |
| flush／identity／固定D境界 | pending singletonがあれば閉じ、`EMPTY`へ。identityの中央phase価格は別加算 |

確率質量πと価格の非正規化moment vを保持し、各遷移` s→s' `をcomponent確率p_iで伝搬する。

\[
\pi'_{s'}\mathrel{+}=\pi_s p_i,\qquad
v'_{s'}\mathrel{+}=p_i\{v_s+\pi_s c_\mu(s,i,role,n)\}.
\]

各microstepで次数nを上の正確な`w_n/B`で分岐し、n個のproduct、その後1個のrotationを伝搬してmergeする。microstep境界ではbasis状態を保ち、固定D境界でflushする。q回のoccurrenceと、境界half-step融合済みの固定D recipe、X/Y wrapper、prep/readoutを加える。phaseは各event／identity／負product／scalarの**潜在的phase寄与ごと**のliteral reserveを別に数え、phaseが必ず消えるとはしない。実回路では既存phase集約を全armへ適用する。

このΦは上記の限定されたliteral planの期待countである。異basis相対Gaussian再合成、Pauli cluster最適化、角度依存のcompiler簡約、phase全体の相殺、DとRの境界の追加相殺まで含めた **actual compiled costの予測値・上界ではない**。それらを対照から禁止もしない。状態数はpending atom×role×cutoffとbasis数に依存し、denseな遷移では各位置でatom数との積を払う。係数代数、状態数、数値精度、q/r/K、cacheを含む実workを記録する。理想の安いoracleとはしない（PO-05）。

主scoreは `J_mu(ξ)=N_X(ξ) Φ_X,mu(ξ)+N_Y(ξ) Φ_Y,mu(ξ)`。Φには選択されたprep/readoutを含める。省略する診断では全arm共通に明記し、実総費用と呼ばない。古典時間・memory・保存量は別座標で、秒を根拠なくRZへ換算しない。score区間が必要なprecisionでは区間を出し、非重複の改善だけを採る。

## 8. 選択規則と擬似コード

Pが使えるfragmentの集合Aを確定する。二つの初期点は全Sと、Aだけ全P。各点からstable fragment ID順に**一回だけ**反転を検討する。反転ごとに全体collect、λ、次数law、価格を再評価する。局所の単価差だけで採用しない。各runの元初期点も候補に残す。

```text
design(task, frozen_fragments, recipes, precision_policy):
    validate identities, scalar policy, fixed D/q/r/K, price/task inputs
    alternatives = [S_l and, if certifiable, exact Pauli map P_l]
    A = fragment IDs with an eligible P_l
    starts = [all-S, P-on-A-and-S-elsewhere]  # duplicate endpoint is evaluated once
    records = []
    for start in starts:
        current = evaluate_whole_candidate(start)
        retain start in records
        if current is ineligible: continue
        for l in A in stable-ID order:
            trial = flip only bit l of current
            candidate = evaluate_whole_candidate(trial)
            # collect across P fragments BEFORE abs, sign, p, lambda
            # recompute canonical RTE law and finite-state literal price
            record eligibility, work, score interval and reason
            if eligible(candidate) and upper(J(candidate)) < lower(J(current)):
                current = candidate
            # ties or overlapping intervals retain current; no precision oracle
        retain current in records
    champion = eligible all-S endpoint, if priced
    visit other retained endpoints in stable lexicographic bit order
    replace champion only if upper(J(endpoint)) < lower(J(champion))
    unresolved overlap or tie => retain champion
    # if all-S is unpriced, first eligible endpoint is provisional champion;
    # it does not certify an improvement over all-S
    if none priced: return frozen baseline with UNPRICED_FALLBACK
    emit dictionary, canonical sampler metadata, typed circuit plan and obligations
```

初期点を含めwhole-candidate評価は高々`2+2|A|`回、同一bit列はcache可能。これは本手順の定義であり、人工的なCPU／wall上限ではない。大域最適性、局所極小の到達、二回目のpassの暗黙追加は主張しない。cache keyはtask・係数／辞書・bit列・scalar・λ・角度／q/r/K・recipe・precisionを含む。既知対照へも同じ係数mapとcacheを渡し、取得費を別に数える。

## 9. actual circuitへ接続する計画

`FRAME_Z`は時間順に `B† → controlled central Z/ZZ → B`、`PHYSICAL_PAULI`は局所Cliffordで対角化し、parity compute、中央制御操作、parity uncompute、局所Cliffordのactual adjoint。basis／parityは非制御でよいが中央と相対位相は制御する。productの負符号、identity rotation、paired event phase、固定scalar phaseをancillaのrelative phaseとして保存する。

同じbasisの隣接相殺・support/full policy・phase集約、D境界同一blockのhalf-step融合を両armに与える。非可換のproductやrotationをbasis別にsortしない。角度を逆にするだけでadjointを作らず、ゲート列の逆順・各gateのadjointを使う。

既存`QiskitDFRTEEventSequenceCircuitBuilder`の中央dispatcherはZ/ZZ support長1/2を要求する。長いPauli wordを偽のDF supportとして渡せない。将来はtyped routerと任意Pauli branchを用意する必要がある。toy `Primitive.apply/apply_pauli`は参考になるが、`PRConstruction.tail()`のdense検証をproduction経路へ移植しない。既存ライブラリにこのmixed production adapterが完成しているとはしていない。

## 10. 不適格、fallback、停止

Pを得られないfragmentはSのまま。追加ΔHの予算・符号・位相・identityの扱いが未決なら該当候補を除外する。価格やconfidenceがoverflow／不確定なら0円とせず不適格または構造案のみ。計算量・memory見積りが入力に不適なら既存構成を返す。近似展開、少数trajectoryへの無断置換、未知のcompiler oracleへの変更はしない。

新しい量子操作列が既存最適化後に消える、強対照と生成手順・取得費・保証まで同じ、または情報取得が節約を明白に上回る場合は、新算法路線を縮小する。単に全Pauliまたは全Sが選ばれる場合も正当な出力であり、mixed成功例を作るための条件変更はしない。

## 11. 未解決事項と次の判定

同一Hの係数導出、有限精度、mixed first moment、controlled phase、価格DPの定義との一致、統計の適用条件、既知法との方法差、sectorと実行空間の整合は[PO一覧](SEMANTICS_AND_PROOF_OBLIGATIONS.md)に残す。今回これらの証明や科学検証は行わない。

最初の判定はF1：小さい同一Hにおいて、既存共有後に辞書／制御列の差が残り、近接した既知構成との違いを説明できるか。F2はその後の条件付き案である。既知法適用として有用かという問いと、独立新法かという問いを分ける。[最小計画](MINIMAL_DISCRIMINATING_VALIDATION.md)は実行未承認。**本設計の公開後にmandatory STOP。**

主要な固定一次SOURCE：`C-SAMPLER`、`C-TAIL`、`C-CONSTRUCT`、`C-SHOTS`、`EV-0028`、`P2-BUILD`、`P2-REPEAT`。各commit・path・identityと読解範囲は[SOURCE_EVIDENCE_INDEX.json](SOURCE_EVIDENCE_INDEX.json)を参照。PRで残差Pauli化と異軌道基底のinvolutionを使う着想は[原PR v1 §VII.B](https://arxiv.org/pdf/2503.05647v1)に既出である。
