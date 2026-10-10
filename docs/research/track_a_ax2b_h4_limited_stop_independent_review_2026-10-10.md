# Track A：H4限定検証・時間上限STOP後の独立科学レビュー

- **作成日**：2026-10-10 JST
- **文書区分**：ユーザーの開始承認を受けて実施したGPT独立科学レビューの記録
- **対象研究**：Partially Randomized Trotter（PR）Track A、AX-2B
- **対象報告**：`track_a_ax2b_h4_limited_execution_v2.md`
- **対象result commit**：`a87cf25548a3b93262027ce780317872d7c4e883`
- **実行science source commit**：`b228f2307f5fea77f066ee13e11ba6d2d8b9bea7`
- **対象branch**：`track-a-ax2b-h4-post-review-20261010`
- **レビュー判断**：**研究方針を維持し、完了6 cellを保存したまま、計算の重複を減らして四次PF 2 cellとexplicit event/control 4群を優先補完する。**
- **実行認可との区別**：本書は研究上の判断記録であり、新しいH4/H6入力生成・科学計算・launchを認可するauthorizationではない。

> **要旨**：今回のSTOPは、保存証拠上、数値的不一致ではなくcorrectness phaseの時間上限による未完了である。独立occupation構成、実stage時間のprimitive検査、6 cellのMP80/120照合には実質的な進展がある。一方、四次PFの全経路と代表event/control接続は残っている。全8 cellの再実行、精度の一律引上げ、無条件の時間延長を先に行うのではなく、同一行列指数・多項式・参照の反復生成を除いた補完計画をCodexに任せる。厳密な総数値誤差certificateをH6技術pilotの一律前提にはしないが、H6への自動進行もしない。

---

## 1. レビューの問い・範囲・証拠区分

今回の目的は、時間上限に達した実行を単に再起動するかではなく、次の判断を行うことである。

1. 完了した検証は、前回指定したH4-N/A/E/Mの何を支持するか。
2. 未完了部分は本当に必要か。必要なら、同じ科学的意味を保ったままどう補完するか。
3. 高精度照合の計算負荷は、研究の目的に対して適切か。
4. H6技術pilotへ進む条件と、H6本検証へ進む条件をどう分けるか。
5. Track Aの研究方針を変える必要があるか。

本レビューは、固定commit上の報告・保存監査・代表的一次JSON・契約・実行source・独立参照sourceを読んだ**静的な科学レビュー**である。公式mpmath文書は、多倍長演算と精度管理の意味を確認するために限って参照した。PRの新規性に関する網羅的な文献調査を今回改めて実施したものではない。

以下では証拠を区別する。

- **保存事実**：公開されたrun・audit・sourceに記録されている値と処理。
- **レビュー算術**：保存値の和、比、複素数ノルム、sourceから数えた論理呼出し数。
- **科学的解釈**：その証拠が支持する範囲と、支持しない範囲。
- **提案**：次の実装・検証計画。現在の保存結果や認可を変更するものではない。

新しい分子生成、Hamiltonian/state生成、時間発展、MP行列指数、trajectory sampling、回路build/compile、モデルfit、量子shot数・総資源の再会計は実行していない。保存6 cellを再計算して独立再現したとの主張もしない。

### 1.1 資料の読取範囲と限界

主要report、saved audit、runner/backend、MP参照と比較関数、H4/H6契約、およびB3のMP120の先頭にあるsignal・分解・状態規約を直接確認した。173 sourceと195 freezeの全件照合はCodexの保存監査を参照しており、GPTが全件のraw-byte SHA-256を再計算したわけではない。

大きいB2 MP JSONはGitHub contents経由の読取が空となり、blob自体の存在と取得可能性は確認したが、長大な全traceを逐一独立解析していない。これは**未pushではなく読取インターフェース上の制約**である。本書の数値評価は、読めた保存監査のdecimal値、比較関数のsource、代表的一次JSONに限定する。全MP traceを独立に再集計したとはしない。[S01][S02][S06][S07]

---

## 2. 固定identityと実行経緯

| 役割 | commitまたは記録 |
|---|---|
| Science source | `b228f2307f5fea77f066ee13e11ba6d2d8b9bea7` |
| Seal保存 | `d2b1511cd87ee51e2a08d0f7c5025e55e920745d` |
| 別authorization・実行base | `a699a74dbf7a10a4e43c32ed6b6d3cb9d402a708` |
| 結果・保存監査公開 | `a87cf25548a3b93262027ce780317872d7c4e883` |
| 旧v5のレビュー時点 | `d6510db9326e9335bedd03d0d07c490561e23112` |
| Snapshot SHA-256 | `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a` |
| Coverage canonical SHA-256 | `eacf9dec340e9b356090527597cfc1a277775350f050d5b50e2bb26e3c5e9609` |

報告ファイル名は`execution_v2`だが、本文はcoverage修正後のv3 sourceによる一回実行を記述している。ファイル名、source version、launch directoryのversionを同一視しない。[S01][S02]

旧v5は、8 correctness・28 wrapperの技術pilotを完了した別の実行である。今回の限定検証は、その後に追加した独立occupation構成、多倍長照合、実stage、代表event/control接続の検査である。したがって、**今回6/8でSTOPしたことは旧v5の完了を取り消さず、旧v5の完了は今回の欠測を埋めない。**[S01][S12]

### 2.1 今回のtarget

- 線形H4、隣接1.00 Å、STO-3G。
- legacy DF rank12、generation-prefix 0/6/12。
- 8 system qubits、Nα=Nβ=2、sector dimension36。
- T=0.8、q=1/4、B2/B3はR=8、r=2、K=2/4/6。
- targetは、保存binary64 DF係数が定める数学的Hamiltonianと、保存vectorを数学的に正規化した指定state。

元の連続的な化学Hamiltonian、切断前の積分Hamiltonian、厳密な基底状態をtargetへ黙って置き換えない。MP計算で精度を上げても、元の化学入力そのものの精度が80桁・120桁になったわけではない。[S08]

---

## 3. STOPの意味と、保存結果の採否

### 3.1 親terminalが示すこと

```text
status = H4_LIMITED_STOP
reason = PHASE_WALL_CAP:correctness
wall_seconds = 1802.0276073073037
worker_exit_code = -9
worker_terminal = null
retry = false
N = null
G = null
mandatory_stop = true
next_stage_authorized = false
```

保存された親terminalに基づき、今回の終了は**correctness phaseの時間上限による停止**と扱う。worker terminalがないので、未完了区間のworker自身のreasonや最終call counterは分からない。[S03]

これを次のいずれにも読み替えない。

- 分子計算で数値的不一致が確認された。
- 四次PFまたはPRの科学的仮説が反証された。
- H4限定検証がすべて成功した。
- 未完了cellはまだ一切実行されていなかった。
- workerの最後の処理内容や計算回数を完全に把握できた。

**保存結果が示すのは、完了した部分と、記録されなかった残りがあるということ**である。時間上限STOPそれ自体は、研究方針を変更する根拠にならない。

### 3.2 部分結果は維持する

完了recordをすべて無効とする理由は、今回確認した証拠にはない。6 cellの結果は、当該source・target・規則による**部分的な技術・経験的検証**として採用する。全8 cellの完了という主張だけを保留する。

この扱いでは、同じ6 cellを「今度はrun全体が完走するようにするため」だけに全面再計算する必要はない。新sourceを使う補完では、新旧の対応を確認する限定的な回帰検査と、異なる実行を統合する証拠台帳が必要になる。[S01][S02]

---

## 4. 完了coverageと、未完了項目

| Cell | Correctness | MP80/120 | 保存cell wall（秒） |
|---|---|---|---:|
| H4_B1_S2_q1 | 完了 | 両方保存 | 79.0476 |
| H4_B1_S2_q4 | 完了 | 両方保存 | 286.7813 |
| H4_B0_q4 | 完了 | 両方保存 | 169.8544 |
| H4_B2_K2 | 完了 | 両方保存 | 551.1064 |
| H4_B2_K4 | 完了 | 両方保存 | 558.6321 |
| H4_B3_K6 | 完了 | 両方保存 | 77.8716 |
| H4_B1_S4_q1 | 未完了・欠測 | 両方欠測 | 未確定 |
| H4_B1_S4_q4 | 未完了・欠測 | 両方欠測 | 未確定 |

出典は保存監査。cell wallは、そのcellのnative作用、MP、比較、書出し等を含む値であり、純粋なMP行列指数の時間ではない。[S02]

代表event/controlは次の4群がすべて未完了である。

| 対象 | Order |
|---|---:|
| H4_B2_K2 | 0 |
| H4_B2_K2 | 2 |
| H4_B3_K6 | 0 |
| H4_B3_K6 | 2 |

未保存なのはcorrectness/MP 6ファイル、explicit event 4ファイル、`phase_wrapper_cost.json`、`worker_terminal.json`。存在しない記録を再構成して原実行の結果に見せてはいけない。[S01][S02]

---

## 5. H4-N：独立参照と状態規約の評価

### 5.1 独立occupation構成の追加には意味がある

旧v5のfull/sector照合は共通のDF作用実装へ依存する部分があった。今回は、occupationとfermionic符号による別の構成を用い、36-columnのsector作用を照合している。保存された列差指標は`1.832288481395828e-15`である。[S01][S04]

sourceでは`occupation_column()`がone-bodyをoccupation上で作用させ、各DF成分についてone-body作用を二回行って二乗項を構成している。主たる`df_linear_operator`とは異なるコード経路である。これは共通実装への依存を減らす。ただし、同じ保存係数とsectorを使うことは意図された共通条件であり、「すべて独立した化学計算」ではない。[S04][S11]

### 5.2 数学的正規化の明示は妥当

MP側は保存vectorをexactなbinary64比として読み、MPでnormを計算し、初期stateだけを正規化している。途中のfinite平均stateは正規化し直さない。

B3 MP120では保存state norm beforeが`1.00000000000000008379...`と記録されている。この小さい差を無視して「もともと数学的にnorm=1」と扱わず、targetの状態規約を揃えている点を評価する。[S07]

これは基底状態の証明ではない。指定されたstateの信号を正しく比較することと、stateが厳密なground stateであることを分ける。

### 5.3 参照差の読み方

保存監査にあるbinary64 referenceとMPの差は、全完了cellで約`2.0014830212433605e-16`である。同じreferenceを繰り返し比較した値なので、6個の独立したHamiltonian検証として数えない。[S02]

また、reference差の簡易gateではMP値をbinary64へ変換して比較している。これはbinary64実装との一致を調べる用途には合うが、その値自体を120桁の差の測定値と表現してはいけない。別途、MP80/120の比較はMPのまま行われている。[S04][S06]

**判断：H4-Nは経験的な数値・構成の検証として大きく進展した。総誤差certificateや全scopeの保証は未達成である。**

---

## 6. H4-A：実stage時間・高精度比較・非unitary作用

### 6.1 179は時刻の個数ではない

保存されたprimitive検査は、**primitive IDと実binary64時刻の組179件、各3 probe、計537作用**である。probeは保存stateとsectorのfirst/last basis column。最大差は`1.7636654707726433e-15`と報告されている。[S01][S08]

sourceの検査は8登録cellのscheduleから実時間集合を作り、S4の負時間やcontrol側の時間、H4-E用のmicrostep half/undoを含む。したがって、S4 full-cell記録が欠測でも、S4に関係するlocal primitive時間が全く未検証ということではない。[S04][S08]

ただし、local primitiveが代表probeで一致することと、長いS4合成列の最終state・MP比較が完了することは違う。**179組の検査を、未完了のS4 q1/q4のend-to-end検証に置き換えない。**

### 6.2 MP80/120の比較は適切に高精度で行われている

`compare_mp_records()`は120桁のcontextでdecimal文字列を読み、MP複素数の差を計算してdecimal文字列として保存する。`compare_stages()`もMP contextでstate差を計算し、nativeとMPの対応stageを照合している。[S06]

| Cell | 保存corrected信号のMP80/120差 |
|---|---:|
| B1 S2 q1 | 約7.05×10^-81 |
| B1 S2 q4 | 約5.68×10^-81 |
| B0 q4 | 約1.25×10^-80 |
| B2 K2 | 約6.51×10^-81 |
| B2 K4 | 約1.75×10^-80 |
| B3 K6 | 約1.37×10^-81 |

reference、raw、exact-tail等の他信号も保存auditにある。上表はcorrected欄のみの転記であり、全中間stageの最大MP80/120差を示した表ではない。[S02]

この一致は、固定したMP実装の精度変更に対する安定性を強く支持する。しかし、**80桁または120桁が数学的に保証されたという結果ではない**。MP演算のworking precisionと、対象関数の誤差保証は別である。mpmath公式文書も、個々の数のaccuracyを自動追跡するものではないことを説明している。[E01][E02]

### 6.3 高精度oracleが確認する対象を限定する

MP側は、保存DF係数と実行時に用いたbinary64時刻をinteger-ratioで持ち上げて計算する。有限tailのτも実行時binary64値を持ち上げている。[S05]

したがって、これは次の検証である。

> 指定された保存Hamiltonian・正規化state・実行時間列に対して、native/binary64経路と独立occupation＋MP経路が一致するか。

一方、完全に独立したPF係数生成器で次数条件を証明したこと、非丸めの理想Yoshida係数に対する厳密誤差を認定したこと、元の化学Hamiltonianとの差を全て含めたことではない。共通scheduleへの依存は契約にも明記されている。[S08]

### 6.4 Hornerと前進Taylorは別経路である

native側はfinite Taylor numeratorをHornerでstateへ作用させる。MP側は行列の前進Taylor和を構成する。raw側はMPで独立評価したnormalizationで割り、corrected/raw/exact-tailを別経路で計算している。[S05]

これは有限多項式評価の実装独立性として有用である。一方、同じfinite-RTE定義を共有するので、その定義と確率的estimatorの一致はH4-E等の別検証で扱う。

---

## 7. 今回、有限RTEの小さい差をどこまで読めるか

B3 MP120一次JSONには、finite差とouter-PF差が高精度文字列で保存されている。[S07]

保存値のノルムを本レビューで算術計算すると、

\[
|\Delta z_{\mathrm{finite}}|\approx7.03327\times10^{-13},\qquad
|\Delta z_{\mathrm{total}}|\approx3.687572\times10^{-4}.
\]

その比は約`1.91×10^-9`である。保存native corrected信号とMP120値の差は、表示されたnative binary64値を用いた算術で約`4.54×10^-16`である。

**この指定B3 cellでは、finite成分がtotal差より桁違いに小さいという説明を、以前より強い経験的根拠で支持できる。** 旧v5では小さいfinite差がbinary64差分だけだったが、今回は別precision・別構成の証拠が追加された。

ただし、次を主張しない。

- すべてのB3または他サイズでK=6が十分である。
- この1点のfinite差が厳密上界である。
- 小さいfinite差によってB3の測定込み費用が小さくなる。
- MP80/120一致によりprobability/phase/estimator全体も認定された。

B3のlarge normalizationという別問題は残る。今回のrunはshot会計をしておらず、N/Gはnullのまま保つ。以上の算術は、新しい時間発展計算や資源順位の再会計ではない。[S02][S03][S07]

---

## 8. 総数値誤差uの扱い：検証の目的を過大化しない

### 8.1 今回の証拠を無価値にしない

厳密なu_boundがまだないことを理由に、6 cellのMP・独立構成・stage一致をすべて未検証扱いにするのは不適切である。経験的な数値検証として有効であり、そのscopeを明示して研究に使える。

同時に、差が小さかったことだけで`numerical_allowance_certified=true`へ変更してもいけない。以下を分ける。

| 区分 | 意味 | 現状 |
|---|---|---|
| 技術的整合性 | 仕様に対する実装・構成・phase・basis等の検査 | 登録部分で進展 |
| 経験的数値検証 | 別構成・別精度の比較が安定 | 完了6 cellで支持 |
| 条件付き資源会計 | 明示したu仮定の下でN/Gを評価 | 今回未実行 |
| 保証付き誤差上界 | 数値不確かさを包含する上界 | 未認定 |

### 8.2 残りの誤差予算に対して管理する

将来の会計で、軸aのbiasを\(\hat b_a\)、数値不確かさを\(u_a\)とするなら、

\[
h_a=\epsilon_{\mathrm{sig}}/\sqrt 2-\hat b_a-u_a
\]

を区別する。既定のHoeffding会計では丸めを除いて\(N_a\propto h_a^{-2}\)なので、

\[
\frac{\partial \log N_a}{\partial \hat b_a}=\frac{2}{h_a}
\]

となる。誤差予算の境界に近ければ、小さいuでも費用に大きく影響する。逆に、必要な判断精度から桁違いに小さい数値差が安定しているとき、さらに一律200桁・300桁へ上げても、研究判断に役立つとは限らない。

**推奨**：新しいu-aware会計では、経験的なuの根拠とheadroom感度を明記する。reference差に根拠のない定数を掛けただけのものを保証としない。今回のrunのN/G=nullと認定falseは書き換えない。[S08][S09]

### 8.3 非unitary作用のnormは正しい種類を用いる

一般の誤差伝播は、例えば

\[
e_j\le \|A_j\|_2e_{j-1}+\delta_j
\]

という形で考える。\(\delta_j\)は定義した入力に対するlocal errorであり、累積state差をそのままlocal errorとして再加算しない。

保存MP traceには`operator_frobenius_norm`がある。36次元のunitaryならFrobenius normは\(\sqrt{36}=6\)で、作用素2-normは1である。実際にB3のunitary stageで6が保存されている。[S07]

**その6を実際の誤差増幅率と解釈して全stageで掛けるべきではない。** Frobenius normは上界として使えるが、unitaryに対して極端に保守的になる。unitaryは数学的な1を使い、その実装誤差は別に扱う。finite多項式の非unitary stageには、その作用に適したnormまたは証拠水準を明示したstate-specific診断が必要である。現在のsourceが誤って6を増幅率として使っていると認定したものではなく、今後のu会計への注意点である。

### 8.4 厳密certificateを全研究の前提にしない

最終目的はPRの比較可能な資源評価である。任意の全Hilbert空間・全時刻・全演算に対する厳密数値certificateの開発へ、研究主題を自動的に変更しない。

H6技術pilotは、数値guardとscopeを明記してN/Gをnullにする契約なら、総u未認定でも実施を検討できる。H6本検証の資源結論には、少なくとも経験的な数値根拠とheadroom感度、公平な比較、estimator接続、費用統計が必要である。この区別は前回レビューから維持する。[S09][S12]

---

## 9. 時間上限STOPのsource上の原因候補

### 9.1 単に「MPは重い」で終わらせない

`mp_cell()`はcell・dpsごとに、次を新しく作る。[S05]

- one-bodyと全DF fragmentの36次元MP行列。
- full target行列と数学的に正規化したMP state。
- 各deterministic stageの`mp.expm(-i t A_i)`。
- finite多項式行列、raw用のnormalization、exact-tail指数。
- full targetへのreference指数。

特に`evolve()`は呼び出すたびに`mp.expm()`を計算する。outer反復、forward/reverse、corrected/raw/exact-tailで同じ行列と時間が繰り返されても再利用していない。これはsourceから確認できる重複であり、実測プロファイルで割合を測った結果ではない。

### 9.2 B2の具体例

H4_B2_K2およびH4_B2_K4では、prefix6にone-bodyを加えた7 deterministic primitive、q=4、各outerのforward/reverseがある。MPはcorrected/raw/exact-tailの3経路を評価する。

各cell・各precisionにおけるdeterministic指数生成の論理回数は、

\[
3\times4\times2\times7=168
\]

である。一方、当該cell内のdeterministic時間は各primitiveの\(T/(2q)=0.1\)なので、指数生成器の種類は7個である。

**168回のstateへの作用は必要でも、同じ7個の行列指数を168回生成する必要はない。** 同じ行列・時刻・precisionの結果を再利用しても、stage順序と各stateへの作用を維持できる。[S05][S08]

この168/7という比は、演算生成の重複を示す算術であって、実行時間が24分の1になる保証ではない。MP行列積、state適用、JSON書出し、native作用なども残る。

### 9.3 保存wallから読める範囲

完了6 cellのwall合計は約1,723.2934秒。そのうちB2 K2/K4は合計約1,109.7385秒、約64.40%を占める。[S02]

B3 K6はKが高いのに約77.9秒である。Kだけで負荷が決まるわけではなく、deterministic primitive数や重複生成の構造も重要である。ただし、異なるcellには他の違いもあるため、この表だけで時間差のすべてを単一要因に因果分解しない。

worker lifetime peakは不明である。完了cellに残るRSS下限は約380.08 MiB。これはAS 8 GiBを使い切った証拠ではなく、親reasonもmemoryではなくwallである。メモリ上限を増やせば解消すると結論する根拠はない。[S01][S03]

### 9.4 旧v5のfingerprint問題とは分ける

旧v5では回路fingerprintが大きな古典費用だったが、今回はcompile契約0で、explicit回路phaseにも到達していない。したがって、**今回の時間STOPへの第一対応として、旧fingerprintをさらに最適化する根拠は弱い。** 現在の実行経路にあるMP反復生成を先に検討する。[S01][S04][S12]

---

## 10. 推奨する技術修正：科学条件は変えず、生成の重複を減らす

### 10.1 第一候補は小さい固定oracleの再利用

Codexに任せる第一案は、MP基礎行列・deterministic指数・finite多項式・exact-tail指数・referenceを、意味論が同一の範囲で再利用することである。

最小限のcell-local再利用でも、B2の指数生成の反復を大きく減らせる。さらに固定H4 inputとprecisionに対するrun-local contextを使えば、cell間で同じtargetやdeterministic指数を共有できる。ただし、どの粒度が安全かはCodexが実装・テストで決めてよい。

### 10.2 再利用の一致条件

単なるcell名や丸めた時刻文字列だけをkeyにしない。少なくとも、次の意味の違いを混ぜないことが必要である。

- 保存target/係数、sectorとbasis順序。
- primitiveまたは行列のidentity。
- 実行した時刻のexact表現と符号。
- MP working precision。80桁の結果を120桁計算へそのまま持ち上げない。
- λ、identity extraction、K、τ、normalizationとraw/correctedの役割。
- 固定source/数値backend。

新しい分子・別DF policy・別precisionに旧cacheを無条件で使わない。cache値のin-place変更を防ぎ、entry/bytes上限と解放規則を設ける。精度のcontextをまたぐ演算や文字列化にも注意する。[S05][E01]

**MP working precisionは値ごとの精度認証ではない。** 高精度contextへ切り替えただけで、低精度で作った行列の情報が回復するわけではない。精度ごとの生成・再利用を分離する。

### 10.3 保持する科学的条件

次は変更しない。

- H4 target/state、DF prefix、q/R/K、formula/control意味論。
- 登録stage順序、負時間、scalar phase、finite cutoff。
- raw/corrected/exact-tailの役割と、初期state以外は正規化しない規則。
- MP80/120の照合。
- stage照合と保存する誤差区分。
- conditional/guaranteedの区別。

「cacheで速くする」と「PF合成列をまとめて別の近似へ変える」は別である。最初は同一演算の再利用を選び、solverの変更、precision低下、stage省略、モデル改善は混ぜない。

### 10.4 独立性を失わない

MP側の行列を高速にするために、native側のprepared diagonalizationや同じcallbackを流用すると、今回確保した構成の独立性を損なう可能性がある。**独立occupation構成は維持し、その結果をMP経路内で再利用する。**

限られたsynthetic fixtureで、cacheあり/なし、正負時間、K・precision・input切替、mutable matrixの保護、stage出力一致を確認する。旧6 cellとの回帰照合は必要な代表範囲を事前に定めるが、全6 cellのフル実行を自動的に要求しない。

### 10.5 演算上限の意味を分ける

再利用により、論理stage数と実際の指数生成回数が異なる。logical action、exp生成、cache hit、MP行列積、state適用、記録量のどこにcapを掛けるかを新planに明示する。

旧scopeのsource/hashを変えずに新cacheを差し込まない。改訂source・synthetic audit・補完対象・予算を新しく固定する。過去の凍結sourceと結果はそのまま残す。

---

## 11. 四次PF 2 cellは補完すべきか

**本レビューの推奨は、q1/q4の両方を補完すること**である。

理由は、両者が以前から登録した強い決定論対照であり、現在の欠測が比較の一方に偏っているためである。S2・B2・B3の検証だけを完成とし、S4の独立MP/全stage検証を恒久的に省略すると、後のPR比較でbaseline側だけ証拠強度が低くなる。[S08]

q1は大きい正負時間を含む合成、q4は反復による累積とより長い列の検査になる。local primitive時間の検査は両者の重要な一部を支えるが、全列を置き換えない。

ただし、補完を理由に、さらにq8/q16、多数geometry、すべての高次formulaを今回追加しない。旧v5の費用結果は別scopeで保存し、今回はS4の数値・経路確認を閉じる。

この判断は、S4が数値的不具合を起こしていると予測したものではない。また、q1とq4の2点だけで四次の漸近次数を実証したと主張しない。

---

## 12. Explicit event/control 4群を優先して実行する理由

現在のコードでは`correctness()`が全cellを完了した後に`costs()`へ進み、H4_LIMITEDではそこで`explicit_estimator_probes()`を呼ぶ。今回MPで時間上限に達したため、4群は一度も完了しなかった。[S04]

これらはMP precisionの追加とは**別の誤りを検出する検査**である。

- eventのproduct順序・符号・rotation・phase。
- basisとregisterの対応。
- ordinary/directionalの両ancilla branch。
- cosine/sine Hadamard測定と信号の対応。

同じ多項式信号をさらに高精度に計算しても、個々のevent回路や測定の接続誤りを排除できない。したがって、**4群を長いMP処理の最後まで待たせず、入力・primitive等の必要条件が成立した後に独立したbounded phaseへ分けることを推奨する。**

順序やphaseの分割は、新しいplanで明示する。前提gateを削除するのではなく、何に依存する検査かを整理して、独立な検査が高価な別検査に遮られないようにする。これは成功する結果を選ぶ順序変更ではなく、検出したい不具合の種類を保ったスケジューリング改善である。

### 12.1 4群の完了が保証するもの・しないもの

4群は代表eventであり、全部のTaylor order・component列・trajectoryの確率重み付き平均を列挙するものではない。したがって、完了後も次を区別する。

| 検証 | 内容 |
|---|---|
| Finite平均計算 | 多項式operator/state作用の計算 |
| 代表event/control | 選んだunitary eventの回路・phase・測定接続 |
| 完全な期待値の接続 | 確率・event相関・phase・normalizationを含む平均と多項式の関係 |
| 測定shot則 | 各shotのfresh trajectoryなど、推定器の仮定 |
| 期待cost | 別の費用標本・統計・rare-event評価 |

最後の三つを4群の成功だけで認定しない。既存の小型完全列挙test・source上の確率定義・samplerの独立性を再利用して対応を確認し、不足は明示する。全分子trajectoryの巨大列挙や大量の量子shotシミュレーションを今回新たに要求しない。[S08][S12]

---

## 13. 補完方法と証拠の統合

今回のgrantは消費済みである。提案する補完は、停止したプロセスへの無許可resumeではなく、**未完了義務を明示した新しい補完実行**とする。[S01][S02]

### 13.1 基本構成

- 既存6 correctnessと12 MP recordをimmutableな証拠として保持する。
- 新sourceの妥当性を必要なsynthetic/限定回帰で確認する。
- S4 q1/q4のcorrectnessとMP80/120、explicit4群を補完対象とする。
- input/target/state・time・operator semanticsの対応を新旧で記録する。
- cache・phase順序・記録方法を変えたことを明記する。
- 新outputを旧`launch_v2`へ上書きせず、補完用namespaceへ保存する。

### 13.2 統合表の書き方

最終的なcoverage表には、各義務がどのrun/source/record/hashで満たされたかを記載する。旧runのterminalをCOMPLETEへ変更しない。

例えば、「旧run6 cell＋補完run2 cellにより、登録8 cellの証拠を統合した」と書く。**一回のrunが8/8完了したように書かない。** 補完で変えたsourceが科学的意味を変えていないことは、独立した回帰・scope対応で示す。

二つのsourceの意味論一致が立証できない場合、無理に統合しない。どの追加照合が必要かを切り分ける。数値不一致が出た場合も、既存thresholdや候補を変えて帳尻を合わせない。

### 13.3 予算

現runのphase/call/output上限は当時の契約として保存する。補完の時間・memory・cache・出力上限は、Codexがsource上の重複削減と保存timingに基づいて提案し、結果を見る前に固定する。

本レビューでは、未測定の新実装が何秒で終わるか、どのCPU/RAM割当を使用してよいかを決め打ちしない。単に上限を延長する案より、同じ検証量を保って生成の重複を減らす案を先に選ぶ。

---

## 14. Worker terminal欠測と進捗監査

親が強制終了すると、workerの通常のfinally/terminal保存が完了しない場合がある。今回は親terminalがreasonとworker terminalの欠測を正しく残している。これを実行成功へ書き換えない。[S03][S10]

次の改訂では、最終worker terminalだけに依存せず、最低限の進捗をboundedな副記録として残すことが望ましい。

- 最後に開始したcell/path/precision。
- 最後に完了・原子的に保存したrecord。
- attemptedとcompletedを分けたcounter。
- 最後に観測したwall/RSSと観測時刻。
- cache hit/missと高価な演算の経過。

これは全stateを頻繁に書き出すcheckpointを要求するものではない。ログ/出力capを守る小さいheartbeatでよい。**観測時点以降の処理量やlifetime peakを推定して埋めない。**

現状の欠測から、source173件や結果30件が失われたとは判断しない。保存監査がPASSしたことと、workerの未保存counterが分からないことは両立する。

---

## 15. H4-N/A/E/Mのレビュー判定

| 義務 | 今回の到達点 | 本レビューの扱い | 次の優先作業 |
|---|---|---|---|
| H4-N：入力・独立参照・数値水準 | occupation列、指定state正規化、参照MP80/120比較 | 経験的検証として大きく進展。厳密uではない | 同一target規約を維持し、必要な新source回帰のみ |
| H4-A：実時間・8 signal・中間norm | 179組×3probe、6 cell全経路、12 MP record | 部分達成。S4 2 cellは未完了 | 重複MP生成を減らしてS4 q1/q4を補完 |
| H4-E：event/phase/control/axis接続 | 今回の4群は0件。旧toy/構造検証は別に存在 | 当該分子代表eventの義務は未達成 | MP完了待ちから分離し、4群を実行 |
| H4-M：u-aware会計 | 既存synthetic準備を参照。今回N/Gはnull | 本runで資源会計を実証したとはしない | 後続会計前に契約・synthetic対応を確認 |

H4-N/Aが部分達成したことを利用し、同じ検証をゼロから繰り返さない。一方、H4-Eを「高精度平均が合ったから不要」とはしない。[S01][S02][S08]

---

## 16. H6へ進むか：技術pilotと本検証を分離する

### 16.1 今はH4の未完了義務を先に補う

現時点でH6を無条件に開始するより、S4 2 cellとexplicit4群を補う方が、比較の正しさと接続の説明を強くする。これは新しい全面レビューやH4全再実行を増やすという意味ではない。

H6対応sourceの整理、input/plan/budgetの準備はCodexが並行して進めてよい。ただし、現在の依頼はレビューであり、新しいH6入力生成・実行の認可にはしない。

### 16.2 以前のDF rank問題を再び未修正と扱わない

H6準備契約v2では、tol-only adapterが`truncation_threshold=1e-8`だけをdecomposerへ渡し、`final_rank`や分子config rankを供給しない方針とmock検証が記録されている。これは前回の「`df_rank=None`がconfig rankへ戻る」問題への修正方針である。[S09]

したがって、今回再び同じ問題を新発見したかのように指摘しない。残るのは、実入力生成時にactual kwargs/rank/切断値・補正・Hermitizationを記録し、準備契約どおりであることを確認すること。

また、最新sourceにはH6専用snapshot loaderとH6分岐が存在する。古い準備文書の「molecular backend未実装」という記載を、そのまま最新sourceの現状として繰り返さない。**source経路があること、入力が固定されていること、実H6で検証済みであることは別**である。[S04][S09]

### 16.3 H6の7 cell・36 wrapperは維持

現案は、S2/S4、discard、PR、random-dominant、ordinary/symmetric、cosine/sineを小さい範囲で接続する技術pilotである。今回のSTOPだけを理由に候補を増やしたり、この案を最適化campaignへ変えたりしない。[S09]

| 対象 | 維持する役割 |
|---|---|
| B0 S2 q2 | discardとPF分解 |
| B1 S2 q1/q2 | 二次決定論の基準 |
| B1 S4 q1/q2 | 強い対照・負時間 |
| B2 K2 q2/R4 | 部分ランダム化接続 |
| B3 K6 q2/R4 | tail・finite cutoffの技術検査 |

5 deterministic cell×4 wrapper＋2 random cell×2 replica×4 wrapper＝36。H6の全露出はdevelopmentとする。

H4で時間がかかったMP処理を、400次元H6 sectorへそのまま全stage・全precisionで拡張する計画にはしない。H6の現sourceは別のbinary64独立oracle経路を持つ。その技術scopeを保持し、必要な精密照合は不一致や数値条件に基づいて限定する。[S04][S09]

### 16.4 H6技術pilotへの条件と、本検証への条件

H6技術pilotには、H4の必要な接続検証、tol-only実入力、sector/basis、source/input/plan、独立oracle、hard cap、失敗記録、別の明示的launchが必要である。**総uの厳密certificateは、N/G=nullのtechnical実行を認める契約なら一律必須にしない。**

H6本検証には、それに加えて、資源結論を支えるu/headroom、finite平均とestimatorの関係、公平な候補集合・baseline・探索境界、期待cost統計とconfirmationが必要である。H6 pilotの完走だけで本検証へGOを出さない。

H8は、H6後のmodel・仮説・候補・標本・主張をfreezeするまで独立評価用に保護する。[S09][S12]

---

## 17. 研究方針・新規性・着地点への影響

**RQ-R主軸、RQ-P1補助、RQ-P2未達成という前回方針を維持する。**

今回得られたものは、新たな資源winnerやPR優位ではなく、比較の基盤となる数値実装の信頼性である。これは重要だが、MP計算の高速化そのものをTrack Aの主題にすり替えない。

主RQは引き続き次である。

> 同じDF target・指定state・finite-time signalと測定会計の下で、PRはどの精度・partition・サイズ・回路実装条件で競争力を持ち、資源差は回路費用・normalization・bias marginのどこから生じるか。

今回の高精度照合は、その問いに答えるための検証基盤である。PRが勝つこと、S4が負けること、FEWが改善することは本レビューの進行条件にしない。

旧FEWの追加fit、H4 geometryの網羅、全wrapperの再compile、H8先行計算、full QPE/物理runtimeへの拡張は、今回のSTOPを直接解決しない。優先しない。

今回の限定的な外部確認は数値ライブラリのprecision規約のみである。新たなPR関連研究の網羅比較や、独立論文の新規性をここで確定したとはしない。以前の新規性評価・論文化の限界はそのまま残す。[S12]

---

## 18. 代替案の比較

| 案 | 利点 | 問題 | 判断 |
|---|---|---|---|
| 同じsourceで全8 cellを再実行し、時間だけ延長 | 変更が少ない | 完了6 cellを再計算し、同じ重複を維持する | 第一案にしない |
| MPをさらに高精度へ上げる | 精度依存の追加情報 | 既完了の極小差をさらに詰めても、S4欠測・event接続を解決しない | 現段階で不採用 |
| S4またはeventを削除してH4完了にする | 見かけ上早い | 既登録の重要義務と公平性を落とす | 不採用 |
| 直ちにH6へ進み、H4欠測を後回しにする | サイズ進展が早い | 小系で切り分けられる未確認事項を持ち越す | 第一案にしない |
| **同一演算の再利用＋不足2 cell/4群の補完** | 意味論と証拠強度を保ち、不要な再計算を減らす | 改訂sourceの回帰・新旧証拠統合が必要 | **採用** |
| 全数値演算の厳密certificateを先に完成 | 最も強い保証を目指せる | 元研究を数値認証研究へ変更しやすい | 全体の必須条件にしない |
| H4で予測モデルを改良する | 既存データを使える | 現在の未完了義務と無関係 | 不採用 |

この採否は研究価値と検証coverageに基づく。単なる計算正常終了率の改善を目的にしない。

---

## 19. 次の担当と作業単位

**次の担当：Codex。** 今回のGPTレビューで研究上の判断を行ったため、次は具体的な実装・検証の実行側へ戻す。

### Codexへまとめて任せる範囲

1. MPの同一演算再利用を実装し、precision・input・時刻・mutationの分離を検証する。
2. 必要な前提を保ってexplicit4群とS4 MP補完を独立したbounded単位へ整理する。
3. 完了6 cellを保全した補完coverage表と、新旧sourceの対応を作る。
4. partial/heartbeat/terminal、attempted/completed counterの記録を補強する。
5. 新source、test、補完対象、予算、output、authorization依存を固定して公開する。
6. 別途ユーザーが明示的に実行を指示した場合にだけ、固定した補完を実行し、結果と監査を保存する。
7. H6の既存準備を維持し、input・actual-rank・plan・資源の未確定事項を整理する。

module分割、cache実装、synthetic fixtures、細かなbudget enforcementはCodexの裁量でよい。本書はコーディング手順の逐条指示ではなく、科学的に維持する条件と目的を定める。

### 不要なGPT往復を増やさない

同じtarget・意味論・coverage・証拠強度のまま、cache、logging、schema、path、metadata、testを改善する工程ごとに、本格的なGPT研究レビューを必須にしない。

早期に戻すべき条件は、数値的不一致、target/stateやPF・event意味論の変更、MP precisionや登録coverageの削減、独立性を失うoracle変更、予算内に必要証拠が得られない科学的な計画変更である。

H4補完が整い、H6入力・実行について別の明示的認可が与えられた後は、登録した7 cell/36 wrapperの技術pilotを実施する流れを維持する。**次の主要なGPT研究レビューは、H4補完とH6技術pilotの証拠に基づいて、H6本検証の科学的GO/STOPを判断する節目**とする。新たな矛盾が出た場合はその前に戻る。[S09][S12]

---

## 20. GO/STOPと判断変更の条件

### 今回確定すること

- Track Aを継続する。
- 完了6 cellを経験的な数値検証として保持する。
- S4 2 cellとexplicit4群を優先して補完する。
- 最初の対策はprecision低下・候補削除ではなく、同じ行列指数等の生成の再利用。
- 研究方針とH6 7 cell/36 wrapperの技術目的は維持する。
- H4/H6の旧terminal・N/G=null・認定falseは変更しない。
- 本書自体から新しい科学実行のauthorizationは発生しない。

### 科学的に止めて再判断する条件

- 同じ保存target・state・scheduleの比較で、説明できないnative/MPまたはevent/controlの不一致。
- precisionを変えても収束しない、scale guardを破る、または数値不確かさが必要なheadroomを侵食する。
- 補完のために科学的条件を変えないと実行できない。
- 旧結果と新sourceの対応を確認できず、統合coverageを主張できない。
- H6のDF policyが実際のdecompositionと不一致、またはsector/入力が不正。

### 研究全体の停止理由としないもの

- 今回の一回の時間上限STOP。
- 比較の片方が期待より高価だったこと。
- PRより強い決定論法が有利だったこと。
- 予測モデルの改善余地が小さいこと。

科学的に比較可能で、どの条件で利益が失われるかを説明できれば、それもRQ-Rの有効な結果である。

---

## 21. 前回レビューからの変更履歴

| 論点 | 前回 | 今回の更新と理由 |
|---|---|---|
| H4独立参照 | 共通実装への依存を補う必要 | occupation列とMPの証拠を得た。未着手へ戻さない |
| 実stage時間 | ±0.2だけでは不足 | 179組×3probeの検証を追加済み。local scopeとfull-cellを区別 |
| 8 signalのMP照合 | 追加検証の提案 | 6 cell完了、S4 2 cellのみ補完が必要 |
| Event/control | 必要な接続検証 | 今回0/4。MPと独立なphaseに整理して優先 |
| MP負荷 | 未実測 | 保存wallとsource上の反復生成が判明。再利用を第一対策にする |
| DF tol-only | config fallbackが問題 | 専用adapter方針・mockは既に記録済み。実H6生成確認を残す |
| H6実装 | 準備・portが必要 | 最新sourceにはH6分岐がある。全面新規実装を再要求しない |
| 総u | 厳密認定と経験値を分離 | 方針維持。完成6 cellを有効な経験的証拠として利用 |
| 研究方針 | RQ-R主、RQ-P1補助 | 維持。STOPで方針選定をやり直す理由はない |

**担当をCodexへ戻すのは、レビューを省略したためではない。今回の科学的評価と補完方針を確定したためである。**

---

## 22. 本レビューの実施記録

### 実際に行ったこと

- 固定commitのreport・audit・契約・sourceを読取。
- MPの入力持上げ、正規化、stage、比較、記録、呼出し順序の静的確認。
- 代表的なMP120一次JSONのsignal・分解・規約を読取。
- 保存wallの和・比、保存B3複素差のノルム、B2のsource由来の演算回数を算術計算。
- 以前の詳細レビューと今回の到達点を比較。
- 公式mpmath文書によりworking precision・表現の意味を補足確認。
- 本Markdownを作成。

### 行っていないこと

- 科学runner、MP行列指数、solver、sampling、compile、fit、shot/G会計の実行。
- 全173 source、195 freeze、30 raw出力の独立な全件hash監査。
- 全MP traceの独立再集計。
- H6/H8入力の生成・load・性能評価。
- H4/H6の実行authorization作成やrepoへの変更・push。
- 最新PR文献の網羅調査、新規性の再確定。

---

## 付録A：レビュー算術

保存値から次を計算した。いずれも新しい科学実行ではない。

| 算術 | 結果 |
|---|---:|
| 完了6 cell wallの和 | 1,723.2933916752227秒 |
| B2 K2＋K4 wall | 1,109.7384679438546秒 |
| 上記の完了cell wallに対する割合 | 約64.3964% |
| 観測されたworker RSS下限 | 380.08203125 MiB |
| 原出力30件の容量 | 約16.4727 MiB |
| B3 MP120 finite差ノルム | 約7.03327×10^-13 |
| B3 MP120 total差ノルム | 約3.687572×10^-4 |
| finite/totalノルム比 | 約1.90729×10^-9 |
| B3 native correctedとMP120の差（表示native値使用） | 約4.54×10^-16 |
| B2 1 cell・1 precisionのdeterministic MP指数生成の論理回数 | 168 |
| 同一cell内のdeterministic指数生成器の種類 | 7 |

時間の比はspeedupではない。RSSはlifetime peakではない。MP差はu_boundではない。演算回数はsourceからの計数であり、欠測workerの最終counterを復元したものではない。

---

## 付録B：出典と固定参照

本文の[Sxx]は以下の固定GitHub資料、[Exx]は公式数値ライブラリ文書を指す。ローカル絶対pathを唯一の参照先にしない。大きな資料については、本書§1.1に読取範囲の制限を記録した。


- **S01** — [結果・レビュー索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/docs/research/track_a_ax2b_h4_limited_execution_v2.md)
- **S02** — [保存結果監査：完了数・数値・欠測・来歴](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/artifacts/resource_applicability/track_a_ax2b_h4_limited_execution_v2/2026-10-10/saved_execution_audit_v2.json)
- **S03** — [親terminal：時間上限STOPとworker terminal欠測](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/terminal_status.json)
- **S04** — [実行backend：準備・primitive・cell・phase順序・比較](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b228f2307f5fea77f066ee13e11ba6d2d8b9bea7/src/trottertracks/resource_applicability/ax2b_molecular_ports_v3.py)
- **S05** — [Stage/MP oracle：independent occupation、MP80/120、反復生成](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b228f2307f5fea77f066ee13e11ba6d2d8b9bea7/src/trottertracks/resource_applicability/ax2b_stage_validation_v2.py)
- **S06** — [MP比較関数：compare_mp_records / compare_stages](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b228f2307f5fea77f066ee13e11ba6d2d8b9bea7/src/trottertracks/resource_applicability/ax2b_molecular_ports_v3.py#L565-L609)
- **S07** — [B3 K6 MP120：signal、誤差分解、state規約、代表trace](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/H4_B3_K6_mp120.json)
- **S08** — [H4限定検証契約v3：N/A/E/M、time集合、精度・記録量](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/docs/research/track_a_ax2b_h4_prelaunch_contract_v3.md)
- **S09** — [H6技術pilot準備契約v2：tol-only、7 cell/36 wrapper、未認可](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/docs/research/track_a_ax2b_h6_pilot_preparation_contract_v2.md)
- **S10** — [Bound runner：grant、phase watchdog、terminal](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b228f2307f5fea77f066ee13e11ba6d2d8b9bea7/scripts/resource_applicability/run_track_a_ax2b_bound_v3.py)
- **S11** — [独立occupation reference実装](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b228f2307f5fea77f066ee13e11ba6d2d8b9bea7/src/trottertracks/resource_applicability/ax2b_independent_reference.py)
- **S12** — [前回GPT独立科学レビューのrepository記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/docs/research/track_a_ax2b_h4_post_independent_scientific_review_2026-10-10.md)
- **S13** — [入力・参照の一次記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/input_reference.json)
- **S14** — [179 primitive/time組・3probeの検査記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/primitive_validation.json)
- **S15** — [実coverageと上界の一次記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/actual_coverage.json)
- **S16** — [実coverageとsealed coverageの照合receipt](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/coverage_comparison_v3.json)
- **S17** — [173 source/195 freezeと出力来歴のinventory](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/artifacts/resource_applicability/track_a_ax2b_h4_limited_execution_v2/2026-10-10/execution_inventory_v2.json)
- **S18** — [独立の実行監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/artifacts/resource_applicability/track_a_ax2b_h4_limited_execution_v2/2026-10-10/execution_audit_v2.json)
- **S19** — [H4 coverage修正準備v3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/docs/research/track_a_ax2b_h4_coverage_preparation_v3.md)
- **E01** — [mpmath公式文書：Basic usage / working precision](https://www.mpmath.org/doc/current/basics.html)。2026-10-10閲覧。閲覧ページ表示は1.3.0。working precisionは個々の結果のaccuracy保証とは別であることを参照。
- **E02** — [mpmath公式文書：Matrices / matrix functions](https://www.mpmath.org/doc/current/matrices.html)。2026-10-10閲覧。mp.expmと通常のMP行列、interval行列の区別を参照。通常MP計算の成功をvalidated numericsへ自動昇格しない。

[S01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/docs/research/track_a_ax2b_h4_limited_execution_v2.md
[S02]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/artifacts/resource_applicability/track_a_ax2b_h4_limited_execution_v2/2026-10-10/saved_execution_audit_v2.json
[S03]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/terminal_status.json
[S04]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b228f2307f5fea77f066ee13e11ba6d2d8b9bea7/src/trottertracks/resource_applicability/ax2b_molecular_ports_v3.py
[S05]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b228f2307f5fea77f066ee13e11ba6d2d8b9bea7/src/trottertracks/resource_applicability/ax2b_stage_validation_v2.py
[S06]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b228f2307f5fea77f066ee13e11ba6d2d8b9bea7/src/trottertracks/resource_applicability/ax2b_molecular_ports_v3.py#L565-L609
[S07]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/H4_B3_K6_mp120.json
[S08]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/docs/research/track_a_ax2b_h4_prelaunch_contract_v3.md
[S09]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/docs/research/track_a_ax2b_h6_pilot_preparation_contract_v2.md
[S10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b228f2307f5fea77f066ee13e11ba6d2d8b9bea7/scripts/resource_applicability/run_track_a_ax2b_bound_v3.py
[S11]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b228f2307f5fea77f066ee13e11ba6d2d8b9bea7/src/trottertracks/resource_applicability/ax2b_independent_reference.py
[S12]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/docs/research/track_a_ax2b_h4_post_independent_scientific_review_2026-10-10.md
[S13]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/input_reference.json
[S14]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/primitive_validation.json
[S15]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/actual_coverage.json
[S16]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/coverage_comparison_v3.json
[S17]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/artifacts/resource_applicability/track_a_ax2b_h4_limited_execution_v2/2026-10-10/execution_inventory_v2.json
[S18]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/artifacts/resource_applicability/track_a_ax2b_h4_limited_execution_v2/2026-10-10/execution_audit_v2.json
[S19]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a87cf25548a3b93262027ce780317872d7c4e883/docs/research/track_a_ax2b_h4_coverage_preparation_v3.md
[E01]: https://www.mpmath.org/doc/current/basics.html
[E02]: https://www.mpmath.org/doc/current/matrices.html

---

**最終判断：研究方針は維持。次はCodexで、既存6 cellを保全したS4 2 cell・explicit4群の補完を、反復生成の削減と新しいsource/plan/budget固定により進める。H6への自動launch、本検証・H8への自動GOは行わない。**
