# 段階的探索・価格取得と必要最小限の科学実行案

2026-11-13 JST。**未承認・未実行。** 今回は外部候補もNも新しく生成せず、既存の実H4値も再解析しない。以下は候補menuと比較目的を受けた後に具体化する有限計画であり、科学実行の開始指示ではない。固定L_D1の追加path、旧24 local-window案、m7→m8→m9を先に実行する要件はない。

## 1. 探索と価格取得の段階

| 段階 | 入力・作業 | 次へ進む条件・出力 |
|---|---|---|
| 0：比較問題の固定 | 探索担当からfinite physical menu、B/N担当から同targetのN資格又は𝒩、ユーザーからγ又はτ・confidenceを受ける | 比較対象H/state/energy条件とprofileが不一致なら比較せず返す。十分N/feasibility不明は価格で補わない |
| 1：metadata照合・探索順序 | 既存assess/evidence.plan、保存証拠のreference照合。必要なら新候補のcount/proxyを別scopeで取得する | infeasibleを根拠付きで除外。S/proxyによる順序はheuristic、未探索候補を最適性対象から黙って消さない |
| 2：粗い価格区間 | 証明済み上界/下界、実行可能な直接mean CI、同一quantity bridgeを優先。未知セルは[0,+∞] | feasible incumbentの全正重みセルの上界を囲むか、有効な候補差区間を作る。全候補の高精度価格は不要 |
| 3：区間で候補を残す/落とす | L_x>U_b又は差分下端>0で除外。比較gapと未知項の寄与から競合点を選ぶ | 重なればUNRESOLVED。減らせる幅/測定費用を比較し、次の有限cell集合だけを提案 |
| 4：最終比較用取得 | 残った候補の決定的なcellを直接測る、又は根拠付き残差法の使用domainを検証 | 全使用domainとN資格を満たした区間だけをBへ返す。厳密優劣又はτ-regret、absolute G取得を別のclaimにする |

現時点では、新しいPR候補について非自明なcompiler下界/有限上界も、全round residual modelの誤差上界も未取得である。したがってこの設計だけで候補を除外できるわけではない。すべて[0,+∞]なら判定不能であり、有限upper又は有効な差分を最初に得る費用を要する。

## 2. 最初に提案する有限科学scopeの選び方

**第一候補は、外部menuのfeasible incumbent候補1個と、価格精度が判断を左右するchallenger1個以内に限定した取得案**である。候補をこの2個に限定して調べること自体は、menu全体の最適性証明ではない。他候補は既存適格証拠で除外できるか、明示的に未解決として残す。

### A：既存部品で直接取得できる場合

1. whole configurationとphysical cellを凍結し、既存同一価格の共有・参照可否を照合する。
2. incumbentの有限上界に欠けるcell、又は差分で消えない競合cellだけを集合Jにする。単に全26軸の精度を揃えるJにしない。
3. 各cellのnormal cohortを固定n_jで取得する。同path X/Yのfull-wrapper CとS・6指標・fingerprint・seed/digest・wall/RSSを保存する。Sの期待値が未取得なら直接mean(C)を使う案を優先でき、E[S]構築を強制しない。
4. CとDの双方を記録する場合でも、実dataを見て推定法を選び直す扱いを先に定める。pilotと最終cohortを混合せず、P1/replayをfresh nへ加えない。

両軸を測るcompile上限は **2Σ_(j∈J)n_j**（jはcandidate/roundのpaired cell）。seed計算は元carrierのcandidate/cohort/index/outer/piece規則を使い、新しいcohort namespaceとindex範囲を固定する。価格quantity/law全域で軸同値が立証された場合だけ、別途軸数を減らす案を検討する。

### B：新L_D等でcandidate-context adapterが必要な場合

科学run前に、既存prepare/compose/Accounting/wrapper/compilerを繋ぐ薄いadapterと人工検査を同じ技術作業内で作る。複数候補の交互処理でcontext/cacheが混在しないこと、signed piece・角度・scalar一回・両軸・六指標の保持を確認してSOURCEを固定する。旧C1/P2のguardを解除して実行しない。

科学的な実行可能性・pipeline対応・新候補の分散情報が不足する場合に限り、次の**診断だけ**の有限案がある。

- 供給menuから候補は最大2個。各候補で、未対応構造の小さい合法cell1点と、実際の最終比較に必要な最大/高影響cell1点以内を事前指定する。
- 各点fresh normal 2 path、X/Y個別。最大 **2候補×2 cell×2 path×2軸＝16 compile**。この2 pathは分散・wallの予備診断で、平均価格の精度保証ではない。重複cell・既存適格証拠がある点は減らし、不要な枠を別条件で埋めない。
- 小さいcellはpipelineの技術確認、最大domain点は実行可能性/文脈のchallengeであり、間のroundや別L_Dのmodelを認定しない。最大点を実行不能ならUNKNOWNを返す。小roundを順次増やして予算を置換しない。

16 compileは一律の前段要件ではなく、必要情報が既にあれば省く。最終比較のCIが作れる前提を持つAへ直接進める場合は、診断のためだけに再測定しない。診断後の新n/取得点・保存値再解析は次の明示承認された有限scopeとする。

## 3. 必要nと取得費用を決める条件

Aのn_jは、平均価格又は残差に対するabsolute MC半幅t_jとfailure budget α_jから決める。独立同lawのpath変数Zが全populationで[L_Z,U_Z]、幅W_Zに入ることを根拠付きで示せるなら、例えば

\[
n_j\ge\left\lceil {W_Z^2\over2t_j^2}\log{2\over\alpha_j}\right\rceil
\]

という固定nの保守設計を使える。Dは負なので幅で扱う。sample min/maxをsupport上界へ置換しない。compiler出力rangeが不明なら、この式の数値nも未確定である。pilot分散による概算又は正規近似を使う場合は名目設計であることを明記し、根拠付きCIと混同しない。既存の公開統計手法を選び、D/Bで新推定frameworkを作らない。

t_jは全価格誤差ではない。Sの数値幅、law差、model/window tail、rareの未被覆部分と合わせて、比較差の誤差予算を満たす必要がある。全lawに有効なnormal CIがrareも含むなら二重加算しない。rareを観測しなかった診断cohortを全lawの証拠にしない。

将来、適格N・分散/費用情報が供給されたら、|v_j|σ_j/√k_j等を取得配分の参考にする。未知共分散・systematicを0とせず、誤差を減らせない点への無意味なMC追加を避ける。local-windowやmodelは実際の「必要n×path費用＋校正/feature取得費用」で直接法と比較し、局所回路が小さいことだけで採用しない。

## 4. 残差モデル・長時間への追加実測が必要になる分岐

直接法又は証明済み粗区間で選別できる間は、全L_D共通残差fitやouter-window検証を先に行わない。必要な長時間cellの価格が実行不能で、かつその未知幅が比較結論を左右する場合だけ、次を別案にする。

- 使用候補/L_D・recipe・q/R/K・角度・境界を含むdomainと、候補に有効なモデル誤差根拠を先に定義する。p係数にはrank≥pの校正が必要だが十分条件ではない。
- model選択後の独立challengeは最大使用domain点と、未被覆regimeを最も問う内部点を優先する。別L_Dを旧m0/m6のholdoutとみなさない。
- 少数challengeのPASSだけで残りdomainを認定しない。全域移送根拠がなければ必要なcellの直接CIに戻るかUNKNOWN。
- outer-window法のtail誤差は未証明。最大domainの適格reference又は合法lawの全tail上界なしにm12を埋めない。旧24 compile案はこの候補探索の自動次段階ではない。

この分岐でfit/保存値再解析をする場合も、対象cohort・family・SOURCE・有限path/compile/fit回数を別途固定する。

## 5. 承認前に埋める実行表とSTOP

現在は候補menu/N/必要価格差が未供給なので、以下の対象名・n・scientific SOURCEは**UNRESOLVED**。この文書を実行可能な凍結PLANと表示しない。

| 項目 | 実行前に必要な値 |
|---|---|
| candidate/round | 外部configuration・physical candidate・m/X/Y・T/q/R/K/recipe・lowering/profile・preparation/input参照 |
| method/target | direct C / paired D / 別検証済み方法、ideal Pかsampler Q、cohort role、law/rareと数値契約 |
| source/finite scope | adapterを含むcommit、path数n_j・index/seed生成規則・最大compile数、fresh出力先、cache照合・再利用の対象 |
| decision | t_j/α_j・同時family・γ/τ・N又は𝒩の資格、精度未達/UNKNOWNの保存方法、実行可能性判定 |

ユーザーの対象・SOURCE・有限範囲に対するチャット承認で十分。追加grant/承認証明書を要求しない。正常な技術作業を段階ごとに再承認へ戻さず、SOURCE固定後に許可範囲を続行する。

不一致、support/想定domain超過、実行不能・保存不能は部分結果を保全してSTOP。新しいseed・path・roundへの無断retry/追加/自動P3はしない。CPU/RAM/workerへ任意の固定上限を置かず、利用可能資源と入出力依存に合わせて候補/pathを並列化する。共有環境・他ユーザーの設定/プロセス/ファイルは変更しない。

**次の担当**：探索担当がmenuを渡す。B/N担当がNと比較上の資格を渡す。ユーザーがγ/τ/confidenceと必要な有限科学scopeを決める。Dが必要なadapterと価格取得案、Cが必要時だけE[S]/featureを供給する。今回はcommit/push・remote照合後mandatory STOP。
