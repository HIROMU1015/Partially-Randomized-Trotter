# 最小科学検証案：outer-window残差と長時間domain

2026-11-13 JST。**未承認・未実行。今回の納品はSOURCE調査・設計・人工/回帰テストまで。**
旧P1/P2・C K0/K1の契約とSTOPを変更せず、追加grant・承認証明書は要求しない。

## 1. 先に固定する研究上の条件

対象は固定H4 DF10/S2/L_D1/ΔPF=.1/K2/h=.1/M12/P0、新ZZ、元compilerと全wrapper。
必要なabsolute価格精度をε_mに定め、S数値区間、local-window標本誤差、compiler influence tail、
law差、rare/tail、跨R model誤差に配分する。旧P2の2%・95%はまだ候補である。
厳密保証と、採択したPRNG/IIDモデルに条件付けた経験的評価を区別する。

compilerの有限影響範囲・decay boundが不要になるような比例式は仮定しない。
未知誤差が残る点はUNKNOWNでよく、研究目的のために全機構を分解する義務はない。

## 2. 最初の有限案：最大24 local-wrapper compile

短windowは元のouter V_iをそのまま連結し、head/rotation角/順序を変えない。
**wrapper・scalarは各targetのfull Tを固定して一回**とする。
n=1/2/3をlocal window長と呼び、q_targetや元physical候補IDと別に記録する。

| 段階 | 対象 | input draw/reuse | local lengths | compile上限 |
|---|---|---|---|---:|
| A：既知の長q参照へのdevelopment照合 | m6 q_target64/R128/T6.4 | P2保存fresh index0,1の元event/seedをread-only再利用。最初3 outerのprefixのみbuild。元full q64のC/S recordを参照 | 1,2,3 ×2 path ×X/Y | 12 |
| B：最終使用R・小角の直接probe | m12 q_target4096/R8160/T409.6 | fresh2 drawのouter0..2だけをnormal lawで生成し、prefixを共用。**full q4096 trajectoryは生成しない** | 1,2,3 ×2 draw ×X/Y | 12 |
| 合計 | 2 target conditions | Aはsampling0、Bは2×3×8160=48,960 event位置 | 最大windowは24,480 event位置 | **24** |

Aのpathは既知P2からindex順で選ぶので、結果を見た独立holdoutと表示しない。
Bは長qのreferenceではなく、target R/角度/全体scalarを保った局所測定である。
現在P1/P2で未観測だったrareを強制的に追加する設計にはしない。normal drawのorder countsを記録する。
この2 drawは診断用であり、平均価格の必要nという意味ではない。

共通設定：Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、CONTROLLED_ZZ_CX_CRZ_CX_V1、
measurementを含むX/Y個別compile。同じdrawのnested prefixを使い共分散を保持する。
cache hitはcompiler/fingerprint照合後だけ使い、節約分でwindowやdrawを追加しない。

将来のB seedはcohort `D_OUTER_WINDOW_DIAGNOSTIC_V1`、index0/1、元carrier trial_seedの
target physical candidate/outer0..2/piece0で固定する。partial drawをfull-path標本と混同しない。
必要な限定window adapter・人工同値性テストを実装し、そのSOURCE・PLAN・全seedを固定してから実行する。
この納品のhelperと固定5a67cea7の構築/compileを基礎にするが、実行用adapter SOURCEはまだ未作成である。

### 必須のwindow構築条件

- C1 contextの元input/preparation検査、元tail builder、新ZZ、native PFとwrapperを再利用する。
- 各outerのsigned step_timeは元requestの値を使う。**T_target/nへ変更しない。**
- n×deltaを作ってから割り直す浮動小数点差にも注意する。元1-outer構築の語を連結する。
- scalarはp(−scalar T_target)を一度付ける。window長に合わせたscalarへ置換したり、
  scalar係数のrescaleで別の丸めを導入したりしない。
- Sも同じn outerとtarget capsで元Accountingを一回finishする。既存structural_pairを利用できる。
- C1 price_requestsの元candidate/q/time配分検査を偽装して通さない。
  新しいlocal-window quantityを使い、元physical候補のfull-price recordを上書きしない。
- barrierや追加gateを入れて境界を作らない。別compileの価格和が元compiler価格になると仮定しない。

## 3. 記録・判定

各n/axis/drawでC、S、F_n=C−S、wrapper fingerprint、6指標、元event digest/seed、
target/R/Tとlocal長、order/Z/ZZ/role/basis count、wall/RSSを記録する。
対応prefixのF_n共分散、δ_3とh=2/3 stencilの差を診断する。

Aでは保存D_64と(63平均F_2−62平均F_1)等の差を、**pairedな有限標本の差**として記録する。
1 pathのprefixと残りouterの変動は同じではないので、pathごとの差をそのままmodel biasと呼ばない。
同じデータでモデルを選択した場合もDEVELOPMENTのまま保つ。
短windowでδ_3が小さいことから、δ_4以降や全qのtailを0にしない。

input/SOURCE/復元/角度・cap・compiler検査の不一致は部分記録を保存してSTOP。
診断誤差が大きい・標本誤差が不十分という場合も自動でwindow長やnを増やさず、有限案を返す。
24 compile範囲を承認する場合はA→Bを一つの範囲として指定でき、正常時の段階ごとの再承認は不要。
実際の資源不足・異常は保護するが、一時的な負荷だけで正常workerを再起動しない。
並列度は実際の利用可能資源とcaseの依存から選び、CPU/RAMの固定した低い上限は設けない。

## 4. 長qの認定に別途必要な証拠

24 local compileの案だけではm12のE[C]を認定できない。長時間域を妥当に扱う道は次のいずれか。

1. **構造/解析のtail証拠**：合法H4 alphabetと固定compiler規則で、影響がwindow外へ進む条件を
   定義し、全残余spanに対する上界を出す。有限radiusの閉包、またはlawに条件付けた影響spanのtailと
   残差振幅の上界が候補。Sのreset証明を流用して成立としない。
2. **最大domainの適格reference**：model/horizonを凍結後、最も大きな使用qで独立referenceを得て
   予測誤差・cohort/rare不確かさを評価する。full q4096 compileが実行不能なら、この道の資格は得られない。
   既存未資格native bridge出力や、同R/小qのwindowをgold referenceへ格上げしない。

どちらも得られなければm12と未検証長qはUNKNOWN。
full-q referenceを検討する場合、まず最大2 path/4 compile等の実行可能性診断を**別の**有限案にし、
必要nの十分性は別途決める。この案を現在の24 compileへ潜り込ませない。
より小さいm7→m8→m9を順番に通したことを、最終qのtail証明の代用にしない。

跨Rのモデルを使わなければ、方法成立後は各targetの短windowを測って全13Rを扱える。
跨Rを減らす場合は小角・compiler簡約閾値・capsのdomain誤差が別に必要で、
m0/m6だけから全Rへ移送しない。全13 E[S]の既存enclosureはその誤差を含まない。

## 5. 必要nと誤差予算

paired stencilのZ_i=a F_h,i+b F_(h−1),iを用い、
s_Z²=a²s_h²+b²s_(h−1)²+2ab s_crossとする。
名目予算値はn≈ceil(z²s_Z²/b_MC²)だが、小標本分散の代入は保証ではない。
同時confidenceやtail保証の方式を固定し、確認cohortを独立にして、最終有限nと予算を先に定める。
rare/law/window tailが予算を超えている状態で標本だけを増やさない。
local法の1 draw費用t_Zとv_Zからv_Z t_Z/b_MC²を比較する。
q重みで分散が増幅される場合は、小回路であるだけでは直接paired法より安いとしない。
既存full-path参照との同じ誤差配分での仕事量比較を将来行い、利点がなければ採用しない。
今回の2 draw/targetから十分nや厳密なwall予算を確定せず、追加nは別の有限案にする。

strictな価格誤差には、e_S+e_MC+e_window+e_law+e_rare+e_crossR≤ε_mを要求する。
normal cohortのCIがrareも含むなら二重にrare誤差を足さない。未観測rareを0として引かない。
N/GやBの4方式menuを計算せず、価格精度を研究者が指定したabsolute/relative基準で判断する。

## 担当とSTOP

- **ユーザー**：精度・law・保証水準を判断し、必要なら上記24 local compileのSOURCE・有限範囲を別途明示承認する。
- **Codex D**：承認後に限って限定adapter/SOURCEを具体化して実行。現在はcommit/push・remote照合後mandatory STOP。
- **Codex C**：必要なfeatureだけを[C_HANDOFF](C_HANDOFF.md)に従って別SOURCEで評価。今回は実行不要。
- **Codex B**：未資格価格は使用しない。現在の追加作業なし。
