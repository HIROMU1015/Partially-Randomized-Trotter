# PR-DFの次の分割：LD3とLD7の比較計画

1. **先に評価する分割はLD7を推奨する。** 遠い候補はLD5/7/9からLD7を選び、最終比較を**LD3対LD7**に絞った。これは取得優先度の選択であり、保存資料からLD7の価格や費用順位を予測した結果ではない。LD3/5/7/9の現行configuration・R列・信号・価格は今回生成していない。
2. **決め手は、同程度の種類の新規入力で、headが大きくtailが小さい別の領域を一度に調べられること。** LD3はfragment 2を一つ移す局所比較、LD7はfragment 2..6を移し、tailを[7,8,9]にする。両者とも現行13信号・価格・Q3を新しく整える必要があり、LD3の旧価格も現loweringには使えない。LD7はLD9ほどheadを端まで増やさず、残るtailが正の領域を調べられる。
3. **LD2長roundの1点確認は先行させず、後回しにする。** m7..12がLD2 B_PRICE総額の99.857098%という制約は保持する。ただし、その1点がモデルより高い場合も低い場合も、ここで選ぶ「次の分割」はLD7のまま。1点は既存費用の信頼性を改善するが、LD3とLD7のhead/tail交換のどちらが有望かを直接識別しない。今回の探索順に価格実測を待ち依存として追加しない。

対象は保存H4 linear 1.00 Å／STO-3G／tol-only DF10、元H/state/snapshot・fragment順、one-bodyを別headとするPR-DF、S2・Δ=0.1・K2・元κ、元のRPE/Q3精度契約。保存資料・SOURCEの読取と計画のみ。新しい科学的候補・schedule・信号・価格・費用の計算は0。

[候補差分JSON](CANDIDATE_COMPARISON.json)／[CSV](CANDIDATE_COMPARISON.csv)／[LD7有限案](NEXT_FINITE_PLAN.json)／[固定SOURCE一覧](SOURCE_LOCATORS.json)／[HANDOFF](HANDOFF.md)。起点はf42f6e9bの候補表、e0b3cad2のLD2価格、8ec7c882の6点比較。L_D=0..10の棚卸しは繰り返していない。

## 保存証拠からの差分

L_Dは元保存順の二体DF fragmentの本数。one-body correctionは別の決定論blockであり、本数に含めない。LD2はhead[0,1]・tail[2..9]。carrier155529e2 `h46_new2_native.prepare`からprovider c31ef424の`generation_partition`へ進み、元index順を使う。一般APIのweight順prefixへ変えない。

| L_D | LD2からtail→headへ移る元index | head / tail | 保存λ_R | 保存tail identity s_R | 今回の役割 |
|---:|---|---|---:|---:|---|
| 3 | 2 | [0..2] / [3..9] | 0.5832704246339966 | 0.09261516589219046 | 最終の隣接比較案 |
| 5 | 2,3,4 | [0..4] / [5..9] | 0.05440471795776202 | 0.008913337788299013 | 未選択・未評価 |
| 7 | 2,3,4,5,6 | [0..6] / [7,8,9] | 0.0003129539143720037 | 4.5302064712666724e−05 | 遠い候補・先行推奨 |
| 9 | 2..8 | [0..8] / [9] | 1.5605698403540042e−07 | 5.193351122457893e−08 | 未選択・未評価 |

LD2の保存descriptorはλ_R=1.6030956277929231、s_R=0.2223759094107018。上表の数値は再抽出・再計算したλ/scalarではなく、固定SOURCE_INDEX7a8bc7f1が参照する`H4_context.json`の保存値をコピーした。context SHA-256 `8396605fa0c5ee34f29b436ee16e0f731cfaa6cd94f3304033106a85e9f5d52f`、selector `prefixes.<L_D>`、H/state/snapshot/rankを照合。component表・operator・tail hashを新しく生成していない。

**s_Rは総scalarではない。** 実回路・signalはHamiltonian constant cと、その候補自身のs_Rを合わせて総Tで一回適用する。今回c+s_Rの新しい数値は算出していない。将来のSOURCE/input固定時にcの元保存欄とs_Rを別々に束縛し、runtimeのtotal scalarを照合する。

共通κのSOURCEは510fbacdの`paper_event_mapper`。literal規則は`target=κ λ_R² T²`、`r_target=ceil(target)`、`R=max(J,ceil(r_target/q))`、S2のJ=1。各候補は自身のλを使う。λが小さくなると抽象event予算は減少又は同じになり得るが、head/basis/相殺・scalar/信号/Nも変わるためcompiled価格や適合費用の単調減少は導けない。**本turnでこのmapperを呼ばず、新R列もphysical IDも生成していない。**

### 遠い候補をLD7に絞る理由

- **LD5**も有望な中間点。保存descriptorは使えるが、旧価格があることは現価格を再利用できる理由にならない。今回の一つの追加点には、LD3と異なる低tail側の領域をより広く調べるLD7を選ぶ。LD5を高費用と判断したわけではない。
- **LD7**はhead7本・tail3本の領域を調べる。小さい保存λと最低event数の規則から、R=1の下限が効く領域を検査する候補にする。これは全roundでR=1を新たに確認したという意味ではない。後続のconfiguration生成直後に検査する条件とした。
- **LD9**はtailが1本で、さらに決定論endpointへ近いがheadも最大に近い。極めて小さいλに対するproviderの`b>1`丸めgateも確認が要る。今回そこで失敗した証拠はないが、LD7より端の条件を先に選ぶ必然性はない。Full PFやB0I p7の価格へ読み替えない。

価格順位・実行時間順位は保存資料だけでは不明。LD7のheadがLD3より多いこと、tail eventとbasisの構成が異なることから、実compileのCPU/RSSがどちらで小さいとも断定しない。取得上限と必要なモデル入力の種類で負担を比較する。

## 信号・価格の再利用範囲

| 分割 | 分割descriptor | 現行13round signal / Q3 / 価格 | 旧証拠の位置づけ | 新規に要るもの |
|---:|---|---|---|---|
| 3 | 同じH4 contextにあり | 確認範囲で未同定 | f42表に価格metadata98件・signal metadata792件。価格例0469d792…はT2/q8/R32/K4/4th/Δ.25で不一致 | 現configuration/R列、13信号、固有価格、total view・有限配分Q3 |
| 5 | 同上 | 同上 | 価格metadata57件・signal metadata792件。価格例05777e93…はT1.2/q4/R32/K4/4th/Δ.3で不一致 | 同種類の不足 |
| 7 | 同上 | 同上 | 確認した旧5価格台帳にPR-prefix7価格は未同定。signal metadata792件あり | 同種類の不足 |
| 9 | 同上 | 同上 | PR-prefix9価格は同台帳で未同定。signal metadata792件あり | 同種類の不足 |

これらはf42の既存metadata件数の再利用で、独立path数ではない。旧signalはT=.8/1.2/1.6/2.0の散在grid。追加確認した四分割の近い旧recordには、いずれもT=.8/q8/Δ.1/S2/K2でR=4/8/16/32/64がある。**T/q/K/公式の部分一致**までは確認できるが、新候補のR・physical ID・law・source-bound intakeはまだ未定義なので、現signalとして受け入れない。候補IDと固定locatorはCANDIDATE_COMPARISONの`nearest_legacy_signal_metadata`へコピーし、signal値や価格値を再解析していない。

旧compiled価格はlegacy controlled-ZZ経路を含む。currentのCONTROLLED_ZZ_CX_CRZ_CX_V1、full controlled measured wrapper、axis、compiler/P0、SOURCE・cohort/seedを一致させた現単価ではない。新価格の較正へプールしない。LD2の係数・LD1のE_I[S]も他prefixへ移さない。PR-DF LD7はtailを捨てるB0I p7、PR-PauliのLD7とは別候補である。

**再利用できるのは**保存H/state/snapshot、元順fragmentとdescriptor、その元loader/provider、全13時点のFull-H reference pair、既存精度契約・Q3 kernel、LD1/LD2の6結果と凍結単価・費用、exact fit/generator・native/compilerの適合する部品。現行の新候補mean受入は0、現価格受入も0。将来新scheduleを固定後、§10.5で完全一致する保存mean/receiptが判明した場合だけ出所付きで新規件数を減らす。

## 最終2案：LD3かLD7か

| 比較点 | LD3を先行 | LD7を先行（推奨） |
|---|---|---|
| 初めて答える問い | LD2からfragment 2を追加する局所の交換は得か | headを7本まで増やしtail3本へ減らした地点でnetの低費用が残るか |
| 保存再利用 | descriptor・共通input/reference・library部品 | 同じ種類。13信号・current価格は双方未整備 |
| 価格案 | 固有full-C (1,q,qR)三点モデル。m0/m3/m6のrank3を後続metadataで確認、各4 pathなら最大24 compile。rank未確認 | 全13R=1を後続で確認できた場合、同じ一般形の識別可能な(1,q)版。m0/m6較正4 pathずつ＋m3独立2 pathで最大20 compile |
| 信号・Q3の規模 | 13 signal、共通V0→独自A/Bの最大3 Q3が必要 | 同じ上限。今回はこちら一分割だけを有限案にした |
| 分かる範囲 | 2→3の一点差、間の大きなhead領域は残る | 2と7の隔たりを一度に調べる。3..6が単調に動くとは分からない |
| 取得負担の不確実性 | tailが多くheadは小さい | headが大きく、tailのcomponent・event構成は変わる。wall/RSSの新しい数値見積りはしない |

LD7の(1,q)は安く見せるための係数削減ではない。Rが全round1ならqR=qなので、qとqRの別係数を三点fitしても識別できない。元のfull-C affine familyをそのdomainに制限し、合成slopeだけを求める。**R=1条件は今回未検査**。合わなければsignal/price/Q3前に停止し、この20 compile案を勝手に三係数モデルや別cal cellへ変更しない。

### 得られた結果が次の分割選択をどう変えるか

費用の比較は各候補自身が適合した有限配分内最小のモデル費用で行う。数値未取得の本turnでは下表のどの行に当たるかは判断しない。

| 最初に調べた分割と、その後の結果 | 次の分割判断 |
|---|---|
| LD3がLD2より低い適合モデル費用を持つ | 次にLD5を調べ、局所から中間領域へ進める。LD4等を単調性で補間しない |
| LD3が高い／同額 | 次にLD7を調べ、局所悪化の先の別領域を検査する。LD5/7/9を高費用として枝刈りしない |
| LD7がLD2より低い適合モデル費用を持つ | 次の候補はLD9。より大きなheadの地点で、残りのtail削減とhead増加がどう交換するか調べる。3/5が劣るとは結論しない |
| LD7が高い／同額 | 次はLD3。2と7の間の局所交換を追う。必要ならその後にLD5を加える |
| signal/SOURCE gate不成立、Q3未適合・不確定で有限比較点が得られない | 部分成果を保存し停止。物理方式の不可能性や全Nの不適合に読み替えず、技術修正又は別の有限shot案を判断する。自動の別prefix投入はしない |

LD1→LD2で費用が下がった事実は探索を続ける動機だが、この二点からLD7が安いとも、単調減少が続くとも予測しない。今回は「局所一点を足す」より「低tail・大headの領域を観測する」識別価値を優先した。

## LD2長roundの1点確認を先行させるか

8ec7c882の保存値を再利用する。LD1/LD2の最小適合モデル費用は189,232,849,610.910739／46,341,978,336.212733 RZ、選択Nは各軸[64,60,56,52,47,43,38,34,30,25,21,17,7]。**LD2 m7..12占比99.857098%**を再計算せず引き継ぐ。これにより既存費用の金額的な信頼性は長round外挿に依存する。

| 仮にLD2の一つの長roundを観測したら | この選択への影響 |
|---|---|
| 既存モデルと近い有限価格 | 既存点の参照を強めるが、LD3/7の未知のhead/basis/信号を比較できない。次はLD7 |
| モデルより高い価格 | LD2モデル総額を過小と疑う材料だが、一点/有限pathから全長roundの誤差は出ない。別prefixへの係数移送はせず、次の分割はLD7 |
| モデルより低い価格 | LD2の利点の大きさは変わり得るが、LD7のhead costやQ3の応答は分からない。次の分割はLD7 |
| SOURCE・gate・保存異常 | 技術問題を解決してから進む対象であり、これをLD3選択の科学的根拠にはしない |

ここで「近い」の数値許容差を採択したのではない。上下の例は選択の情報価値を示す仮定で、観測後の閾値設定や新たな境界算出はしない。どちらの価格方向でも選択が同じなので、**次の有限案にLD2追加取得を含めない（0 compile）**。旧HANDOFFの長round優先は既存費用順位の根拠を強める目的であり、本問の「次に調べる分割」の決め方とは分けた。正式E[C]/CI/Gの不足を解消したとは扱わない。

## LD7を比較へ加える最小の段階的有限案

以下は**未採択・未実行の提案**。 後続では生成用SOURCEとprefix/input/mapper指定をcommit固定してから候補を生成し、生成結果を固定してからsignalへ進む。機械可読正本はNEXT_FINITE_PLAN。R列やphysical IDがnullなのは未生成のためであり、0やLD2の値を代入していない。

| 段階 | 最初に取得するもの／再利用 | 新規上限と後続条件 |
|---|---|---|
| 0. 候補定義・入力固定 | a9abf8deのPREPARE_MENU型を一分割LD7へ適応。S0のinputs、保存prefix7 λ/scalar、元B/A metadata builderを使用し、現在のoriginal/total configurationと13 physical IDを後続で固定 | PR configuration1・13row。全R=1、S2単一piece、q/time/κ/input identityを検査。R条件外なら科学call前STOP |
| 1. 精度側の入力 | 固有13roundのraw/corrected/exact-tail meanとnormalizationをfefd548eで取得。元H/state、sector36、既存13reference pairを再利用 | 最大13 round-evaluation、39 complex出力、13normalizer。新Full-H reference action0、trajectory sampling0。欠測/数値gate失敗ならSTOP |
| 2. 共通配分での比較 | 既存LD1/LD2 V0と同じ各軸[64,60,56,52,47,43,38,34,30,25,21,17,12]で、LD7自身のsignalをQ3評価 | 新規Q3最大1。旧V0 2件は再利用。適合した有限anchorが得られた場合だけ価格/独自A/Bへ進む。不適合を全配分の不可能性とはしない |
| 3. 固有価格の較正・凍結 | m0（継承座標T.1/q1）とm6（T6.4/q64）の各normal fresh4 path・X/Y。R/ID/seedは新LD7正本から。whole-wrapper C平均へ軸別(1,q)をexact fit | 8path・16compile、2fit・26価格。全26価格positive/finiteを確認し公開・凍結。LD2係数移送0、S再加算0 |
| 4. 独立価格チェック | 凍結後にm3（T.8/q8）のfresh2 path・X/Y。較正と別cohort、同一pathから軸別receipt | 2path・4compile。軸別exact平均と凍結Pとの差を記述。許容差なし、refitなし。m3は補間確認で長round検証ではない |
| 5. 候補自身で配分を選ぶ | 固有価格によるV0モデル総額の3/4を予算に、元の正の有理数breakpoint法でA_SHAPE/B_PRICEを各1本生成。各軸の価格を使用し、列・実費用・残額をQ3前に固定 | 新列最大2・新Q3最大2。全vector重複は同候補内で再利用。own V0/A/Bの最小適合実モデル費用をLD1/LD2保存結果と比較 |

**全体上限：新signal13 round、fresh10 path、compile20、fit2、単価予測26、配分2、Q3最大3。** LD1/LD2の6結果・凍結価格は再利用し、再signal/再compile/再fit/再Q3は0。LD2長round、LD3/5/9、全round直接compileは含めない。モデル総額は後続でLD7 V0/A/Bの最大3件だけを求める。

較正各4 pathは完了したLD2有限pilotのcell規模を再利用し、holdoutの2 pathは独立した二つの価格を記述する小さい枠とした。母平均の必要精度やCIから導いたnではなく、精度保証は付けない。

最初の共通配分をV0にしたのは、既存LD1/LD2で評価済みの同一controlかつ同じcap/protocolを再利用するため。いまの最良B_PRICE列を新候補にも固定し続ける案ではない。第2段階以降は候補自身の3/4予算・価格でA/Bを選び直す。Bの現在のshot探索や未公開結果を待たず、この有限3点menuとして実行可能な設計である。

### 価格規則・独立性・外挿

`C_ref(7,m,a)=alpha[a]+beta[a] q[m]`（モデルoffset0、全13R=1を必要条件）。固定(1,q,qR)形のqR=qへの射影で、βはheadとtailと境界/optimizerの合成費用を表す。pure head単価、普遍event単価とは同定しない。source上で部品は共通でもhead/basis/tail分布・scalar・相殺はprefix依存するので、LD2係数は使わない。

m0/m6の各軸4 pathの算術平均から、既存d83737b6 `_solve`のexact二元一次算術を利用する。candidate IDを含むpoint-lawはcellごとに保持し、generic fitのlaw一致guardを緩めず、prefix7/R1 familyの薄いfit wrapperを固定する。同pathのSは診断として残すが、full C fitにS/残差を重ねて足さない。二点二係数の残差0は予測検証ではない。

m3価格holdoutは各軸n=2を保持し、O=(C0+C1)/2、凍結PからP−O、絶対値、相対誤差をexactで保存。O=0なら相対null、ZERO_OBSERVED_PRICE。1path欠測なら平均を1pathへ変更しない。価格の数値許容差・モデルPASS/FAILは設けず`NOT_ASSESSED_NO_ADOPTED_TOLERANCE`。差の大小でclip・feature変更・refitはしない。m7..12はq64を超える未検証外挿であり、初回から長round保証を求めない。

価格quantityはP0の測定付きfull controlled Hadamard wrapperのcompiled RZ/shot、Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、symmetric-directional、新ZZ。元normal samplerの出力をそのまま使い、order2等が出るまで引き直さない。X/Yは同pathの別軸で、path数を二倍にしない。cohort案・元trial_seed規則はPLANに書いたが、seed値は今回生成せず、次のconfiguration/physical IDが決まった後で固定する。

### 既存SOURCEを使う際の具体的差分

- metadataはa9abf8deのPREPARE_MENUのbodyとfefd548eの`b_metadata`を、元と同じB_RELEASE437ba4acのowned namespaceで再利用する。新prefixの入力sourceを明示し、LD2のconfiguration IDをそのまま置換しない。
- signal providerの`load_h4_context`/`make_context`/`evaluate`は保存descriptorのprefixを一般に受け取れる。旧LD1/LD2限定callerのloopとB_MEANS公開部はLD7用の新SOURCEが必要。既存の有限normalizer b>1・norm・phase・reference pair・closureのgateは維持する。
- 価格はe8273676の`candidate_price_adapter.from_b`→`NativeRuntime.context`→`Adapter.acquire`を使う。`ld12_price_requests.request_for`や既存callerの固定LABELSはLD1/2専用なので、名前だけLD7へ替えて呼ばない。generic from_bは1≤prefix<rankを扱える。実準備でhead[0..6]、tail[7..9]、λ/scalar/preparation・basis・H/stateを照合する。
- corrected total scalarの束縛とmatching axes/observer対応fastを維持する。元C1のprefix1固定contextへ戻さず、元raw/旧cacheの出所不明hitをfallback compileで埋めない。新SOURCEの実入口を人工prefix7/fake compilerで確認してから科学callを行う。元P2 fresh範囲外の新しい作用保証やfull-wrapper statevector PASSは付けない。
- Q3は005dc8c0の実worker/constructor/persistenceと固定27e70efe/710aa69b/9e294778を再利用する。Python3.11.0rc1の元AST guardを本番worker経路で確認し、期待hashを変えない。新prefix own configuration/signalへのmetadata intakeだけを適応し、decoder/target/loss/閾値を固定する。

signalは既存のhead/macro cacheとmatrix powerを用い、trajectoryや巨大event列を作らない。価格は独立path並列と元fast structural/compose/fingerprint部品を使う。X/Yは別compileし、holdoutを価格凍結前に開かない。資源の実績・進捗・終了codeを保存し、過去worker/CPU/RAMを恒常的上限にしない。新しい高速化実装やbenchmarkは本turnで行っていない。

## §10.5、STOP、ロードマップ

f42の旧catalog照合を再利用し、selected descriptorのhash・四分割の近い旧signal metadata、関係SOURCEと最近のremote差分だけを追加確認した。a50dce8fのPR-Pauli次分割案とe7d24c21の決定論p7 shot探索は別方式であり、PR-DF LD7の結果には流用しない。確認範囲で同等のcurrent PR-DF LD3/7完了result・進行jobは未同定。[DELTA_CHECK](DELTA_CHECK.json)に範囲を保存し、他担当job・worktreeは変更していない。

次回はこの計画以降に増えた同prefix configuration/physical/axis/law/SOURCE、signal packet、価格receipt/cohort/seed/lowering/wrapper/compiler、model/配分/Q3、進行jobとoutput/cacheを差分照合する。同等結果は出所付きで再利用し、判定不能なものは保留。旧T2枠や旧STOPを新規実行許可として流用しない。

後続のSTOPは、R1 domain不成立、入力/λ/scalar/identity/SOURCE不一致、signal normalizer/norm/phase/closure異常、V0未適合・不確定、receipt欠落・gate/保存失敗、rank/positive-price不成立。部分成果を保全して新規投入を止める。無断retry・seed差替え・追加path・負値clip・model/κ/Rの事後変更はしない。A/Bの不適合・INCONCLUSIVEは有効な有限結果で、別列を追加する許可にはしない。

ロードマップへの差分は「LD2長round確認を自動的な待ち依存にせず、次の分割選択ではLD7を先に提案」。旧6点比較・旧STOP・m4独立holdout・凍結価格は保全。LD3/5/9は未評価のまま残し、次の一候補の結果から選択を更新する。全law E[C]/CI・実PCG64 joint law・正式G/winner・全L_D最適性は未認定。ロードマップ本体は統合担当が更新する。

**今回の新candidate/schedule/signal/sampling/build/compile/fit/price/N/Q3/総額再計算・数値費用ランキングは全て0。** 科学moduleや保存H4 P0 operator回帰suiteを起動していない。計画をcommit・push、remote SHA/成果物一致を確認後mandatory STOP。有限案はまだ実行しない。
