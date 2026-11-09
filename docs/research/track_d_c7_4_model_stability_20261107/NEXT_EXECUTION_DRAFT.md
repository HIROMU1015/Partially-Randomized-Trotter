# C7-4：次の独立長q検証条件案（未採択・未実行）

本書はユーザー/GPTの研究判断用。C7-4の承認をq≥64のH4実行へ流用しない。下記はすべて提案であり、sampling/回路生成/compileは未実施。

## 推奨domainと選定理由

条件付き第一候補は**q64/cut32、fresh8 whole-path**。自動採択ではない。q64は既存最大q32の2倍で、basis-runの全P0価格移送、左右lane揺らぎ、CX整数候補の外挿を一つの独立domainで検査できる。q128はモデル間の分岐が広がりやすいが、q32比4倍の長さと数値余裕の不足を先に解決すべきため第一候補にしない。q256〜4096の直接compileは、現時点では数値精度/計算可能性/全path bridgeが不足し採択しない。q64に通っても次qへ自動進行しない。

q32 observable残差は既存閾値の約60%であり、q64 numerical GOは未成立。C7-4の人工native比較は精度改善を示さなかった。従って下記の先頭path preflightを全cohortの一部として扱う実行案をユーザー/GPTが承認するか、先に限定的な数値経路の改善検証を追加するか判断する。

## モデル・estimandの事前固定

- 元C6a凍結版を比較用に保全。C7-4全q affine（2係数）、sqrt版（3係数）、basis-run版（3係数）、lane-run版（3係数）のspec/係数/hashをsampling前に固定する。新q結果で再fitしない。
- Paired countsの簡潔な候補は全q affine、CXは208−224qの整数残差を別に保存。depthの有限データではtotal-run版がpairedを、strongはtotal-run/Referenceはlane-runが絶対価格を説明した。価格精度の要求は未固定なので、これらを新規独立domainで並列評価する候補とし、都合よく有利な方式/資源だけを採択しない。
- 共通designを使う各既存版は、方式別の差と直接paired fitが整合する。方式ごとに異なるdesignを選び組み合わせる場合は別モデル扱いとし、method間・cross-model係数共分散を持つ一貫した仕様が追加で必要。現候補の共分散を独立に足して済ませない。
- 評価対象は6資源×XYのpaired12量を一次候補、方式別絶対24量を別familyの診断候補。qubitsは9/17の幅制約で別記。主資源指標/最終energy精度/成功確率はユーザー/GPTの研究判断とする。
- run/lane特徴は保存した新event descriptorから**価格compile前**に抽出する。pathごとにmaxを取る。feature条件付きpath予測とlawで平均した期待価格を区別し、未知のjoint feature lawを勝手に置かない。

## 最小の科学call案

|対象|提案する固定上限|条件|
|---|---|---|
|whole-path|8 path、各q64/cut32|新domain/sample ID/seedを事前固定、既使用864 occurrence seedと分離|
|canonical occurrence|512、events16384|追加samplingなし、イベントを先に保存/hash固定|
|raw correctness|8 path、各一回|既存CAR/AB/actual adjoint/vacuumχ/phase/全256入力observable|
|full P0 wrapper|最大32|strong/Reference×XY×8、各path raw PASS時のみ|
|postcheck|同32 wrapper|CAR/raw/pairedXY/χ/reference leakage、固定閾値|
|追加A/B/shell compile、別q、calibration、N/総費用|0|次判断なしの拡張禁止|

最初に固定sample00のrawと4 wrapper/postcheckを行う。これは32 wrapper上限に含む。数値不成立ならsample00を除外・再抽選せず、依存する残処理をSTOPする。全8本が適格に揃わなければ母平均をcomplete-caseで作らない。先頭preflightを成功扱いへ変えるためのphase fit/閾値緩和/列削減/Reference leakage無視はしない。

保存X/Y価格が一致した有限観測だけを根拠にY compileを省略しない。P0全体のaxis価格等価性・局所テンプレートの閉包がSOURCEで別途保証できた場合のみ、科学call削減の別案を採択する。

## 標本数と幅・古典的信頼度の候補

n8は独立domainの**探索的技術pilot**であり、必要母平均標本数の保証ではない。既存n/SDはqごとに保持し、X/Yをn16に数えない。CXの観測SD0も母分散0ではない。

価格幅の候補は事前strong予測規模Sに対するη=1+0.05*S、古典nominal α=.05、paired12量と絶対24量を別familyとする。これは既存探索規則と比較するための案で、採択値/確認的coverageではない。母平均精度が目的なら、事前のvariance上界またはtail/support根拠とbias/law予算を用い、例えばnormal proxy n≈ceil(z²*s(q)²/η²)を設計診断として比較する。q32のSDを√2倍/2倍してq64へ移送する感度案は作れるが、いずれも仮定で厳密上界ではない。現在の88標本だけから保証付き必要nは確定できない。

bootstrap校正共分散は同じ係数を使う複数qで共有される。prediction側と独立holdout側の不確かさ・model biasを別にし、同じ観測をモデル選択と独立検証に二重使用しない。

## Numerical preflight・高速化・GO/STOP

同H4/CDF8/LD2/S2/s1/r32/K0/h=.1、同P0と現行component1e−10/observable2e−10/rtol0/phase fitなし。元H厳密ground仮定・準備費除外。C3/C5/StrongBuilder/CAR cache/C7-3 exact conversion・分離compile/post dispatch・既存batch state-actionを再利用する。C7-4 native adapterを改善済みverifierとして自動選択しない。SOURCE/入力/seed/cut/位相/actual adjointの一致が必須。

実空きCPU/RAMとquota・共有非干渉を確認し、独立raw/compile/postを適切に並列化する。過去のworker数/CPU/RAMを恒常上限にせず、共有OS/venv/schedulerや他jobは変更しない。各工程の完了memory観測と実MemAvailableに基づくadmissionを使う。

SOURCE/law不一致、公平なstrong baseline不成立、raw/post不一致・閾値超過、位相/χ/Reference leakage不成立、保護原本破損、継続不能な資源不足なら該当依存処理をSTOP。価格残差/価格幅外は科学的に有効な結果として保存し、correctness failureとは分ける。無断retry/fallback・モデル再fit・幅拡張・追加samplingなし。結果公開・remote SHA/tree/bytes照合後mandatory STOP。

## 全13-round費用へ進む前の残課題

q1のcut/価格境界と新q≥64の適用域、q4096までの期待価格/bias/分散/共有係数共分散、全supportのP0と丸め誤差、finite-bit law/joint-law、元Hに対する共通N/actual RPE成功資格が必要。B4-2の条件付き14,738をactual Nとして流用せず、B5の未資格を残す。モデル値×Nの全13-round数値集計は本案で行わない。routing/T/noise/実機時間も別の対象。

次の担当はユーザー/GPT：数値経路の追加技術検証を先にするか、上記の停止条件付きq64 pilotを承認するかを判断する。Codexは採択scopeの新しい明示的許可後だけ担当する。
