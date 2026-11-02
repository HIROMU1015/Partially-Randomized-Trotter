# H4/H6 S4：counter互換性修正と固定7候補の再会計

**全3 Stage COMPLETE_MANDATORY_STOP。** counter/native間の8行の不一致を解消した。
修正版の84行・84行・360行を公開し、旧比較表はそのまま保存した。
理論対transpiled回路の残差と、H4主条件での優劣方向の不一致は修正後も残る。

[一つの資料入口・全JSON/CSV・監査](../../artifacts/resource_applicability/track_a_h46_s4_counter_native_compatibility/2026-10-18/README.md)。

## 固定範囲とSOURCE

利用者の添付指示がStage 1/2/3の実行を明示承認した。
Stageごとに独立したSOURCE closure・input binding・grant・output・lock・call ledgerを固定し、
正常終了・mandatory STOPと公開照合が成立した場合だけ次へ進んだ。
原grantはprivateに保持し、公開projectionは再利用可能なgrantではない。

branch：`track-a-h46-s4-counter-native-compatibility-20261018`。
起点：`cb501906269ef31fe8733a3391db946b34467163`。

|Stage|実行SOURCE|結果commit|
|---|---|---|
|1 修正・人工回帰|`f33d917f10c5f92f63b23776387cbd16e85abfd1`|`a139250850bd9eee313162454affbbd46e2fa6fa`|
|2 実整数受入|`f01fa5fbd9c508e468dac3cff8405f57370dbb51`|`fcb937a0c56ee75e2d5d32a9472357f3f346aaa8`|
|3 保存再会計|`da8eb64ba8e88c219a6b707e1c72d130f5f37fcb`|本報告を含む結果commit。最終チャットに40文字SHAとremote一致を報告|

元比較結果 `7e454b91185ed7b0f92a3790f800ba5d65d5f156`、元科学SOURCE
`536a3671c0ec2939ff3886e40467db26a27ec65a`、元native provider
`217c971dd537b1cc7fbc13742bc0f6c263ff7f45`を区別する。
GitHubから元counter・adapter・native schedule/係数・生成器のfile取得を確認した。
既存の実SOURCE/mirrorの区別も元provenanceのまま保持する。

H4はlinear 1Å/STO-3G/legacy DF12/8 modes/T=.8、full PF p12 q1 S2/S4と
PR p3 q1 r4 K2・p3 q8 r2 K4。
H6はlinear 1Å/STO-3G/tol-only DF19/12 modes/T=.8、full PF p19 q8 S4と
PR p10 q8 r1 K2・p15 q8 r2 K4。
全てsymmetric-directional、full Hadamard wrapper、P=0、state preparation除外。
異なるDF政策を一つの系サイズscalingへ統合しない。

モデルA/B/NATIVE_MIXED、cosine/sine、RZ/CXを維持した。
H4 PRは保存anchor n32、H6 PRは探索n8、PF n1。独立確認n32を混ぜていない。
5精度、allowance倍率1/10、bias/normalization/N_real/N_imag・seed/event digest・paired identity、
compiled RZ/CXと欠測分類を維持した。
compilerは保存Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、topologyなしのまま。
今回Qiskitを使用せず、新しいcompile出力の同一性を認定していない。

## Stage 1：最小patchと回帰

変更した科学コードは`native_df_gate_model.py:pf_schedule`内だけ。
`a=2**(1/3)`として、負係数を`-a/(2-a)`、stage時間を`T*weight/q/2`へ揃えた。
正・負・正の合成順、block/mode順、native/producer/strict-zero/Gaussian生成/
scalar/phase/wrapper/compiler/loweringは変更していない。
微小RZをepsilonで落とす修正はない。

新しいcounter schedule profileを`native_binary64_s4_v1`としてSOURCE hashへ結合した。
既存lowering profile `explicit_rz_sx_x_cx_v1`は変更していない。

追加9件＋既存counter27件＋既存比較adapter32件、**計68 tests PASS、失敗/skip0**。
係数・時間hexの一致、q1/3/8、正負・signed zero、S2不変、block/mode、support・phase・
scalar・wrapper、非finite/不正q/formula拒否、旧人工fixtureの差0を確認した。
AST検査で`pf_schedule`以外の全counter定義が不変と確認した。
Stage 1のnative exit0・wait/reap・source/environment gate・STOPを保存した。

## Stage 2：実H4/H6の完全一致

固定preparationのH4 codec 1,450,047 bytes/SHA
`f9618a6fda0c28bccfc415f95079ad137e846de12fbe2acaf74456d8369a2c48`、
H6 archive 5,670,237 bytes/SHA
`ad52f9f6a3150981d7be00191d8b3a9f9af992c37c9102cba6d94fcffc8e3010`を前後照合した。
typed DAGのnode hash、preparation、fragment順、diagonal/lambda/scalar・basis記述を再利用した。
live native回路やHamiltonian/DF/stateを再生成していない。

|系|軸|修正前理論RZ|修正後RZ＝保存native|修正前理論CX|修正後CX＝保存native|
|---|---|---:|---:|---:|---:|
|H4|cosine|48,272|48,274|21,444|21,446|
|H4|sine|48,273|48,275|21,444|21,446|
|H6|cosine|1,521,189|1,521,173|667,264|667,248|
|H6|sine|1,521,190|1,521,174|667,264|667,248|

**8行全て整数で完全一致**。33 blockの修正後負時間とprimitiveのRZ全hex・RZZ mask/hash・
phase全hexも、前回の保存native側の証拠と一致した。
旧model/nativeの二者を比較した66評価は再実行していない。
今回のfull counter評価には、正時間33・修正済み負時間33の元producer値を一度ずつ取得し、
model・forward/reverse・q反復・axis間でcacheした。
PFのAとNATIVE_MIXEDは定義上同じためA値を再利用した。
2候補のA/Bを並列処理し、両軸の費用は整数加法性によりwrapper差だけを加えた。
浮動小数点のprimitive演算順は変更しない。

不一致の原因はcounter scheduleのnative binary64互換性であり、Qiskit処理前に解消した。
元native生成器や共有producerを誤りと判定したものではない。
整数count一致を全wordのbit完全同一性やsignal精度の認定とはしない。

## Stage 3：A/B/mixed全体への影響と再会計

S4は新SOURCEのStage 2で得たA/B/mixed値を、同じcounter hashに結合して使用した。
S2/PRはStage 1のコード不変性・S2 hex一致と既存row identityに基づき再利用し、
新しい数値変化を拒否した。PRのlaw/DPやeventを再生成していない。
全70条件の保存shot・eligibility、compiled費用は元値と一致した。

以下はcosineのcount。sineのRZは各+1、CXは同じ。

|系|model|旧RZ→新RZ|旧CX→新CX|
|---|---|---:|---:|
|H4|A/NATIVE_MIXED|48,272→48,274|21,444→21,446|
|H4|B|45,216→45,218|20,452→20,454|
|H6|A/NATIVE_MIXED|1,521,189→1,521,173|667,264→667,248|
|H6|B|1,452,789→1,452,773|644,840→644,824|

修正は旧mixed8行だけに限定されず、**A/B/mixed全て**のS4行へ及ぶ。
期待値対保存平均は84行中24行、same-eventは84行中24行、PR/PF比は360行中168行が変化した。
S2/PRの各表60 metric行は数値不変、欠測・eligibilityの変更はない。
全528行の旧値/新値対応と数値差をJSON/CSVへ保存した。

ρ=C_compiled/C_theory、e=(C_compiled−C_theory)/C_compiledを従来定義のまま出力した。
期待値のlower/upperを保持し、中点に置き換えず、信頼区間とも扱わない。
same-eventのNATIVE_MIXEDは全28 metric行でnative explicit lowered数と一致した。
一方、compiled/theory比はこの固定panelで約**0.448721〜0.638070**となり、残差は残る。
これは明示理論loweringとQiskit lowering＋opt1の合成差であり、opt0の追加compileで寄与を分離していない。

### H4での優劣方向不一致

H4 PR p3/q8/r2/K4対full PF p12/q1/S4、ε=.05、allowance倍率1、P=0、保存shot使用。

|metric|旧理論PR/PF|修正後理論PR/PF|保存compiled PR/PF|
|---|---:|---:|---:|
|RZ|0.8716682443|0.8716321313|1.0393357766|
|CX|0.8125591095|0.8124833323|0.9520900533|

RZでは理論がPRを安価とし、compiledではPFが安価という方向不一致が**残る**。
今回のbinary64修正でこの解釈が反転したわけではない。
全360行中192行が共通適格で、旧→新の優劣方向変化は**0件**。
理論対compiledの方向不一致は6行で、同じH4/RZ条件を3model×2allowance倍率で表した相関する行である。
NATIVE_MIXED/RZ/倍率1の主表では従来どおり1条件。6独立実験の結果とはしない。

## 終了・資源・保全

全Stageでnative exit0・wait/reap・最終source/input/選択runtime file gate・STOPを保存した。
Stage 2の2 workerもnative exit0・wait/reap。科学retry0。

|Stage|本体wall秒|coordinator peak RSS bytes|call|
|---|---:|---:|---|
|1|0.731384|58,728,448|人工suite 3、実科学0|
|2|9.923508|45,613,056|候補2、unique primitive66|
|3|0.404010|51,519,488|保存再会計1、primitive0|

RSSはcoordinatorの値で、Stage 2 worker peakを測った値ではない。
いずれもcompile時間・メモリではない。Stage 2だけ独立2候補を並列化し、それ以外は短い1 process。
内部thread1、数値演算順は維持し、重複するprimitive・preparation・compiled結果を再利用した。
各起動前にavailable CPU affinity・RAM・disk/inodeを記録し、共有環境・既存venv/package・
OS/scheduler/cgroup/他job・他worktreeを変更しなかった。全作業/log/一時fileはhome配下。

旧grant/消費・予約/marker/STOPの返却・reset・流用は0。
private raw、元grant、認証情報をcommitしていない。
公開時のstage対象file指定の誤記は訂正したが、科学Stageの失敗・再実行やSOURCEの事後変更はない。
primary agentで実施し、独立別担当review・GPT/Claude詳細科学レビューは開始していない。

## 解決したこと、残ること、次の担当

解決したのは、有限精度S4 scheduleによるcounter/native間の8行のcount不一致と、
その修正をA/B/mixed・同じ固定7候補の理論資源比較へ正確に反映する技術課題である。

残るのは、理論対transpiled費用の構成依存残差、全word bit同一性、旧compiler出力の完全同一性、
PR母平均・未観測希少次数のcompiled費用上界、固定panel外の最適性などである。
保存有限標本や経験的allowanceを保証へ格上げせず、一般的なPR優位や研究全体の採否は確定しない。
旧UNATTRIBUTED分類を含む歴史的結果は書き換えず、新しい解決証拠を別結果として保存した。

次の担当：**ユーザー**が修正版表をどの原稿・主張へ反映するか判断する。
Codexは別branchへのmerge・追加sampling/compile/fit・候補再最適化・追加科学レビューを開始しない。
全Stageの公開確認後、**最終mandatory STOP**。
