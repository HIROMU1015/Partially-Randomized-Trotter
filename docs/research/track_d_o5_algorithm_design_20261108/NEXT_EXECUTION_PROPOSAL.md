# 次の最小作業提案

提案のみ。新しい実装・人工test・保存価格再解析・実H4計算・契約採択は今回行っていない。q64別担当の完了を催促・変更・停止せず、必要なら公開固定結果の引渡しを待つ。本書はq64 jobの継続許可やO2未採択H4契約の代替ではない。

## 優先1：Aの「追加余地」を先に判定する小さなselector接口

**次の担当：ユーザー/GPTが採否を判断し、採択後のCodexが最小実装・人工検証。**

最初に固定するのはprimary資源または多目的/幅制約と、比較対象の三つ（常時strong、常時Reference、事前固定round-only）。CX目的とRZ目的の結論を混ぜず、path-awareがround-onlyを上回るかを問う。

技術作業案は、既存descriptorから必要な少数特徴だけを渡すmethod decision接口と、片方だけを遅延生成するdispatch/cache記録。StrongBuilder/reference_gadget/whole-pathを複製しない。既存prepare_stepsは両方式生成なので、新selectorをその後ろへ置いて安価な事前選択と称しない。既存tuple-returning builderを壊さず、必要なら選択後の一arm構築を薄く分離する案を検討する。

人工入力で、同じomegaを保つこと、bit/信号/真価格を入力にできないこと、fallback・17-qubit制約、positive pathを落とさないこと、axis/profile/identity/cache分離、事後compile価格をfeatureへ漏らさないことを確認する。今回その実装/人工testは許可されていないため開始しない。

**期待する新証拠：** 選択の情報が実行前に取得でき、oracleを使わず、既存builder/測定targetを保つ小さな接口として実装できること。ここでは資源改善の数値を主張しない。

**棄却できる仮説：** 安価なdescriptor選択だと思っていたものが、実際には両方式のcompileや実matrix評価を必要とするなら、その「安価な事前選択」案を棄却または有料compile選択という別案へ戻す。

その後に効果を調べるなら、保存価格/metadataを使う**新しい科学的再解析**への明示許可が必要。元88 pathは設計資料なので独立holdoutにしない。存在/取得可能性を確認した固定feature/価格cohortで選択余地・round-onlyとの差・誤選択損失を探索し、モデル/閾値/主要資源を凍結した上で、必要な場合だけ別の独立有限価格検証へ進む。新しい平均/min/順位/fitを今回作らない。

科学段階に進む場合に限り、使用SOURCE、raw metadataの可用性、固定input、学習/評価分離、call上限、compile費/cache償却、STOPを一つの実行契約へ固定する。現時点で全13-round再会計・新N・q≥128は不要。q64の進行中出力を探索データや未使用holdoutとして取り込まない。

負の結果は有用である。paired差の符号が全domainで同じ、q-onlyと同等、または選択/学習/compile費が利益を消すならpath-aware selectorを追加しない。常時Referenceまたは単純mixedという既知運用へ戻す案を提示する。科学的GO/NO-GOの正式判断はユーザー/GPTへ戻す。

## 優先2：Bの近傍complete-step cut

**次の担当：Aの選別結果を受けたユーザー/GPT、採択後のCodex。**

descriptorだけのprefix/suffix run proxyと、中央/事前固定近傍一cutを扱う小さな人工計画を提案する。既存prepare_stepsのcomplete internal-step cut接口、physical_segment、actual inverse、B6b恒等式とC3/C5 checkerを再利用する。別gadgetを新設しない。

期待する証拠は、同じordered V・chi・全入力X/Yを保ち、単純なrun均衡がfull-wrapper depthにも反映されるか。まず人工的に不均衡/均衡とfusion損失の反例を比較する技術許可が必要。その後、実H4のrun分類・cut選択・build/compileを行う場合は別の限定科学実行許可が必要。対象q/cutやcall数は候補採択後にだけ固定する。

中央で十分均衡、近傍cutのP0/closure overheadが相殺、またはrun proxyがcritical pathを説明しなければ、この候補を縮小する。新cutの全RPE/全support保証や最大cut探索は初回に要求しない。

## 優先3：C1の未融合head-tail接口

**次の担当：ユーザー/GPTによるO1/O2/O3担当境界の確認後、Codexの局所技術作業。**

新しいevent並替えやsampling法ではなく、明示的に未実装のhead-tail接口を一種類に限定する。StrongBuilder.transition、tail emitter、mixed control mode、scalar/gauge処理を再利用し、strong/Reference双方に同じ合法最適化を与える。既存step融合の再実装はしない。

期待する証拠は、接口で残っているexact同値の書換えと、P0が既に行う消去との差。最初は対応する人工局所caseのみで、技術実装・人工compileへの許可を別途得る。実分子での接口分類や新価格比較は科学実行許可が必要。P0が全利益を既に回収、phase/branch条件不成立、両方式への同じ改善で差が残らないなら候補を終える。

Law/PF/event順序変更を伴うC2はこの提案の外に置き、現行固定変数へ黙って追加しない。

## 判断順序と今回のSTOP

1. ユーザー/GPTが、独立新機構を求める研究か、既知手法のPR総資源への有用な適用を狙う研究かを明確にする。
2. Aの安価な接口/追加余地を先に選別する。大きなselectorや学習pipelineを先に作らない。
3. Aが単純mixed以上の価値を持たなければBの局所depth案を検討し、C1は接口が具体化した場合だけ進める。
4. 三方向とも追加効果がなければ、Reference適用成果を固定し、O1/O2/O3と同じ精度・成功率の総資源最適化へ重点を戻す案を採るか判断する。

今回は文書のみ公開してmandatory STOP。SOURCE/入力/call数は実際に進める候補の次回契約で固定し、候補ごとの大量の未採択契約を作らない。
