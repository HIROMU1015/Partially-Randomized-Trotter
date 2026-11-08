# Track D（O5）：改善候補の静的調査報告

2026-11-08。**結論案B：改善候補はあるが、新規性と追加効果は不明。** A（compile前のControlled/Reference選択）を最初の小さな技術選別候補とし、B（近傍complete-step cut）、C1（target-preserving head-tail接口）を次順位に置く。新しい研究方針・主資源・精度・契約の正式採択は行わない。

主な所見：

- Track Dのstrongは既にdirectional control、relative/head融合、tail同basis共有、合法step境界融合を持つ。O2のdiagonal_only対照より強く、O2人工価格をそのまま比較できない。両Referenceは既知SEと同じ基本機構で、gadgetの再実装は不要。
- 確認したTrack Dの入口は両方式を構築/会計し、compile前selectorは確認できなかった。ただしround/powerによる混在は既知SEの範囲。独立新機構は確認していない。
- 保存報告のCX差は全88 pathで208−224qであり、有限範囲ではpath選択余地が乏しい可能性がある。RZの短q/長qの差はround-onlyでも利用可能。Aの追加価値は、両方式をcompileせずに取得できるPR特徴で、単純mixedより安く選べるかにある。効果の数値は未取得。
- C7-4のbasis-run/lane-runは価格説明の探索候補で、勝敗を予測する保証ではない。compiled DAG endpointや保存価格の事後minを無料featureにしない。
- Bはlane imbalanceとcutのclosure/fusion損失を同時に扱うdepth中心の候補。C1はSOURCEに未融合と明記されたhead-tail接口に限定し、PF/law/event変更のC2から分離する。どちらもcount/depth改善の証拠はまだない。
- pathごとに適格な同じobservableを使い、同じQ・fresh independent trials・誤差/半径/decoder条件を保持するideal laneでは、方式混在でも同じround/axis別十分Nを使い得る。actualの数値誤差・law/joint independence・hardware/reset/routingは別資格で、旧Nを自動流用しない。

## 成果物

| 資料 | 内容 |
|---|---|
| [CURRENT_ALGORITHM_MAP](CURRENT_ALGORITHM_MAP.md) | strong/Track D/既知SE/O2の構造・SOURCE・domain差 |
| [EXISTING_EVIDENCE](EXISTING_EVIDENCE.md) | 固定測定同値性、人工/実H4、qモデル、条件付き13-round、q64とO2未採択契約の境界 |
| [IMPROVEMENT_CANDIDATES](IMPROVEMENT_CANDIDATES.md) | A/B/Cの情報、数学条件、費用、否定条件と担当境界 |
| [PRIOR_ART_BOUNDARY](PRIOR_ART_BOUNDARY.md) | 既知mixed/SE/融合との重複。限定一次文献確認のみ |
| [NEXT_EXECUTION_PROPOSAL](NEXT_EXECUTION_PROPOSAL.md) | 最小技術選別→必要時だけ科学検証という次段階案 |

## 調査範囲・保全

基点は固定b1e7cc5b6270e6e62ab57b7c889f37484ce7d7b7。B6b 7ffe8b28、C7-4 64072ba8、B/C会計8828b298、q64修正a708a130と限定PLAN b1e7cc5b、O2 0edc0fa5/781e155dの資料とSOURCEを実際に読んだ。基点にない資料は固定Git object/公開linkで参照し、旧ファイルをmerge/上書きしなかった。

実H4数値を引用するdomainはLinear H4／隣接1.00 Å／STO-3G／Nalpha=Nbeta=2／元H／number-CDF8／元frame順L_D2／S2/s1/r32/K0/h=.1／準備費除外／P0 opt1。O2想定のlegacy DF12/L_D3/.02/r1/K2/opt2とは別。数値引用のscopeは証拠表に明記した。

既存13-round結果は条件付きproxy会計として尊重し、最終的な保証付き総資源優位としない。新しい平均/min/係数/順位・N・総費用は計算していない。新builder/optimizer/人工test/環境構築/科学sampling/build/compile/価格再解析/詳細科学レビューはすべて0。

q64は別担当の管理下にある。固定PLAN以外の進捗照会・完了認定、controller/worker/CPU/memory/入力/SOURCE/output/STOPへの操作は行わない。O2の未採択H4契約も実行しない。元mainと既存dirty/untracked、旧SOURCE/raw/結果/grant/ledger/marker/STOP、他TrackとC1凍結を保全する。

## 次の担当・mandatory STOP

**次の担当はユーザー/GPT。** Aの小さなselector接口と人工検証を次段階に採択するか、改善余地の薄さを踏まえBへ移るかを判断する。保存H4価格の新たな再解析・新fit・独立価格検証には、対象候補を選んだ後の別途明示許可が必要。今回の設計を実装/科学計算の許可としない。

専用branch codex/track-d-o5-design-20261108 で文書をcommit/pushし、remote SHA/tree/公開本文を照合する。公開照合完了後mandatory STOP。実装・科学計算・q64介入・別候補探索・重要レビューへ自動進行しない。
