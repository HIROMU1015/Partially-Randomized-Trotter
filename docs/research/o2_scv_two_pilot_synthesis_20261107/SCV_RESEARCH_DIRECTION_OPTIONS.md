# SCVの継続・一区切り・一時停止の選択肢

2026-11-07 JST。**判断材料とCodexの技術的提案。研究方針・科学的GO/NO-GOの正式採択ではない（NOT_ADOPTED）。重要科学レビューも開始していない。**
対象は保存linear H4／1.00 Å／STO-3G／legacy DF12／全4電子sector・8qubits／L_D3／m0,q1,δ=.02,r1,K2／coefficient_atol0、N1 actual Q/selected full、N2保証ground、認証済みexact Fraction engine、理想gate・X/Y。[統合証拠](SCV_TWO_PILOT_EVIDENCE.md)と[原因の確度](SCV_NONIMPROVEMENT_HYPOTHESES.md)を前提とする。

## 第一提案

**現2種類の具体的controlの検証を一区切りとし（A）、SCVの追加科学計算はいったん保留する（D）案を推奨する。** 理由は、2件で学習費込み利益の明確な材料が得られず、同候補の自動増量では新しい問いが定まらないため。これは資源配分に関する技術的提案で、SCV全体の無効性や正式終了の判断ではない。

将来の再開候補にはProduct一致（C）がある。別の代数構造を問えるが、希少な活動層・有料trainという費用上の弱点がある。ユーザーがその問いの価値を認めた場合、まず[最小実装・人工確認とmetadata-only一回の別scope](NEXT_MINIMAL_VALIDATION_DRAFT.md)へ進み、fresh量子shotの承認を分ける。未検証であることだけを実行理由にしない。

## 四方向の比較

| 基準 | A：現2候補を一区切り | B：同controlの限定追加検証 | C：別構造control | D：SCV探索全体を一時停止 |
|---|---|---|---|---|
| 既存証拠の支持 | rank/range・実装は完了、学習費込み点利益なし。直接の根拠がある | 不確かさは残るが、単なる増量を積極的に支持する材料は乏しい | 完全label一致ならP_l²=I。実H4相関・費用利益はUNKNOWN | 今の追加費用に対する情報価値が不明という保留根拠 |
| 得られる新情報 | 新scienceなし。既得成果の境界を確定 | 凍結λの効果の評価誤差、または再学習の係数品質。二つは別の問い | h2/h3のproducts構造をbitから学べるか | 当面scienceなし。将来候補の機会を残す |
| 測定・実装 | 追加shot/実装不要 | fixed-model fresh評価なら接続差分小。再trainなら全学習費・独立性を再設計 | mean/bankとh2/h3 model差分、人工確認、metadata-only、必要ならfresh評価 | 追加shot/実装不要 |
| 学習費・量子RZ | 新規費用なし。旧費用は保全 | frozen deploymentでも過去費用を併記。再学習費無料化禁止 | 希少層でもconditional trainはfull価格。局所利益の全体への重みが弱点 | 追加費用を抑え、他課題の優先度を検討できる |
| 再利用性 | 文書・law/engine/managerの実績を保持 | 既存のdriver核を使いやすいが旧STOP解除は不可 | sampler/OLS/価格器は再利用、h1-only型と固定quotaは不適合 | 独立性・保存・会計・型を将来にも利用可能 |
| 有限confidence接続 | 現在は未実施のまま | 効果の根拠が出れば別途cal/整数N/価格保証 | same。同じrange/cal/floorの障害が残る | 未完了事項を明示して保留 |
| PR＋RPE全体への寄与 | 小規模負の材料を固定し、無目的な追加を避ける | 明確な費用回収の問いを解決できる場合に限定 | 別情報の可能性。ただしm0だけで全round優位性を示さない | 最終目的への寄与が不明な探索への投入を留保 |
| 判断を変える証拠 | 新しい説明情報・経済的な検証目的 | fresh評価の再現性/不確かさ、あるいは有限train損失を問う事前計画 | exact mean/非定数性・全体重みの理由、さらに独立bitによる説明力 | 再開候補の具体的な情報価値と実行可能性 |

Aは今の具体的controlだけを区切り、Cなどの検討は残す。Dは別候補を含むSCVの新規探索全体を当面保留する。両者を同義にせず、ユーザーが採否を決める。今回、repositoryの科学的status・旧契約・旧STOPを変更しない。

## 同じcontrolの追加が判断を変えるための条件

| 限定目的 | 何を固定し何を変えるか | 結果が変える判断 | 答えられないこと |
|---|---|---|---|
| 現モデルの評価不確かさ | control/保存λを固定、新しい独立評価だけ | 現凍結モデルを追加利用する価値 | 不正確なλの修復、別群、学習policyの期待性能 |
| 学習誤差の寄与 | 同controlを固定し、別の独立train設計とfresh評価 | 再学習へ資源を使う価値 | 一fitからoracle/母相関や全policyを確定すること |
| 保証付きdeploy費用 | 上記の根拠に加え、独立cal・対称な整数配分と価格根拠 | 同一ε/α・返却条件での十分費用比較 | m0一条件から全RPE優位性を確定すること |

実行前に問い・結果の利用先・有限scopeを定められない場合は保留を維持する。既存評価を見て都合のよい追加shot、旧bitでの再fit、range緩和は行わない。新しいH4 break-even数値は今回算出していない。

## 全体研究・他作業との関係

[研究目的の正本][purpose]は有限RTE・誤差/失敗配分・full compiled costを全round RPEへ接続すること。SCVはその一候補であり、現pilotは有限signal診断に留まる。全roundのbias/半径/位相関係とconfidence・期待価格の接続は別課題である。

Exact Fraction engine高速化は古典simulation時間を減らし得る別の技術方向として比較できる。しかし量子RZの1shot価格やmetadataの相関を改善するものではない。O1/O2の実装・計算・worktree/process/cacheへ今回干渉せず、その成果・速度を新しく評価しない。GPU/floatへの置換も採択しない。

## 次の担当と承認境界

1. **ユーザー**：A/B/C/Dの採否を判断する。Codexは正式採択を代行しない。今すぐ追加科学計算する必要はない。
2. **SERVER1 Codex（Cを選ぶ場合）**：新control実装・人工correctnessの限定指示後、既存SOURCEへ最小adapterを別branchで作り、SOURCEを公開固定する。今回は未実装。
3. **ユーザーによる別の明示実行許可**：固定N1 bytesからProduct一致のmean/momentを一回取得するmetadata-only scope。その結果後もSTOP。fresh bitは含めない。
4. **さらに科学的に必要と判断した場合**：独立train/paired評価の新SOURCE、shot数・range・stream/seed・全費用・STOPを別途採択・許可する。metadataのELIGIBLEだけで開始しない。

重要な科学レビューが必要かの判断と、実際の開始も区別する。本作業で詳細レビューを開催したという記録は作らない。

**現時点の保留提案、将来の候補、正式方針の採択は別。** 本資料の公開照合後mandatory STOPし、旧224/448の停止、新campaign/実装/再解析の未許可状態を維持する。

[purpose]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a7b70ffb6a39a3af4dbbf438f35f959e9cafe22e/docs/research/研究目的・研究課題.md
