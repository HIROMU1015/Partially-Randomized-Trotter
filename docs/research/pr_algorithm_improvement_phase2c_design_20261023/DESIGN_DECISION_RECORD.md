# Phase 2C設計判断記録

日付：2026-10-23 JST。設計判断の記録であり、既存研究の科学的statusや旧STOPを変更しない。

## D01：採択済み方針を固定する

ユーザーのPhase 2C指示により、RQ-01表現とRQ-03生成の統合方針は採択済み。RQ-02は必要な補助だけ、RQ-04～07は保留。主線の再選定・70研究の再調査はしない。指示で明示されたPhase 2B方針を基準とし、途中でユーザーが指定したローカルPhase 2Bレビュー原本を読んで照合した。原本の「採択待ち」はレビュー時点の状態であり、その後のユーザーの採択を取り消すものではない。

原本は設計前の新規性完全証明を要求せず、既知法の有用なPR接続を新原理と区別して評価する。最大費用の未測定を設計禁止にしない一方、basis／childの支配性を仮定しない。この二点を主案のclaimと最小計画へ反映した。private review原本は公開しない。

## D02：変更対象はfragment別の残差辞書だけ

normだけの最適化は、回路単価や準備費で負ける既存固定例を説明できない。一方、軌道・DF rank・split・sampler・nativeを同時最適化すると因果と取得費が不明になる。既存F_lのS/P二択に限定し、D/R、q/r/K、scalarを固定した。新しいsplit selectorの研究へ転じていない。

この変更はchild rotationにも作用する。長いparentがほぼ出ない場合にも作用し得るため、旧RAの稀な部分だけを縮める案とは異なる。ただし実費の占有率や利得を取得していない。

## D03：候補生成と選択を手順にする

保存低次数fermion係数からP mapを得て、fragment間で同一wordをcollectする。全S／全利用可能Pの二端点を含め、stable順一passの反転を最大二回行う。選択にはcanonical normalizationを必ず戻し、既存run共有を与えたliteral期待価格を使う。Φは実compiler予測と呼ばず、粗い価格で十分かも未確定とした。

価格計算の有限state伝搬は評価手段であり、新しい量子生成原理やP-Aの区間DP再開ではない。追加pass、precision探索、価格pilot、sampled近似への自動拡張は含めない。

## D04：既知の構成要素を新法の根拠にしない

原PR §VII.Bの残差Pauli化／異軌道involutionと、既存toy `Primitive` の異種basis接口を確認した。cost-aware lexicon／greedyも近接先行研究にある。混在辞書だけを今回の独立方法差とする案は採らなかった。

最終分類は `KNOWN_METHOD_APPLICATION_CANDIDATE`、独立方法差 `INDEPENDENT_METHOD_DELTA_UNESTABLISHED`、現時点の独立新法claimは `NO_DEFENSIBLE_NEW_METHOD_YET`。完全な同一手順への還元も未確定。具体的な有限PR価格selectorが既知構成の通常適用を越えるか、安い実装上の価値を持つかを、明示した比較で判断する。世界的新規性の保証はしない。

## D05：採らなかった着想

共通左frame Vだけの変更は相対basis積を簡単にしない。相対Gaussian合成のみは既知compilerの呼出しであり、独立主案としない。既存interval DP、Gaussian積生成、BM、RA、Parent allocationを別名で再開しない。cost-aware ISや多round配分は別課題として保留。代案を同時設計する必要性は認めず、主案一つだけを成果物とした。

## D06：sector条件を明示する

P展開は全体Rの粒子数保存を保持しても、各atomのsector保存は保証しない。full registerでのtarget meanと、sector内だけのprimitive実行を区別した。途中投影でmeanを変えず、符号空間が不変でない場合はPを不適格とする。最小fixtureは1粒子だけの退化を避けた4mode/2粒子とし、full registerのoperator検査を提案した。これは数値検証結果ではない。

## D07：設計完了と科学的採択を分ける

9必須資料、入出力、手順、意味論、未解決条件、F1/F2、出典、公開identityを揃えることが今回の完了条件。新実装や数値を作らない。既存研究のnormative文書・manifestは、新しい受理済み科学結果がないため更新しない。新namespaceだけに追加し、公開後STOPする。

次の判断者はユーザー。独立新法としての追加設計、既知法のPR接続としてのF1、または方向の変更を選ぶ。通常の技術確認に詳細GPTレビューを一律要求しない。新しい中心claimや非自明な保証を正式判断する必要が生じた場合のみ、必要なレビュー範囲と別承認を定める。
